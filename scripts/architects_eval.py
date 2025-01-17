from arc.config import all_configs
from arc.external.architects import (
    load_model_tokenizer_formatter,
    get_peft_model_with_lora,
    InputMaskingDataCollator,
    save_model_and_tokenizer,
)
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
from unsloth import FastLanguageModel, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainer as Trainer
from unsloth import UnslothTrainingArguments as TrainingArguments
import numpy as np
import math
from arc.core import Task, Grid, Example
from arc.transform import (
    Identity,
    Rotate,
    Reflect,
    Compose,
    Transform,
    PermuteColor,
    generate_train_only_tasks,
)
from arc.datasets.seed import Datasets
from arc.datasets import transform as dst
from datasets import Dataset
import random

from tqdm import tqdm
import pickle as pkl
from arc import settings
from datetime import datetime
import os

EVAL_TMP_SAVE_FILE = (
    "/shared/research/arc_challenge/runs/arc_public_eval_2025-01-10.pkl"
)

torch.set_default_device(
    "cuda"
) if torch.cuda.is_available() else torch.set_default_device("cpu")


fine_tuning_config = next(
    config for config in all_configs if config.name == "architects"
)


def _dedupe_np_arrays(arr_list: list[np.ndarray]):
    result = []
    for arr in arr_list:
        if not any(np.array_equal(arr, x) for x in result):
            result.append(arr)
    return result


def _apply_transform_to_task(
    task: Task,
    transform: Transform,
    input_only: bool = False,
    shuffle_train_seed: int | None = None,
) -> Task:
    new_train = []
    for example in task.train_set:
        input_ = transform.apply(example.input_)
        if input_only:
            output = example.output
        else:
            output = transform.apply(example.output)
        new_train.append(Example(input_=input_, output=output))

    if shuffle_train_seed:
        random.seed(shuffle_train_seed)
        random.shuffle(new_train)

    new_test = []
    for example in task.test_set:
        input_ = transform.apply(example.input_)
        # TODO: handle the case of nonexistent test output (true test case)
        if input_only:
            output = example.output
        else:
            output = transform.apply(example.output)
        new_test.append(Example(input_=input_, output=output))

    return Task(id=None, train_set=new_train, test_set=new_test)


# architects use 16
# we could add some color permutations here
TRANSFORMS: List[Transform] = [
    Identity(),
    Rotate(-1),
    Rotate(-2),
    Rotate(1),
    Reflect(Reflect.Type.HORIZONTAL),
    Reflect(Reflect.Type.VERTICAL),
    Reflect(Reflect.Type.DIAGONAL),
    Compose(
        transforms=[Reflect(Reflect.Type.DIAGONAL), Reflect(Reflect.Type.HORIZONTAL)]
    ),
]


@dataclass
class TaskEvaluation:
    task: Task
    solutions: List[Grid]
    exception: Optional[Exception] = None

    def success(self):
        if self.exception is None:
            for solution in self.solutions:
                if np.array_equal(self.task.test_set[0].output, solution):
                    return True
        return False


class SolutionGenerator:
    def __init__(self, peft_checkpoint_path: str):
        """Initialize the inference module.
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        self.peft_checkpoint_path = peft_checkpoint_path
        self._load_model()

    def _load_model(self) -> None:
        """Load the fine-tuned model from checkpoint."""

        model, tokenizer, formatter = load_model_tokenizer_formatter(
            fine_tuning_config, self.peft_checkpoint_path
        )

        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter

        logger.info("Model loaded successfully")

    def _prepare_model_for_finetuning(self) -> None:
        self.model = get_peft_model_with_lora(self.model, fine_tuning_config)
        FastLanguageModel.for_training(self.model)

    def _prepare_ttt_dataset(self, dataset: Dataset) -> Dataset:
        def format_row(row):
            task = Task.from_dict(row)
            row.pop("train")
            row.pop("test")
            return self.formatter.format_task_for_sft(task)

        def not_too_long(row):
            return (
                len(self.tokenizer.tokenize(row["text"]))
                <= fine_tuning_config.sftt_config.max_seq_length
            )

        base_ttt_dataset = Dataset.from_list(
            [
                ttt_task.to_dict()
                for row in dataset
                for ttt_task in generate_train_only_tasks(Task.from_dict(row))
            ]
        )

        transformed_ttt_dataset = dst.concat(
            base_ttt_dataset,
            dst.apply_transform(base_ttt_dataset, Reflect(Reflect.Type.DIAGONAL)),
            *[dst.apply_transform(base_ttt_dataset, Rotate(i)) for i in range(4)],
            dst.apply_transform(base_ttt_dataset, PermuteColor(seed=42)),
        )

        return (
            dst.concat(
                transformed_ttt_dataset,
                dst.shuffle_train_order(transformed_ttt_dataset, seed=42),
            )
            .map(format_row)
            .filter(not_too_long)
        )

    def run_ttt(self, dataset: Dataset, run_name: str) -> None:
        ttt_dataset = self._prepare_ttt_dataset(dataset)
        self._prepare_model_for_finetuning()

        save_path = f"{(settings.TEMP_ROOT_DIR)}/runs/{run_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=ttt_dataset,
            # eval_dataset=eval_dataset,
            dataset_text_field=fine_tuning_config.sftt_config.dataset_text_field,
            max_seq_length=fine_tuning_config.sftt_config.max_seq_length,
            data_collator=InputMaskingDataCollator(
                instruction_template=self.formatter.input_head_token,
                response_template=self.formatter.output_head_token,
                mlm=False,
                tokenizer=self.tokenizer,
                mask_first_n_examples=0,
            ),
            args=TrainingArguments(
                run_name=run_name,
                per_device_train_batch_size=4,
                # per_device_eval_batch_size=fine_tuning_config.sftt_config.per_device_eval_batch_size,
                gradient_accumulation_steps=1,
                warmup_ratio=0.25,
                num_train_epochs=1,
                learning_rate=1e-4,
                embedding_learning_rate=1e-5,
                # eval_strategy="steps",
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=500,
                optim="adamw_8bit",
                weight_decay=0.00,
                lr_scheduler_type="cosine",
                seed=42,
                output_dir=save_path,
                resume_from_checkpoint=True,
                save_strategy="steps",
                save_steps=1000,
                save_total_limit=10,
                report_to="wandb",
            ),
        )
        _ = unsloth_train(trainer)
        store_path = os.path.join(save_path, "final_ttt_model")
        save_model_and_tokenizer(
            store_path=store_path,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info(
            f"TTT for {run_name} complete, model ready, final model saved at {store_path}"
        )

    def _score_candidate(self, task: Task, candidate: Grid) -> float:
        transform_log_probs = []
        for transform in TRANSFORMS:
            transformed_task = _apply_transform_to_task(task, transform)
            transformed_candidate = transform.apply(candidate)
            transform_log_probs.append(
                self._calculate_candidate_log_prob(
                    transformed_task, transformed_candidate
                )
            )
        return sum(transform_log_probs)

    def solve_task(self, task: Task, num_solutions: int = 1) -> list[Grid]:
        candidates = self._get_candidates(
            task=task,
            response_log_prob_threshold=math.log(0.10),
        )
        # scoring candidates
        scores = [self._score_candidate(task, c) for c in candidates]
        candidate_and_scores = list(zip(candidates, scores))
        candidate_and_scores.sort(reverse=True, key=lambda x: x[1])
        return [c[0] for c in candidate_and_scores[:num_solutions]]

    def _get_candidates(
        self,
        task: Task,
        response_log_prob_threshold: float = math.log(0.1),
        max_tokens: int = 10000,
    ) -> List[Grid]:
        """Generate candidate responses using depth-first search with log probability threshold.

        Args:
            task: Task
            response_log_prob_threshold: Minimum cumulative log probability threshold
            max_tokens: Maximum length of response

        Returns:
            List of CandidateResponse objects
        """

        # Optimizations:
        # KV caching
        # Parallelize DFS and merge results - might be able to batch eval
        # Postpone decoding until the end (don't encode/decode in the next_token function) - but need to also handle attention mask

        FastLanguageModel.for_inference(self.model)

        # Note: this function uses mutation of variables outside its scope.
        # We could change this to return candidates instead... I think.
        def dfs(
            current_text: str,
            current_log_prob: float,
            depth: int,
            candidates: List[Grid],
            token_log_probs: List[float],
        ) -> None:
            if depth >= max_tokens:
                return

            next_allowable_tokens = self._get_next_tokens_above_threshold(
                prompt=current_text,
                log_prob_threshold=response_log_prob_threshold - current_log_prob,
            )

            for token, log_prob in next_allowable_tokens.items():
                new_text = current_text + token
                new_log_prob = current_log_prob + log_prob
                new_token_log_probs = token_log_probs + [log_prob]
                assert depth + 1 == len(
                    new_token_log_probs
                ), f"depth: {depth}, length: {len(new_token_log_probs)}"

                if token == self.tokenizer.eos_token:
                    try:
                        candidates.append(
                            self.formatter.parse_test_output_grid(new_text)
                        )
                    except ValueError:
                        # assuming this is due to malformed grid
                        pass
                else:
                    # Continue DFS
                    dfs(
                        current_text=new_text,
                        current_log_prob=new_log_prob,
                        depth=depth + 1,
                        candidates=candidates,
                        token_log_probs=new_token_log_probs,
                    )

        all_candidates = []
        for transform in TRANSFORMS:
            candidates = []
            transformed_task = _apply_transform_to_task(task, transform)
            prompt = self.formatter.format_task(
                transformed_task, include_test_output=False
            )
            dfs(
                current_text=prompt,
                current_log_prob=0.0,
                depth=0,
                candidates=candidates,
                token_log_probs=[],
            )
            all_candidates.extend([transform.inverse.apply(c) for c in candidates])
        return _dedupe_np_arrays(all_candidates)

    def _calculate_candidate_log_prob(self, task: Task, candidate: Grid) -> float:
        return self._get_response_log_prob(
            prompt=self.formatter.format_task(task, include_test_output=False),
            response=self.formatter._format_output(candidate),
        )

    def _get_response_log_prob(self, prompt: str, response: str) -> float:
        FastLanguageModel.for_inference(self.model)

        full_text = f"{prompt}{response}"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt")
        response_ids = full_ids[:, input_ids.shape[1] :]

        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits

        response_logits = logits[:, (input_ids.shape[1] - 1) : -1, :]
        log_probs = F.log_softmax(response_logits, dim=-1)

        response_token_log_probs = torch.gather(
            log_probs, 2, response_ids.unsqueeze(-1)
        ).squeeze(-1)

        return torch.sum(response_token_log_probs).item()

    def _get_next_tokens_above_threshold(
        self, prompt: str, log_prob_threshold: float
    ) -> Dict[str, float]:
        """Get log probabilities for all possible next tokens
           with log probability of occurrence >= threshold.

        Args:
            prompt: Input prompt text
            log_prob_threshold: minimum acceptable log probability

        Returns:
            Dictionary mapping token text to log probability
        """

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)[0]

        indices_above_threshold = torch.where(log_probs >= log_prob_threshold)[0]

        # Convert to dictionary of token: log_prob
        token_log_prob_dict = {}
        for idx in indices_above_threshold:
            token_log_prob_dict[self.tokenizer.decode([idx])] = log_probs[idx].item()

        return token_log_prob_dict


def run_test():
    sg = SolutionGenerator(
        "/shared/research/arc_challenge/runs/architects_copy_2024-12-26_keepers/checkpoint-30000/"
    )
    task = Task.from_dict(Datasets.arc_public_train.get_dataset()[0])
    return sg.solve_task(task, num_solutions=10)


def run_evaluation():
    sg = SolutionGenerator(
        "/shared/research/arc_challenge/runs/architects_copy_2024-12-26_keepers/checkpoint-30000/"
    )
    eval_set = dst.concat(
        Datasets.arc_public_train.get_dataset(), Datasets.arc_public_test.get_dataset()
    )
    task_evaluations = []

    for dataset_task in tqdm(eval_set):
        task = Task.from_dict(dataset_task)
        try:
            solutions = sg.solve_task(task, num_solutions=2)
            task_evaluations.append(TaskEvaluation(task=task, solutions=solutions))
        except Exception as e:
            task_evaluations.append(
                TaskEvaluation(
                    task=task,
                    solutions=[],
                    exception=e,
                )
            )

        with open(EVAL_TMP_SAVE_FILE, "wb") as file:
            pkl.dump(task_evaluations, file)


def evaluation_metrics():
    with open(EVAL_TMP_SAVE_FILE, "rb") as file:
        task_evaluations = pkl.load(file)

    successes = []
    for evaluation in task_evaluations:
        successes.append(evaluation.success())

    logger.info(f"Tasks evaluated: {len(task_evaluations)}")
    logger.info(f"Tasks solved: {sum(successes)}")
    logger.info(
        f"Tasks with solution attempts: {sum([len(e.solutions)>0 for e in task_evaluations])}"
    )
    logger.info(
        f"Tasks with exceptions: {sum([e.exception is not None for e in task_evaluations])}"
    )


def run_ttt_small():
    eval_set = Datasets.arc_public_test.get_dataset()

    failures_20_random = [
        150,
        224,
        135,
        236,
        168,
        153,
        102,
        182,
        137,
        257,
        193,
        299,
        307,
        218,
        173,
        231,
        217,
        198,
        6,
        145,
    ]

    small_ttt_eval = Dataset.from_list([eval_set[i] for i in failures_20_random])

    sg = SolutionGenerator(
        "/shared/research/arc_challenge/runs/architects_copy_2024-12-26_keepers/checkpoint-30000/"
    )

    sg.run_ttt(small_ttt_eval, "small_ttt")
