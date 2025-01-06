from arc.config import all_configs
from arc.external.architects import (
    preprocess_model_tokenizer_formatter,
    get_and_fix_peft_weights,
    fix_dtypes,
)
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass
from loguru import logger
from unsloth import FastLanguageModel
import peft
import numpy as np
import math

fine_tuning_config = next(
    config for config in all_configs if config.name == "architects"
)


@dataclass
class SolutionCandidate:
    """Data class to store candidate responses and their log-probabilities."""

    text: str
    log_probability: float


class SolutionGenerator:
    def __init__(self, peft_checkpoint_path: str, device: str = "cuda"):
        """Initialize the inference module.
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = None
        self.tokenizer = None
        self.peft_checkpoint_path = peft_checkpoint_path

    def load_model(self) -> None:
        """Load the fine-tuned model from checkpoint."""
        try:
            logger.info(f"Loading model from {self.peft_checkpoint_path}")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=fine_tuning_config.model_config.model,
                dtype=fine_tuning_config.model_config.model_dtype,
                load_in_4bit=fine_tuning_config.model_config.load_in_4bit,
            )

            model, tokenizer, formatter = preprocess_model_tokenizer_formatter(
                model, tokenizer
            )

            model = peft.PeftModel.from_pretrained(
                model=model,
                model_id=self.peft_checkpoint_path,
                device_map="cuda",
            )

            weight_set_result = peft.set_peft_model_state_dict(
                model,
                get_and_fix_peft_weights(self.peft_checkpoint_path),
            )
            assert (
                not weight_set_result.unexpected_keys
            ), "error loading weights - some keys not available in model"

            self.model = fix_dtypes(model.merge_and_unload())
            self.tokenizer = tokenizer
            self.formatter = formatter

            FastLanguageModel.for_inference(self.model)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_full_response(
        self, prompt: str, max_new_tokens: int = 10000
    ) -> Tuple[str, float]:
        """Generate a complete response for the given prompt with its probability.

        Args:
            prompt: Input prompt text
            max_length: Maximum length of the generated response

        Returns:
            Tuple of (generated_text, probability)
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not initialized")

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate with logits
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Get generated text
            input_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs.sequences[0][input_length:], skip_special_tokens=True
            )

            # Calculate sequence probability
            sequence_probs = []
            for logits in outputs.scores:
                probs = F.softmax(logits, dim=-1)
                # Get probability of selected token
                token_prob = probs[0, torch.argmax(probs[0])].item()
                sequence_probs.append(token_prob)

            total_probability = np.exp(np.sum(np.log(sequence_probs)))

            return response, total_probability

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def get_next_tokens_above_threshold(
        self, prompt: str, log_probability_threshold: float
    ) -> Dict[str, float]:
        """Get probabilities for all possible next tokens
           with probability of occurrence >= threshold.

        Args:
            prompt: Input prompt text
            log_probability_threshold: minimum acceptable probability

        Returns:
            Dictionary mapping token text to log probability
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not initialized")

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)[0]

            indices_above_threshold = torch.where(
                log_probs >= log_probability_threshold
            )[0]

            # Convert to dictionary of token: log_probability
            token_log_prob_dict = {}
            for idx in indices_above_threshold:
                token_log_prob_dict[self.tokenizer.decode([idx])] = log_probs[
                    idx
                ].item()

            return token_log_prob_dict

        except Exception as e:
            logger.error(f"Error getting token distribution: {str(e)}")
            raise

    def validate_response_is_grid(self, response: str) -> bool:
        # TODO(Sid): implement something like return self.formatter.validate_is_grid(response)
        return True

    def get_candidate_responses(
        self,
        prompt: str,
        response_probability_threshold: float = 0.1,
        max_tokens: int = 10000,
    ) -> List[SolutionCandidate]:
        """Generate candidate responses using depth-first search with probability threshold.

        Args:
            prompt: Input prompt text
            response_probability_threshold: Minimum cumulative probability threshold
            max_tokens: Maximum length of response

        Returns:
            List of CandidateResponse objects
        """

        # Optimizations:
        # KV caching
        # Parallelize DFS and merge results - might be able to batch eval
        # Postpone decoding until the end (don't decode in the next_token function), maybe batch it?

        log_response_prob_threshold = math.log(response_probability_threshold)

        # Note: this function uses mutation of variables outside its scope.
        # We could change this to return candidates instead... I think.
        def dfs(
            current_text: str,
            current_log_prob: float,
            depth: int,
            candidates: List[SolutionCandidate],
        ) -> None:
            if depth >= max_tokens:
                return

            next_allowable_tokens = self.get_next_tokens_above_threshold(
                prompt=current_text,
                log_probability_threshold=log_response_prob_threshold
                - current_log_prob,
            )

            for token, log_prob in next_allowable_tokens.items():
                new_text = current_text + token
                new_log_prob = current_log_prob + log_prob

                if token == self.tokenizer.eos_token:
                    if self.validate_response_is_grid(new_text):
                        candidates.append(
                            SolutionCandidate(
                                text=new_text,
                                log_probability=new_log_prob,
                            )
                        )
                else:
                    # Continue DFS
                    dfs(
                        current_text=new_text,
                        current_log_prob=new_log_prob,
                        depth=depth + 1,
                        candidates=candidates,
                    )

        try:
            candidates = []
            dfs(
                current_text=prompt,
                current_log_prob=0.0,
                depth=0,
                candidates=candidates,
            )

            return candidates

        except Exception as e:
            logger.error(f"Error generating candidate responses: {str(e)}")
            raise
