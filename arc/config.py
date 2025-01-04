from dataclasses import dataclass
from datasets import Dataset
import torch
import typing as ta
from arc.datasets.seed import Datasets
from arc.datasets import transform as dst
from arc.tokenizers import Formatter
from arc import transform as t
from unsloth import FastLanguageModel

from tokenizers import Tokenizer
from transformers import DataCollatorForLanguageModeling
from arc import settings
from datetime import datetime


from arc.external.architects import (
    keep_single_char_tokens,
    InputMaskingDataCollator,
)


@dataclass
class FineTuningModelConfig:
    model: str
    model_dtype: torch.dtype | None
    load_in_4bit: bool

    def __post_init__(self):
        if self.model_dtype and self.load_in_4bit:
            raise ValueError("expected model_dtype to be None if load_in_4bit")


@dataclass
class FineTuningDataConfig:
    get_train_dataset: ta.Callable[[], Dataset]
    get_eval_dataset: ta.Callable[[], Dataset]
    use_cache: bool = True


@dataclass
class FineTuningLoraConfig:
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    target_modules: list[str]
    random_state: int
    bias: str = "none"
    loftq_config = None

    # unused
    # use_gradient_checkpointing=True, setting to True in outer

    def __post_init__(self):
        if not 0.0 <= self.lora_dropout <= 1.0:
            raise ValueError(
                f"lora_dropout must be between 0.0 and 1.0, got {self.lora_dropout}"
            )


@dataclass
class FineTuningSFTTConfig:
    random_state: int | None
    dataset_text_field: str = "text"
    data_collator_constructor: ta.Optional[
        ta.Callable[[Tokenizer, Formatter], DataCollatorForLanguageModeling]
    ] = None
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    embedding_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    optimizer: str = "adamw_torch_fused"
    save_strategy: str = "steps"
    save_steps: int = 1_000
    save_total_limit: int = 10
    report_to: str = "wandb"

    def __post_init__(self):
        if self.embedding_learning_rate >= self.learning_rate:
            raise ValueError(
                f"embedding_learning_rate ({self.embedding_learning_rate}) must be < learning_rate ({self.learning_rate})"
            )


@dataclass
class FineTuningConfig:
    name: str

    model_config: FineTuningModelConfig
    model_tokenizer_formatter_preprocessor: ta.Callable[
        [FastLanguageModel, Tokenizer],
        ta.Tuple[FastLanguageModel, Tokenizer, Formatter],
    ]

    data_config: FineTuningDataConfig
    lora_config: FineTuningLoraConfig
    sftt_config: FineTuningSFTTConfig

    # not used
    # packing=False

    def __post_init__(self):
        if self.model_config.load_in_4bit and self.lora_config.loftq_config:
            raise ValueError(
                "expected loftq_config to be None if load_in_4bit because quantization already in base model"
            )

        self.sftt_config.random_state = (
            self.sftt_config.random_state or self.lora_config.random_state
        )

    @property
    def output_dir(self) -> str:
        return f"{(settings.TEMP_ROOT_DIR)}/runs/{self.name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"


##### architects config


def get_architects_train_data() -> Dataset:
    base_train_dataset = dst.concat(
        dst.repeat(Datasets.concept_arc.get_dataset(), n=128),
        dst.repeat(Datasets.arc_public_train.get_dataset(), n=128),
        # [TODO] change to n_tasks=644
        Datasets.create_re_arc(
            seed=42, n_tasks=100, test_set_size=1, train_set_size=5
        ).get_dataset(),
    )
    transformed_train_dataset = dst.concat(
        base_train_dataset,
        dst.apply_transform(base_train_dataset, t.Reflect(t.Reflect.Type.DIAGONAL)),
        *[dst.apply_transform(base_train_dataset, t.Rotate(i)) for i in range(4)],
        dst.apply_transform(base_train_dataset, t.PermuteColor(seed=42)),
    )
    return dst.concat(
        transformed_train_dataset,
        dst.shuffle_train_order(transformed_train_dataset, seed=42),
    )


def get_architects_eval_data() -> Dataset:
    return Datasets.arc_public_test.get_dataset()


def architects_data_collator_constructor(
    tokenizer, formatter
) -> InputMaskingDataCollator:
    return InputMaskingDataCollator(
        instruction_template=formatter.input_head_token,
        response_template=formatter.output_head_token,
        mlm=False,
        tokenizer=tokenizer,
        mask_first_n_examples=1,
    )


def architects_model_tokenizer_formatter_preprocessor(model, tokenizer):
    keep_tok = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-="
    ) + tokenizer.tokenize("\n")

    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)
    tokenizer.padding = "right"

    return model, tokenizer, Formatter(output_tail_token=tokenizer.eos_token)


architects_config = FineTuningConfig(
    name="architects",
    model_config=FineTuningModelConfig(
        model="chuanli11/Llama-3.2-3B-Instruct-uncensored",  # "nvidia/Mistral-NeMo-Minitron-8B-Base",
        model_dtype=None,
        load_in_4bit=True,
    ),
    model_tokenizer_formatter_preprocessor=architects_model_tokenizer_formatter_preprocessor,
    data_config=FineTuningDataConfig(
        get_train_dataset=get_architects_train_data,
        get_eval_dataset=get_architects_eval_data,
    ),
    lora_config=FineTuningLoraConfig(
        lora_rank=256,
        lora_alpha=24,
        lora_dropout=0,
        use_rslora=True,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        random_state=42,
    ),
    sftt_config=FineTuningSFTTConfig(
        random_state=42,
        max_seq_length=8192,
        data_collator_constructor=architects_data_collator_constructor,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_ratio=0.25,
        num_train_epochs=1,
        learning_rate=1e-4,
        embedding_learning_rate=1e-5,
        weight_decay=0,
        lr_scheduler_type="cosine",
        logging_steps=500,
        optimizer="adamw_8bit",
    ),
)

#####

all_configs = [architects_config]
