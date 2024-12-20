from dataclasses import dataclass
import torch
import typing as ta
from datasets import Dataset, load_dataset

from tokenizers import Tokenizer
from transformers import DataCollatorForLanguageModeling
from arc import settings
from datetime import datetime


from arc.architects import (
    fmt_opts,
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
        ta.Callable[[Tokenizer], DataCollatorForLanguageModeling]
    ] = None
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    embedding_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    optimizer: str = "adamw_torch_fused"
    save_strategy: str = "epoch"
    report_to: str = "wandb"

    def __post_init__(self):
        if self.embedding_learning_rate >= self.learning_rate:
            raise ValueError(
                f"embedding_learning_rate ({self.embedding_learning_rate}) must be < learning_rate ({self.learning_rate})"
            )


@dataclass
class FineTuningConfig:
    name: str

    data_loader: ta.Callable[[], Dataset]

    model_config: FineTuningModelConfig
    model_and_tokenizer_preprocessor: ta.Callable

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


def architects_data_loader() -> Dataset:
    return (
        load_dataset(
            "barc0/induction_100k_gpt4o-mini_generated_problems_seed100.jsonl_messages_format_0.3",
            split="train_sft",
        ),
    )


def architects_data_collator_constructor(tokenizer) -> InputMaskingDataCollator:
    return InputMaskingDataCollator(
        instruction_template=fmt_opts["query_beg"],
        response_template=fmt_opts["reply_beg"],
        mlm=False,
        tokenizer=tokenizer,
        mask_first_n_examples=1,
    )


def architects_model_and_tokenizer_preprocessor(model, tokenizer):
    keep_tok = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-="
    ) + tokenizer.tokenize("\n")

    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)
    tokenizer.padding = "right"

    return model, tokenizer


architects_config = FineTuningConfig(
    name="architects",
    model_config=FineTuningModelConfig(
        model="nvidia/Mistral-NeMo-Minitron-8B-Base",
        model_dtype=None,
        load_in_4bit=True,
    ),
    model_and_tokenizer_preprocessor=architects_model_and_tokenizer_preprocessor,
    data_loader=architects_data_loader,
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
        max_seq_length=fmt_opts["max_tokens"],
        data_collator_constructor=architects_data_collator_constructor,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_ratio=0.25,
        warmup_steps=0,
        num_train_epochs=1,
        learning_rate=1e-4,
        embedding_learning_rate=1e-5,
        weight_decay=0,
        lr_scheduler_type="cosine",
        logging_steps=10,
        optimizer="adamw_8bit",
    ),
)

#####

all_configs = [architects_config]
