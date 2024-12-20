from dataclasses import dataclass
import functools
import torch
import typing as ta
from datasets import Dataset


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
        if self.use_rslora:
            if self.lora_alpha > (self.lora_rank) ** (1 / 2):
                raise ValueError(
                    f"lora_alpha ({self.lora_alpha}) must be <= sqrt(lora_rank) ({(self.lora_rank)**(1/2)})"
                )
        else:
            if self.lora_alpha > self.lora_rank:
                raise ValueError(
                    f"lora_alpha ({self.lora_alpha}) must be <= lora_rank ({self.lora_rank})"
                )

        if not 0.0 <= self.lora_dropout <= 1.0:
            raise ValueError(
                f"lora_dropout must be between 0.0 and 1.0, got {self.lora_dropout}"
            )


@dataclass
class FineTuningSFTTConfig:
    random_state: int | None
    dataset_text_field: str = "text"
    data_collator: ta.Optional[ta.Callable] = None
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
    lora_config: FineTuningLoraConfig
    sftt_config: FineTuningSFTTConfig

    # not used
    # dataset_text_field="text"
    # max_seq_length=fmt_opts['max_tokens'] # only used if packing=True I think
    # packing=False

    def __post_init__(self):
        if self.model_config.load_in_4bit and self.lora_config.loftq_config:
            raise ValueError(
                "expected loftq_config to be None if load_in_4bit because quantization already in base model"
            )

        self.sftt_config.random_state = (
            self.sftt_config.random_state or self.lora_config.random_state
        )

    @functools.cached_property
    def output_dir(self) -> str:
        return f"tmp/runs/{self.name}"
