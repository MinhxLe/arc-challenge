from dataclasses import dataclass
import functools
import torch


@dataclass
class FineTuningConfig:
    name: str

    model: str
    model_dtype: torch.dtype | None
    load_in_4bit: bool

    dataset_name: str
    dataset_split: str

    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    peft_random_state: int
    sftt_random_state: int | None

    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    optimizer: str = "adamw_torch_fused"
    save_strategy: str = "epoch"
    report_to: str = "wandb"

    def __post_init__(self):
        if self.lora_alpha > self.lora_rank:
            raise ValueError(
                f"lora_alpha ({self.lora_alpha}) must be <= lora_rank ({self.lora_rank})"
            )

        if not 0.0 <= self.lora_dropout <= 1.0:
            raise ValueError(
                f"lora_dropout must be between 0.0 and 1.0, got {self.lora_dropout}"
            )

        self.sftt_random_state = self.sftt_random_state or self.peft_random_state

    @functools.cached_property
    def output_dir(self) -> str:
        return f"tmp/runs/{self.name}"
