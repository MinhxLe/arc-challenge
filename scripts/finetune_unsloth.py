"""
TODO:
- logging to wandb
- checkpointing
- logging stdout
"""

from arc import settings
import torch
from unsloth import FastLanguageModel


model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    dtype=torch.float16,
    token=settings.HF_API_TOKEN,
)
