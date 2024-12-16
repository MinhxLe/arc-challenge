"""
TODO:
- logging to wandb
- checkpointing
- logging stdout
"""

from transformers import TrainingArguments
from trl import SFTTrainer
from arc import settings
import torch
from unsloth import FastLanguageModel
from datetime import datetime
from datasets import load_dataset

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"llama_3_1_8b_lora_barc_finetune_{timestamp}"
output_dir = f"tmp/runs/{run_name}"

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    dtype=torch.bfloat16,
    token=settings.HF_API_TOKEN,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing=True,
    random_state=3407,
)

dataset = load_dataset(
    "barc0/induction_100k_gpt4o-mini_generated_problems_seed100.jsonl_messages_format_0.3",
    split="train_sft",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # dataset_text_field="text",
    # dataset_num_proc=2,
    # packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=0,
        num_train_epochs=3,  # Set this for 1 full training run.
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_torch_fused",
        save_strategy="epoch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="wandb",
    ),
)
trainer.train()
