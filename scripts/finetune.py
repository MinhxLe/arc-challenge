import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# DATASET_FNAME = f"{settings.DATA_DIR}/arxiv-metadata-processed.json"
# MODEL_DIR = f"{settings.MODEL_DIR}/llama_1b_lora_ft"


# loading in models
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

inputs = tokenizer(
    "you are an expert python programmer. write a fibonacci function in python",
    return_tensors="pt",
).to(model.device)
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        num_return_sequences=1,
        temperature=0.9,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode and return the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     r=32,
#     lora_alpha=16,
#     lora_dropout=0.1,
# )
# model = get_peft_model(model, peft_config)
#
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id
#
#
# # preparing FT dataset
# def get_dataset(tokenizer, split="train") -> Dataset:
#     def process_rows(rows):
#         processed = rows["formatted"]
#         return {"text": [t + tokenizer.eos_token for t in processed]}
#
#     dataset = load_dataset("json", data_files=DATASET_FNAME, split=split)
#     return dataset.map(process_rows, batched=True)
#
#
# dataset = get_dataset(tokenizer)
#
#
# # Configure training arguments
# training_args = SFTConfig(
#     output_dir=MODEL_DIR,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     warmup_steps=5,
#     learning_rate=2e-4,
#     fp16=True,  # Enable mixed precision training
#     logging_steps=1,
#     save_steps=10,
#     save_total_limit=2,  # Only keep the last 2 checkpoints
#     # remove_unused_columns=True,
#     report_to="wandb",
#     optim="adamw_torch",
#     max_steps=200,
#     seed=42,
# )
#
# # Initialize trainer
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=512,  # Adjust based on your GPU memory
#     packing=True,  # Enable dataset packing for efficiency
# )
#
# # Start training
# train_stats = trainer.train()
#
# # Save the final model
# trainer.save_model(MODEL_DIR + "/final_model")
#
# # Optional: Push to Hugging Face Hub
# # trainer.push_to_hub("your-username/model-name")
