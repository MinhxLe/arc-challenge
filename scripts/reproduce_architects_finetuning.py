import os

from arc.config import all_configs
from arc.core import Task

from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

from datasets import load_from_disk, Dataset
import typing as ta
from arc.external.architects import (
    InputMaskingDataCollator,
    preprocess_model_tokenizer_formatter,
)


def load_and_process_dataset(
    get_dataset: ta.Callable[[], Dataset], path: str, use_cache: bool
) -> Dataset:
    if use_cache and os.path.exists(path):
        dataset = load_from_disk(path)
    else:
        dataset = (
            get_dataset().map(format_row, num_proc=24).filter(not_too_long, num_proc=24)
        )
        dataset.save_to_disk(path)
    return dataset  # type: ignore


# from model_tools import (
#     save_model_and_tokenizer,
# )
# from model_tools import load_peft_state, merge_peft_into_base

# change this to take command line!
fine_tuning_config = next(
    config for config in all_configs if config.name == "architects"
)

# for action in ["train", "merge"]:
#     # continue if task already accomplished
#     if action == "train" and os.path.exists(f"{save_model_path}-lora"):
#         continue
#     if action == "merge" and os.path.exists(f"{save_model_path}-merged"):
#         continue

# load base model & reduce embedding size
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=fine_tuning_config.model_config.model,
    dtype=fine_tuning_config.model_config.model_dtype,
    load_in_4bit=fine_tuning_config.model_config.load_in_4bit,
)

model, tokenizer, formatter = preprocess_model_tokenizer_formatter(model, tokenizer)

# create lora model
model = FastLanguageModel.get_peft_model(
    model=model,
    target_modules=fine_tuning_config.lora_config.target_modules,
    r=fine_tuning_config.lora_config.lora_rank,
    lora_alpha=fine_tuning_config.lora_config.lora_alpha,
    lora_dropout=fine_tuning_config.lora_config.lora_dropout,
    bias=fine_tuning_config.lora_config.bias,
    use_gradient_checkpointing="unsloth",
    random_state=fine_tuning_config.lora_config.random_state,
    use_rslora=fine_tuning_config.lora_config.use_rslora,
    loftq_config=fine_tuning_config.lora_config.loftq_config,
)

# Set up data


def format_row(row):
    task = Task.from_dict(row)
    row.pop("train")
    row.pop("test")
    return formatter.format_task_for_sft(task)


def not_too_long(row):
    return (
        len(tokenizer.tokenize(row["text"]))
        <= fine_tuning_config.sftt_config.max_seq_length
    )


use_cache = fine_tuning_config.data_config.use_cache
train_dataset_path = os.path.join(
    fine_tuning_config.output_dir, "fine_tuning_data/train/"
)
eval_dataset_path = os.path.join(
    fine_tuning_config.output_dir, "fine_tuning_data/eval/"
)


train_dataset = load_and_process_dataset(
    get_dataset=fine_tuning_config.data_config.get_train_dataset,
    path=train_dataset_path,
    use_cache=use_cache,
)

eval_dataset = load_and_process_dataset(
    get_dataset=fine_tuning_config.data_config.get_eval_dataset,
    path=eval_dataset_path,
    use_cache=use_cache,
)

# run training
FastLanguageModel.for_training(model)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field=fine_tuning_config.sftt_config.dataset_text_field,
    max_seq_length=fine_tuning_config.sftt_config.max_seq_length,
    data_collator=InputMaskingDataCollator(
        instruction_template=formatter.input_head_token,
        response_template=formatter.output_head_token,
        mlm=False,
        tokenizer=tokenizer,
        mask_first_n_examples=1,
    ),
    args=TrainingArguments(
        run_name=fine_tuning_config.output_dir,
        per_device_train_batch_size=fine_tuning_config.sftt_config.per_device_train_batch_size,
        per_device_eval_batch_size=fine_tuning_config.sftt_config.per_device_eval_batch_size,
        gradient_accumulation_steps=fine_tuning_config.sftt_config.gradient_accumulation_steps,
        warmup_ratio=fine_tuning_config.sftt_config.warmup_ratio,
        num_train_epochs=fine_tuning_config.sftt_config.num_train_epochs,
        learning_rate=fine_tuning_config.sftt_config.learning_rate,
        embedding_learning_rate=fine_tuning_config.sftt_config.embedding_learning_rate,
        eval_strategy="steps",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=fine_tuning_config.sftt_config.logging_steps,
        optim=fine_tuning_config.sftt_config.optimizer,
        weight_decay=fine_tuning_config.sftt_config.weight_decay,
        lr_scheduler_type=fine_tuning_config.sftt_config.lr_scheduler_type,
        seed=fine_tuning_config.sftt_config.random_state,
        output_dir=fine_tuning_config.output_dir,
        resume_from_checkpoint=True,
        save_strategy=fine_tuning_config.sftt_config.save_strategy,
        save_steps=fine_tuning_config.sftt_config.save_steps,
        save_total_limit=fine_tuning_config.sftt_config.save_total_limit,
        report_to=fine_tuning_config.sftt_config.report_to,
    ),
)
trainer_stats = unsloth_train(trainer)
# save_model_and_tokenizer(f"{save_model_path}-lora", model, tokenizer)
#
# if action == "merge":
#     # load peft weights and merge
#     load_peft_state(model, f"{save_model_path}-lora")
#     model = merge_peft_into_base(model)
#     save_model_and_tokenizer(f"{save_model_path}-merged", model, tokenizer)
