# import os

from arc.config import all_configs
from arc.tokenizers import Formatter

from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

# from model_tools import (
#     save_model_and_tokenizer,
# )
# from model_tools import load_peft_state, merge_peft_into_base

# change this to take command line!
fine_tuning_config = all_configs[0]

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

model, tokenizer = fine_tuning_config.model_and_tokenizer_preprocessor(model, tokenizer)

# create lora model
model = FastLanguageModel.get_peft_model(
    model=model,
    target_modules=fine_tuning_config.lora_config.target_modules,
    r=fine_tuning_config.lora_config.lora_rank,
    lora_alpha=fine_tuning_config.lora_config.lora_alpha,
    lora_dropout=fine_tuning_config.lora_config.lora_dropout,
    bias=fine_tuning_config.lora_config.bias,
    use_gradient_checkpointing=True,
    random_state=fine_tuning_config.lora_config.random_state,
    use_rslora=fine_tuning_config.lora_config.use_rslora,
    loftq_config=fine_tuning_config.lora_config.loftq_config,
)


# run training
FastLanguageModel.for_training(model)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=fine_tuning_config.data_loader().map(
        Formatter(
            output_tail_token=tokenizer.eos_token
        ).transform_train_test_to_text_schema
    ),
    dataset_text_field=fine_tuning_config.sftt_config.dataset_text_field,
    max_seq_length=fine_tuning_config.sftt_config.max_seq_length,
    data_collator=None
    if fine_tuning_config.sftt_config.data_collator_constructor is None
    else fine_tuning_config.sftt_config.data_collator_constructor(tokenizer),
    args=TrainingArguments(
        run_name=fine_tuning_config.output_dir,
        per_device_train_batch_size=fine_tuning_config.sftt_config.per_device_train_batch_size,
        gradient_accumulation_steps=fine_tuning_config.sftt_config.gradient_accumulation_steps,
        warmup_ratio=fine_tuning_config.sftt_config.warmup_ratio,
        num_train_epochs=fine_tuning_config.sftt_config.num_train_epochs,
        learning_rate=fine_tuning_config.sftt_config.learning_rate,
        embedding_learning_rate=fine_tuning_config.sftt_config.embedding_learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=fine_tuning_config.sftt_config.logging_steps,
        optim=fine_tuning_config.sftt_config.optimizer,
        weight_decay=fine_tuning_config.sftt_config.weight_decay,
        lr_scheduler_type=fine_tuning_config.sftt_config.lr_scheduler_type,
        seed=fine_tuning_config.sftt_config.random_state,
        output_dir=fine_tuning_config.output_dir,
        save_strategy=fine_tuning_config.sftt_config.save_strategy,
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
