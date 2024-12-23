from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

from arc.core import Task
from arc.datasets.seed import Datasets
from arc.tokenizers import Formatter
from arc.datasets import transform as dst
from arc import transform as t
from datasets import load_from_disk

import json
import torch
from tokenizers import Tokenizer
from trl import DataCollatorForCompletionOnlyLM


class InputMaskingDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, mask_first_n_examples=0, **kwargs):
        super().__init__(**kwargs)
        self.mask_first_n_examples = mask_first_n_examples

    def torch_call(self, examples):
        # [TODO] confirm understanding that this is working.
        batch = super().torch_call(examples)  # call super, masking all inputs
        for i in range(len(batch["labels"])):
            for _ in range(self.mask_first_n_examples):
                # mask first still unmasked output block
                beg_pos = ((batch["labels"][i] != -100).nonzero().min()).item()
                mid_pos = (
                    (batch["labels"][i][beg_pos:] == -100).nonzero().min()
                ).item() + beg_pos
                end_pos = ((batch["labels"][i] != -100).nonzero().max()).item() + 1
                if mid_pos < end_pos:
                    batch["labels"][i][beg_pos:mid_pos] = -100
        return batch


def load_unsloth_4bit(model_path):
    from unsloth import FastLanguageModel

    return FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
    )


def get_or_map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get("special_tokens")
        if special is not None:  # find and/or update special token mappings
            for v in special.values():
                tokens.update(v["ids"])
                if mapping is not None:
                    v["ids"] = [mapping.get(i) for i in v["ids"] if i in mapping]
        for v in data.values():  # recursively process dict values
            tokens.update(get_or_map_special_tokens(v, mapping))
    if isinstance(data, list):
        for v in data:  # recursively process lists
            tokens.update(get_or_map_special_tokens(v, mapping))
    return tokens


def remove_tokenizer_normalizer(tokenizer):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get("normalizer") is not None:
        tokenizer_json["normalizer"] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def shrink_tokenizer_vocab(
    tokenizer, keep_indices, keep_special=True, remove_unk=False
):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json["model"]["type"] == "BPE"

    if keep_special:  # get special tokens to keep
        keep_indices.update(tokenizer.all_special_ids)
        keep_indices.update(get_or_map_special_tokens(tok_json.get("post_processor")))

    if remove_unk:  # remove unknown token
        keep_indices -= {tokenizer.unk_token_id}

    # build mapping from old to new id
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

    # update tokenizer info
    tok_json["model"]["vocab"] = {
        k: mapping[v] for k, v in tok_json["model"]["vocab"].items() if v in mapping
    }
    tok_json["model"]["merges"] = []
    tok_json["added_tokens"] = [
        {**t, "id": mapping[t["id"]]}
        for t in tok_json["added_tokens"]
        if t["id"] in mapping
    ]
    tok_json["added_tokens"] = sorted(tok_json["added_tokens"], key=lambda t: t["id"])
    get_or_map_special_tokens(tok_json.get("post_processor"), mapping)

    tokenizer._tokenizer = Tokenizer.from_str(
        json.dumps(tok_json)
    )  # reload json, modifying tokenizer in-place

    if remove_unk:
        tokenizer.unk_token = None

    return mapping  # token mapping to be used later


def shrink_model_embeddings(model, mapping):
    with torch.no_grad():
        # copy embeddings to keep
        row_select = torch.tensor(
            [x[0] for x in sorted(mapping.items(), key=lambda x: x[1])]
        )
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed_t = torch.index_select(
            model.get_input_embeddings().weight.data, 0, row_select
        )
        row_select = row_select.to(model.get_output_embeddings().weight.data.device)
        new_lm_head = torch.index_select(
            model.get_output_embeddings().weight.data, 0, row_select
        )

        # resize model embeddings
        model.resize_token_embeddings(len(row_select))

        # set to copied values
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head

        # map model tokens to new id
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith("token_id"):
                    setattr(
                        config,
                        k,
                        [mapping.get(t) for t in v]
                        if isinstance(v, list)
                        else mapping.get(v),
                    )


def keep_single_char_tokens(
    model, tokenizer, keep=None, keep_norm=False, keep_model_tok=True, **kwargs
):
    if not keep_norm:
        remove_tokenizer_normalizer(tokenizer)  # required for some models
    if keep is None:  # keep all single_length tokens
        keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    else:  # keep tokens that were passed
        keep_indices = set(tokenizer.vocab[t] for t in keep)
    if keep_model_tok:  # keep tokens used by model
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith("token_id"):
                    keep_indices.update(v if isinstance(v, list) else [v])
    keep_indices -= {None}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
    shrink_model_embeddings(model, mapping)
    return mapping


base_model = (
    "nvidia/Mistral-NeMo-Minitron-8B-Base"  # auto-downloaded from huggingface.co
)


model = tokenizer = None  # free memory
model, tokenizer = load_unsloth_4bit(base_model)
keep_tok = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-="
) + tokenizer.tokenize("\n")
keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)
# [TODO] explicitly create a pad token

# create lora model
lora_layers = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
]
model = FastLanguageModel.get_peft_model(
    model=model,
    target_modules=lora_layers,
    r=256,
    lora_alpha=24,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=True,
    loftq_config=None,
)

formatter = Formatter(output_tail_token=tokenizer.eos_token)


def format_row(row):
    task = Task.from_dict(row)
    row.pop("train")
    row.pop("test")
    return formatter.format_task_for_sft(task)


base_train_dataset = dst.concat(
    # [TODO] repeat 128
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

train_dataset = dst.concat(
    transformed_train_dataset,
    dst.shuffle_train_order(transformed_train_dataset, seed=42),
)

train_dataset = train_dataset.map(format_row, num_proc=24)


data_collator = InputMaskingDataCollator(
    instruction_template=formatter.input_head_token,
    response_template=formatter.output_head_token,
    mlm=False,
    tokenizer=tokenizer,
    mask_first_n_examples=1,
)

model = FastLanguageModel.for_training(model)
tokenizer.padding_side = "right"


# [TODO] it will be faster for us to build the dataset formatting ourselves
# this is saved from cache
train_dataset = load_from_disk(
    "/shared/research/arc_challenge/data/train/2024_12_22_train/"
)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=8192,
    packing=False,
    data_collator=data_collator,
    args=TrainingArguments(
        # [TODO] This might affect normalization
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.25,
        num_train_epochs=1,
        learning_rate=1e-4,
        embedding_learning_rate=1e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="tmp_output",
        save_strategy="no",
        report_to="wandb",
    ),
)
trainer_stats = unsloth_train(trainer)
