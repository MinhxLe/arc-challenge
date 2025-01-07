import json
import torch
from arc.tokenizers import Formatter
from tokenizers import Tokenizer
from trl import DataCollatorForCompletionOnlyLM
import peft

#### copied from architects, mutation everywhere


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


### Creating this function for reuse in model loading for eval
def preprocess_model_tokenizer_formatter(model, tokenizer):
    keep_tok = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-="
    ) + tokenizer.tokenize("\n")

    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)
    tokenizer.padding = "right"

    return model, tokenizer, Formatter(output_tail_token=tokenizer.eos_token)


def fix_dtypes(model, fix_weights=True, fix_quant_states=True):
    # fix some data types (workaround for unsloth)
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if weight is not None:
            if torch.is_floating_point(weight):
                if fix_weights and weight.dtype != model.dtype:
                    module.to(model.dtype)
            else:
                qs = getattr(weight, "quant_state", None)
                if qs is not None:
                    if fix_quant_states and qs.dtype != model.dtype:
                        qs.dtype = model.dtype
    return model


def get_and_fix_peft_weights(store):
    # change some keys (workaround for added 'modules_to_save')
    state_dict = peft.load_peft_weights(store)
    for k in list(state_dict.keys()):
        if "modules_to_save" in k:
            del state_dict[k]
            original_module_key = k.replace(".modules_to_save.", ".original_module.")
            if original_module_key in state_dict:
                del state_dict[original_module_key]
            assert k.replace(".modules_to_save.", ".") in state_dict
    return state_dict
