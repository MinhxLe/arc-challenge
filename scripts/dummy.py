from arc.config import all_configs
from peft import PeftModel
import peft
import torch
from unsloth import FastLanguageModel
from arc.external.architects import preprocess_model_tokenizer_formatter
from arc.datasets.seed import Datasets
from arc.core import Task


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


def generate_text(model, tokenizer, message, max_new_tokens=100):
    # Get the device that the model is on
    device = next(model.parameters()).device

    # Tokenize the input text
    inputs = tokenizer(message, return_tensors="pt")

    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,  # Instead of max_length, to control new tokens only
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=inputs["attention_mask"],
        return_dict_in_generate=True,
        do_sample=False,
        temperature=1.0,
        num_beams=1,
    )

    # Get only the generated text (excluding the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs.sequences[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


fine_tuning_config = next(
    config for config in all_configs if config.name == "architects"
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=fine_tuning_config.model_config.model,
    dtype=fine_tuning_config.model_config.model_dtype,
    load_in_4bit=fine_tuning_config.model_config.load_in_4bit,
)

model, tokenizer, formatter = preprocess_model_tokenizer_formatter(model, tokenizer)

train_example = formatter.format_task(
    Task.from_dict(Datasets.arc_public_train.get_dataset()[0]),
    include_test_output=False,
)


def test_untuned_model():
    FastLanguageModel.for_inference(model)
    pre_train = generate_text(model, tokenizer, train_example, max_new_tokens=10000)
    return pre_train[:100]
    # '7\n7\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n'


def test_tuned_model():
    checkpoint = "/shared/research/arc_challenge/runs/architects_copy_2024-12-26_keepers/checkpoint-30000/"

    # set up the model, it will get correct LoRA config from checkpoint
    model = PeftModel.from_pretrained(
        model=model,
        model_id=checkpoint,
        device_map="cuda",
    )

    weight_set_result = peft.set_peft_model_state_dict(
        model,
        get_and_fix_peft_weights(checkpoint),
    )
    assert (
        not weight_set_result.unexpected_keys
    ), "error loading weights - some keys not available in model"

    model = fix_dtypes(model.merge_and_unload())

    FastLanguageModel.for_inference(model)
    post_train = generate_text(model, tokenizer, train_example, max_new_tokens=10000)
    return post_train[:100]
    # architects copy:  '707000707\n707000707\n770000770\n707000707\n707000707\n770000770\n707707000\n707707000\n770770000\n'
    # my rework:        '707000707\n707000707\n770000770\n707000707\n707000707\n770000770\n707707000\n707707000\n770770000\n'
