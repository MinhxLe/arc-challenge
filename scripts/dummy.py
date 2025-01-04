from arc.config import all_configs
from peft import PeftModel
import torch
from unsloth import FastLanguageModel
from arc.external.architects import preprocess_model_tokenizer_formatter
from arc.datasets.seed import Datasets
from arc.core import Task


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
    # Load the adapter
    trained_model = PeftModel.from_pretrained(
        model=model,
        model_id="/shared/research/arc_challenge/runs/architects_copy_2024-12-26_keepers/checkpoint-30000/",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # merge in the LoRA weights
    trained_model.merge_and_unload()
    FastLanguageModel.for_inference(trained_model)
    post_train = generate_text(
        trained_model, tokenizer, train_example, max_new_tokens=10000
    )
    return post_train[:100]
    # '7!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
