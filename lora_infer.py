import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PEFT_MODEL_PATH = "./model-result/checkpoint-286"

config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)


def infer(inst: str):
    inputs = tokenizer(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": inst},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ),
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.75,
            top_k=40,
            repetition_penalty=5.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)
