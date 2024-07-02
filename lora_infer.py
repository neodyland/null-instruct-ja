import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GemmaForCausalLM,
    TextStreamer,
)
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--steps", type=str, required=True)
args = parser.parse_args()

PEFT_MODEL_PATH = f"./model-result/checkpoint-{args.steps}"

config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model: GemmaForCausalLM = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)


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
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    with torch.no_grad():
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.75,
            top_k=40,
            repetition_penalty=5.0,
            pad_token_id=tokenizer.pad_token_id,
        )
