import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    GenerationConfig,
)

model = None
tokenizer = None


def load_model(steps: int):
    global model
    global tokenizer
    if steps == -1:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2-1.5B-bnb-4bit")
        model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2-1.5B-bnb-4bit")
    else:
        PEFT_MODEL_PATH = f"./model-result/checkpoint-{steps}"
        config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                device_map="auto",
            ),
            PEFT_MODEL_PATH,
        )


def infer(inst: str, max: int = 128, would_print: bool = True):
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
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            streamer=streamer if would_print else None,
            pad_token_id=tokenizer.pad_token_id,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_p=0.75,
                top_k=40,
                repetition_penalty=5.0,
                max_new_tokens=max,
            ),
        )
        return tokenizer.decode(
            out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max", type=int, default=128)
    args = parser.parse_args()
    load_model(args.steps)
    while True:
        inp = input("User: ")
        if inp == "exit":
            break
        print("Model: ", end="")
        infer(inp, args.max)
