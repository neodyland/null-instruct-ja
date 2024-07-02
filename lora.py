from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from datasets import load_dataset
import torch
from argparse import ArgumentParser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

parser = ArgumentParser()
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "gate_proj",
        "k_proj",
        "q_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

cfg = SFTConfig(
    num_train_epochs=2,
    learning_rate=5e-5,
    do_train=True,
    logging_steps=5,
    save_strategy="steps",
    save_steps=250,
    output_dir="model-result",
    save_total_limit=2,
    push_to_hub=False,
    auto_find_batch_size=True,
    max_seq_length=8192,
)

dataset = load_dataset("neody/null-instruct-ja", split="train")


def formatting_prompts_func(example):
    output_texts = []
    for i, user in enumerate(example["user"]):
        output_texts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": example["model"][i]},
                ],
                tokenize=False,
            )[: -len("<end_of_turn>\n")]
            + "<eos>"
        )
    return output_texts


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=cfg,
    data_collator=DataCollatorForCompletionOnlyLM(
        "<start_of_turn>model\n", tokenizer=tokenizer
    ),
    formatting_func=formatting_prompts_func,
    peft_config=lora_config,
)

trainer.train(args.resume)
