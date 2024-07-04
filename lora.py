from argparse import ArgumentParser
from datasets import load_dataset
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported


parser = ArgumentParser()
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

max_seq_length = 8192

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2-1.5B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=max_seq_length,
    use_rslora=False,
    loftq_config=None,
)

cfg = SFTConfig(
    num_train_epochs=3,
    learning_rate=5e-5,
    do_train=True,
    logging_steps=5,
    save_strategy="steps",
    save_steps=250,
    output_dir="model-result",
    save_total_limit=2,
    push_to_hub=False,
    auto_find_batch_size=True,
    max_seq_length=max_seq_length,
    optim="adamw_8bit",
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
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
            )
            + "<eos>"
        )
    return output_texts


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=cfg,
    data_collator=DataCollatorForCompletionOnlyLM(
        "<|im_start|>user\n", tokenizer=tokenizer
    ),
    formatting_func=formatting_prompts_func,
)

trainer.train(args.resume)
