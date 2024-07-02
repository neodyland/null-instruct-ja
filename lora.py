from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
import torch
from argparse import ArgumentParser

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

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

args = TrainingArguments(
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
)

dataset = load_dataset("neody/null-instruct-ja", split="train")


def convert(item):
    return tokenizer(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": item["user"]},
                {"role": "assistant", "content": item["model"]},
            ],
            tokenize=False,
        )
    )


dataset = dataset.map(convert, remove_columns=["user", "model"]).filter(
    lambda item: len(item["input_ids"]) <= 8192
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train(args.resume)
