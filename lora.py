from argparse import ArgumentParser
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer
from torch import nn
from typing import Dict, Union, Any
import gc

PatchDPOTrainer()
from datasets import load_dataset
import torch
from trl import DPOTrainer, DPOConfig


parser = ArgumentParser()
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

max_seq_length = 8192

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-1.1-2b-it-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
tokenizer.padding_side = "right"
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

cfg = DPOConfig(
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
    optim="adamw_8bit",
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    max_prompt_length=max_seq_length,
    max_length=max_seq_length,
    loss_type="ipo",
    remove_unused_columns=False,
)

dataset = load_dataset("neody/null-instruct-ja", split="train")


def func(example):
    prompts = []
    chosens = []
    rejecteds = []
    for i, user in enumerate(example["user"]):
        prompts.append(user)
        chosens.append(example["model"][i]),
        rejecteds.append(example["reject"][i]),
    return {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    }


dataset = dataset.map(func, batched=True, remove_columns=list(dataset.features))


class Trainer(DPOTrainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        loss_step = super().training_step(model, inputs)
        torch.cuda.empty_cache()
        gc.collect()
        return loss_step


trainer = Trainer(model=model, train_dataset=dataset, args=cfg, tokenizer=tokenizer)

trainer.train(args.resume)
