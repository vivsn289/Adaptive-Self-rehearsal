"""
Full SFT (not LoRA) of Mistral-7B on self-generated instruction data.
Optimized for RTX 3090 24GB.

RESUMABLE: Auto-resumes from the latest checkpoint in the output dir if present.
"""
import os
import json
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)

import os
SEED = int(os.environ.get('SEED', 42))
EPOCHS = int(os.environ.get('EPOCHS', 4))
OUTPUT_DIR_OVERRIDE = os.environ.get('OUTPUT_DIR_OVERRIDE')
BASE_MODEL = "Qwen/Qwen2.5-3B"
DATA_PATH = os.environ.get("INSTRUCTIONS_PATH", "generated_data/self_instruct_generated.json")
OUTPUT_DIR = OUTPUT_DIR_OVERRIDE or "./qwen-self-instruct-sft"

# Load data
with open(DATA_PATH) as f:
    data = json.load(f)
print(f"Training on {len(data)} self-generated samples", flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

def format_and_tokenize(sample):
    if sample.get("input", "").strip():
        text = (f"Below is an instruction that describes a task, paired with an input. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Input:\n{sample['input']}\n\n"
                f"### Response:\n{sample['output']}{tokenizer.eos_token}")
    else:
        text = (f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Response:\n{sample['output']}{tokenizer.eos_token}")
    tokens = tokenizer(text, truncation=True, max_length=512, padding=False)
    return tokens

dataset = Dataset.from_list(data)
tokenized = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

# Load model with gradient checkpointing
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, seed=SEED,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=25,
    save_steps=10000,            # Frequent checkpoints for resumability
    save_total_limit=1,
    report_to="none",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized,
    data_collator=data_collator, processing_class=tokenizer,
)

# Auto-resume from checkpoint if one exists
resume = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        resume = True
        print(f"RESUMING from existing checkpoint in {OUTPUT_DIR}", flush=True)

print("Starting full SFT (gradient checkpointing ON, effective batch 32, 2 epochs)...", flush=True)
trainer.train(resume_from_checkpoint=resume)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Model saved to {OUTPUT_DIR}", flush=True)
