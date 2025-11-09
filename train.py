from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

dataset = load_dataset("camel-ai/physics", split="train")

def format_dataset(example):
    conversation = [
        {"role": "system", "content": "You are a physics expert. Answer clearly and accurately."},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    return {"text": text}


dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

FastLanguageModel.for_training(model)

trainer = FastLanguageModel.get_trainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    max_seq_length = max_seq_length,
    output_dir = "outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    dataset_text_field = "text",
    learning_rate = 2e-4,
    warmup_steps = 5,
    max_steps = 400,
    weight_decay = 0.001,
    logging_steps = 1,
    seed = 3407,
    optim = "adamw_8bit",
    lr_scheduler_type = "linear",
    report_to = "none",
)

trainer.train()
model.save_pretrained_merged("final_model", tokenizer, save_method="merged_16bit")
tokenizer.save_pretrained("final_model")  
