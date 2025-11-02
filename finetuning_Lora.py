from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_name = "microsoft/phi-2"

# 1️⃣ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2️⃣ Load model (float16, auto device)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="float16"
)

# 3️⃣ LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 4️⃣ Tiny toy dataset (adjust to your JSON file)
dataset = load_dataset("json", data_files="data/sample.jsonl")["train"]

def format_instruction(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n"
        f"### Input:\n{example['input']}\n"
        f"### Response:\n{example['output']}"
    )
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(format_instruction)

# 5️⃣ Training setup
training_args = TrainingArguments(
    output_dir="./phi2-lora-fp16",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=50,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=True,
    save_strategy="no",
    report_to="none"
)

# 6️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

# 7️⃣ Save LoRA adapter
model.save_pretrained("phi2-lora-fp16")
