from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float16", device_map="auto")

lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"],
                  lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)

# Tiny synthetic reasoning dataset
data = [
    {"instruction": "What is 12 * 14?", "input": "",
     "output": "Let's think step by step: 12*10=120, 12*4=48, sum=168. Final answer: 168."},
    {"instruction": "Why do leaves change color in autumn?", "input": "",
     "output": "Let's reason: As daylight shortens, chlorophyll breaks down, revealing yellow and red pigments. Therefore, leaves change color."}
]
dataset = Dataset.from_list(data)

def format(example):
    prompt = f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(format)

args = TrainingArguments(
    output_dir="./phi2-reasoning-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=5,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
model.save_pretrained("phi2-reasoning-lora")