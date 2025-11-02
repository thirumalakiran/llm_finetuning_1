rom transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# 1 Load base model + LoRA adapter
base_model = "microsoft/phi-2"
adapter_path = "phi2-reasoning-lora"  # directory from your fine-tuning step

print("?~_~T? Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="float16", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 2 Define a helper for interactive testing
def ask(prompt: str, max_new_tokens: int = 200):
    formatted = f"### Instruction:\n{prompt}\n### Response:\n"
    output = pipe(formatted, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    print("?~_?|  Model Output:\n" + output[0]["generated_text"].split("### Response:\n")[-1].strip())

# 3 Run test examples
print("\n?~_?? Reasoning Test Examples")

ask("What is 23 * 47? Explain your reasoning.")
ask("Why does the moon cause tides on Earth?")
ask("How can we reduce air pollution in large cities?")
ask("A train travels 60 km in 1.5 hours. What is its average speed?")