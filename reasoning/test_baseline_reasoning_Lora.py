from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 1 Load base Phi-2 model
model_name = "microsoft/phi-2"
print(f"?~_~T? Loading {model_name} ...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2 Create generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 3 Define reasoning test helper
def ask(prompt: str, max_new_tokens: int = 200):
    formatted = f"### Question:\n{prompt}\n### Let's reason step-by-step:\n"
    output = pipe(
        formatted,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )[0]["generated_text"]

    print("\n?~_?|  Model Output:\n" + output.split("### Let's reason step-by-step:")[-1].strip())

# 4 Run baseline reasoning tests
print("\n?~_?? Phi-2 Baseline Reasoning Tests")