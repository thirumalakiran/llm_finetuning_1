from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

base_model = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="float16", device_map="auto")
model = PeftModel.from_pretrained(model, "phi2-lora-fp16")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("### Instruction:\nTranslate to French\n### Input:\nHello world\n### Response:\n", max_new_tokens=40)[0]["generated_text"])
