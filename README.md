# llm_finetuning_1
Sample finetuning using runpod

pip install transformers==4.44.0 peft datasets accelerate bitsandbytes sentencepiece

pip install huggingface_hub

huggingface-cli login


mkdir data

vim sample.jsonl

```{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}

{"instruction": "What is 2+2?", "input": "", "output": "4"}

{"instruction": "Write a haiku about AI", "input": "", "output": "Machines softly dream\nNumbers hum beneath the code\nWisdom wakes in light"}

{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}

{"instruction": "Summarize", "input": "The sky is blue and full of clouds.", "output": "The sky is cloudy and blue."}
```

