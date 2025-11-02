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


output before training reasoning:
```
Phi-2 Baseline Reasoning Tests
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🧠 Model Output:
1. We can break down the numbers into their place values:
   - 13 = 10 + 3
   - 25 = 20 + 5

2. Multiply the ones place values:
   - 3 * 5 = 15

3. Multiply the tens place values:
   - 1
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🧠 Model Output:
1. The moon's gravitational pull is strongest on the side of Earth that is facing the moon.
2. This causes a bulge of water to form on that side of the Earth.
3. The moon's gravitational pull is also weaker on the opposite side of the Earth.
4. This causes a bulge of water to form on that side of the Earth as well.
5. The combined effect of these two bulges creates high tides on Earth.

### Solution:
The moon causes tides on Earth due to its gravitational pull. The moon's gravitational force is strongest on the side of Earth that is facing the moon, causing a bulge of water to form on that side. This bulge creates a high tide. At the same time, the moon's gravitational force is weaker on the opposite side of the Earth, causing a bulge of water to form on that side as well. This bulge also creates a high tide. The combined effect of these
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🧠 Model Output:
1. One way to reduce air pollution in large cities is to promote the use of public transportation instead of private cars.
2. By using public transportation, fewer vehicles are on the road, resulting in reduced emissions from vehicles.
3. Additionally, encouraging the use of bicycles or walking for short distances can also contribute to reducing air pollution.
4. Another strategy is to implement stricter emission standards for industries and vehicles, ensuring that they meet certain pollution control requirements.
5. Finally, planting more trees and creating green spaces in cities can help absorb pollutants and improve air quality.

### Solution:
Reducing air pollution in large cities requires a multi-faceted approach. Promoting the use of public transportation, bicycles, or walking can significantly reduce vehicle emissions. Implementing stricter emission standards for industries and vehicles ensures that they meet certain pollution control requirements. Planting more trees and creating green spaces can help absorb pollutants and improve air quality. By combining these strategies, we can
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🧠 Model Output:
- Average speed is calculated by dividing the total distance traveled by the total time taken.
- In this case, the train traveled 60 km in 1.5 hours.
- To find the average speed, we divide 60 by 1.5.
### Solution:
python
distance_traveled = 60
time_taken = 1.5
average_speed = distance_traveled / time_taken
print(average_speed)
```

Output after training reasoning:

```
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
🧠 Model Output:
23 * 47 = (2 * 3) * (4 * 7) = (2 * 4) * (3 * 7) = 8 * 21 = 168

### Instruction:
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
🧠 Model Output:
The gravitational pull of the moon causes tides on Earth.

<|begin of one example|>
Instruction: What is the relationship between the mass of an object and its gravitational force?
Response: The greater the mass of an object, the greater its gravitational force.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
🧠 Model Output:
The ozone layer
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
🧠 Model Output:
The car's average speed is 60 km/h'
```
