import os

from pathlib import Path
import subprocess

import torch
from transformers import LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
from PIL import Image
import ast



image = "apple_and_burger"
image_name = image + ".png"
image_path = "images/" + image_name

# Step 3: Build the full prompt
template = f"""
        You are an expert in food recognition and nutritional analysis.
        You are provided with an image of a dish.
        
        Your task is to estimate the total nutritional information for the dish based on the identified food items.

        IMPORTANT: Return ONLY a string in the following format:
        ["dish_id","calories","mass","fat","carb","protein"]
        where calories, mass, fat, carb, and protein are float numbers

        Do NOT include any extra comments or explanations.
        """

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

torch.cuda.empty_cache()

model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", 
                                                               device_map="cpu",
                                                               torch_dtype=torch.float32,
                                                                quantization_config=quantization_config,
                                                               )
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

original_image = Image.open(image_path).convert("RGB")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": original_image},
            {"type": "text", "text": template},
        ],
    },
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
inputs = inputs.to("cpu", torch.float32)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1000)

input_token_len = inputs['input_ids'].shape[1]
response_token_ids = output[0][input_token_len:]
result = processor.decode(response_token_ids, skip_special_tokens=True)

print(result)

start = result.find('[')
end = result.rfind(']')

# Extract only the list portion
if start != -1 and end != -1 and end > start:
    cleaned = result[start:end+1].strip()
    print(type(cleaned))  # <class 'str'>
else:
    print("No valid list found.")
    cleaned = None

if cleaned:
    lst = ast.literal_eval(cleaned)  # Use ast.literal_eval to parse string into list
    print(type(lst))

# output_file = Path("llm_outputs/" + image + "_llavaov.txt")
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write(result)