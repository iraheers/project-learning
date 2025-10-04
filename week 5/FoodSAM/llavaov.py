from ultralytics import YOLO

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import ollama
import os

from pathlib import Path
import subprocess

import torch
from transformers import LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from PIL import Image
import ast



image = "apple"
image_name = image + ".jpg"
image_path = "test_images/" + image_name
panoramic_image_path = "outputs/" + image + "/panoramic_vis.png"
panoramic_image_path = "outputs/dish_1558461792/rgb/panoramic_vis.png"

# Step 3: Build the full prompt
template = f"""
        You are an expert in food recognition and nutritional analysis.
        You are provided with a panoramic segmentation image from FoodSAM.
        

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
                                                               device_map="auto",
                                                               torch_dtype=torch.float16,
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
inputs = inputs.to("cuda:0", torch.float16)

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