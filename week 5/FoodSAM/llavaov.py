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

image = "fruits"
image_name = image + ".jpg"
image_path = "test_images/" + image_name
panoramic_image_path = "outputs/" + image + "/panoramic_vis.png"

# Step 3: Build the full prompt
template = f"""
You are an expert in food recognition and nutritional information.

The orginal image and panoramic images of food items are provided.

Analyze both the images, mask labels, and detected objects to provide a comprehensive analysis of the food items.
Your response should have 4 sections:
1. Detected Food Items: List all the food items detected.
2. Quantity Estimation: Estimate the quantity of each food item based on the masks and panoramic image.
3. Nutritional Information: Provide the nutritional information for each detected food item, including calories, protein, fat, and carbohydrates.
4. Total Nutritional Summary: Calculate the total calories, protein, fat, and carbohydrates for all detected food items.
"""

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

torch.cuda.empty_cache()

model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", 
                                                               device_map="auto",
                                                               torch_dtype=torch.float16,
                                                               )
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_path},
            {"type": "image", "url": panoramic_image_path},
            {"type": "text", "text": template},
        ],
    },
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
inputs = inputs.to("cuda:0", torch.float16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1000)
result = (processor.decode(output[0], skip_special_tokens=True))

print(result)

output_file = Path("llm_outputs/" + image + "_llavaov.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)