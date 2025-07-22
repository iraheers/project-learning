import pandas as pd
from pathlib import Path
from PIL import Image
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
import ollama
import subprocess
import json

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

from groq import Groq
import base64
import os

image = "fruits"
image_name = image + ".jpg"
image_path = Path("test_images/" + image_name)
panoramic_image_path = Path("outputs/" + image + "/instance_vis.png")

# Resize to manageable size (e.g., 1024x1024 max)
MAX_SIZE = 512
img = Image.open(image_path)
img.thumbnail((MAX_SIZE, MAX_SIZE))
img.save(image_path)  # Overwrite or save separately

# Define the command as a list of strings
cmd = [
    "python",
    "FoodSAM/panoptic.py",
    "--img_path",
    str(image_path),
    "--output",
    "outputs"
]

# Run the command
# result = subprocess.run(cmd, capture_output=True, text=True)

# # Check return code
# if result.returncode == 0:
#     print("FoodSAM completed successfully!")
# else:
#     print("Error"+result.stderr)
#     exit(1)

prompt = f"""
You are an expert in food recognition and nutritional information.

The orginal image and panoramic images of food items are provided.


Analyze both the images, mask labels, and detected objects to provide a comprehensive analysis of the food items.
Your response should have 4 sections:
1. Detected Food Items: List all the food items detected.
2. Quantity Estimation: Estimate the quantity of each food item based on the masks and panoramic image.
3. Nutritional Information: Provide the nutritional information for each detected food item, including calories, protein, fat, and carbohydrates.
4. Total Nutritional Summary: Calculate the total calories, protein, fat, and carbohydrates for all detected food items.
"""

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Getting the base64 string
base64_image_og = encode_image(image_path)
base64_image_segmented = encode_image(panoramic_image_path)

client = Groq(api_key="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")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_og}",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image_segmented}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(chat_completion.choices[0].message.content)

result = chat_completion.choices[0].message.content

output_file = Path("llm_outputs/" + image + "_groq_llama4.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)

