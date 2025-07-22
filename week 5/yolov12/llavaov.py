from ultralytics import YOLO

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import ollama
import os

from pathlib import Path

import torch
from transformers import LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoProcessor

file_name = "fruits.jpg"
basename = os.path.splitext(file_name)[0]

model_yolo = YOLO("yolov12s-seg.pt")
results = model_yolo.predict("test_images/"+file_name, save=True, task="segment")

#results[0].show()
saved_path = results[0].save_dir + "/"+basename + ".jpg"
print(f"Image saved at: {saved_path}")

# Step 3: Build the full prompt
template = f"""
You are an expert in food recognition and nutritional information.

An image of food items with bounding boxes and segmentation masks is provided.

Your response should have 4 sections:
1. Detected Food Items: List all the food items detected in the image.
2. Quantity Estimation: Estimate the quantity of each food item based on the bounding boxes and masks.
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
            {"type": "image", "url": saved_path},
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

save_dir = Path(results[0].save_dir)
output_file = save_dir / "llava_output.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)