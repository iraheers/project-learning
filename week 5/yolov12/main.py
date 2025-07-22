from ultralytics import YOLO

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import ollama
import os

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


res = ollama.chat(
    model="llava:v1.6",
    messages=[
        {'role': 'user', 
         'content': template,
         'images': [saved_path]}
    ]
)

result = res['message']['content']

print(result)

from pathlib import Path

save_dir = Path(results[0].save_dir)
output_file = save_dir / "llava_output.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)