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

import json

def extract_json(raw_string):
    # Find first { and last }
    start = raw_string.find('{')
    end = raw_string.rfind('}') + 1  # +1 to include the }
    
    if start == -1 or end == 0:
        raise ValueError("No valid JSON object found")
    
    # Extract the JSON portion
    json_str = raw_string[start:end]
    
    # Parse and return
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If still invalid, try cleaning more aggressively
        json_str = json_str.split('{', 1)[1].rsplit('}', 1)[0]
        return json.loads('{' + json_str + '}')

image = "test1"
image_name = image + ".jpeg"
image_path = Path("test_images/" + image_name)
panoramic_image_path = Path("outputs/" + image + "/panoramic_vis.png")

# Resize to manageable size (e.g., 1024x1024 max)
MAX_SIZE = 512
img = Image.open(image_path)
img.thumbnail((MAX_SIZE, MAX_SIZE))
img.save(image_path)  # Overwrite or save separately

# Define the command as a list of strings
# cmd = [
#     "python",
#     "FoodSAM/panoptic.py",
#     "--img_path",
#     str(image_path),
#     "--output",
#     "outputs"
# ]

# # Run the command
# result = subprocess.run(cmd, capture_output=True, text=True)

# # Check return code
# if result.returncode == 0:
#     print("FoodSAM completed successfully!")
# else:
#     print("Error"+result.stderr)
#     exit(1)

# # Load FoodSAM CSV
# file_path = Path("outputs/" + image + "/sam_mask_label/sam_mask_label.txt")
# df = pd.read_csv(file_path)
# df_objects = df[df["category_name"] != "background"].copy()
# detected_items = df_objects[["category_name", "category_count_ratio", "mask_count_ratio"]].to_dict(orient="records")
# detected_text = f"Detected items from FoodSAM: {detected_items}"

# json_file_path = Path("outputs/" + image + "/object_detection/od_UniDet.json")
# with open(json_file_path, "r") as f:
#     json_data = json.load(f)
# json_string = json.dumps(json_data, indent=2) 

# print("Detected items from FoodSAM:")
# print(detected_items)

# # Build prompt
dish_id = "test_dish"

prompt = f"""
    You are an expert in food recognition and nutritional analysis.

    You will be provided with:
    - An original image and a panoramic image of a dish.
    - Mask labels and detected objects have already been applied to these images.

    Your task is to analyze the images and provide nutritional information for the dish.

    IMPORTANT: Return ONLY a JSON object in the following format:

    {{
        "dish_id": "{dish_id}",
        "calories": float,
        "mass": float,
        "fat": float,
        "carb": float,
        "protein": float
    }}

    Do NOT include any extra comments or explanations.
    """

res = ollama.chat(
    model="llava:v1.6",
    #model="llava",
    messages=[
        {'role': 'user', 
         'content': prompt,
         'images': [image_path, panoramic_image_path]}
    ]
)

result = res['message']['content']

prediction = extract_json(result)



row = [
        dish_id,
        prediction["calories"],
        prediction["mass"],
        prediction["fat"],
        prediction["carb"],
        prediction["protein"]
    ]

print(row)


# Save output
# output_file = Path("llm_outputs/" + image + "_llava.txt")
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write(result)


