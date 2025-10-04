import pandas as pd
from pathlib import Path
from PIL import Image
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage

import subprocess

image = "plate3"
image_name = image + ".jpg"
saved_path = Path("test_images/" + image_name)

# Define the command as a list of strings
cmd = [
    "python",
    "FoodSAM/semantic.py",
    "--img_path",
    "test_images/" + image_name,
    "--output",
    "outputs"
]

# Run the command
result = subprocess.run(cmd, capture_output=True, text=True)

# Check return code
if result.returncode == 0:
    print("FoodSAM completed successfully!")
else:
    print("FoodSAM encountered an error.")
    exit(1)

# Load FoodSAM CSV
file_path = Path("outputs/" + image + "/sam_mask_label/semantic_masks_category.txt")
df = pd.read_csv(file_path)

# Ignore background
df_objects = df[df["category_name"] != "background"].copy()

# Get total image size
sample_mask_path = Path("outputs/" + image + "/sam_mask/0.png")
img = Image.open(sample_mask_path)
total_pixels = img.width * img.height

# Aggregate mask ratios
agg_df = df_objects.groupby("category_name").agg({
    "mask_count_ratio": "sum"
}).reset_index()

# Compute pixel areas
agg_df["mask_area_pixels"] = (agg_df["mask_count_ratio"] * total_pixels).astype(int)

# Build detection list
detected_items = []
for _, row in agg_df.iterrows():
    item = {
        "class_name": row["category_name"],
        "confidence": 1.0,
        "box": None,
        "mask_area_pixels": int(row["mask_area_pixels"]),
        "mask_area_ratio": row["mask_count_ratio"]
    }
    detected_items.append(item)

print("Detected items from FoodSAM:")
print(detected_items)

# Build prompt
detected_text = f"Detected items from FoodSAM: {detected_items}"

prompt = f"""
You are an expert in food recognition and nutritional information.

An image of food items with segmentation masks is provided.
{detected_text}

Your response should have 4 sections:
1. Detected Food Items: List all the food items detected in the image.
2. Quantity Estimation: Estimate the quantity of each food item based on the masks.
3. Nutritional Information: Provide the nutritional information for each detected food item, including calories, protein, fat, and carbohydrates.
4. Total Nutritional Summary: Calculate the total calories, protein, fat, and carbohydrates for all detected food items.
"""

# from openai import OpenAI

# client = OpenAI(api_key="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")

# completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
#     messages=[
#         {
#             "role": "system",
#             "content": "You are an expert in food recognition and nutritional information."
#         },
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
# )

# print(completion.choices[0].message.content)

# # Save output
# output_file = Path("llm_outputs/" + image + "/llava_output.txt")
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write(completion.choices[0].message.content)

messages = [
    HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"file://{saved_path}"
                }
            }
        ]
    )
]

model = OllamaLLM(model="llava")
result = model.invoke(messages)
print(result)

# Save output
output_file = Path("llm_outputs/" + image + ".txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)


