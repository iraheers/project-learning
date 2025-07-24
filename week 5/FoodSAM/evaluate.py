import pandas as pd
from pathlib import Path
from PIL import Image
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
import ollama
import subprocess
import json

import torch

import os
import csv
import json
import gc

import subprocess
import time

from groq import Groq
import base64

def chat_with_ollama(prompt, image_path, panoramic_image_path):
    res = ollama.chat(
        model="llava:v1.6",
        #model="llava",
        messages=[
            {'role': 'user', 
            'content': prompt,
            'images': [image_path, panoramic_image_path]}
        ]
    )

    return res['message']['content']

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to predict nutrition information for a dish
def predict_nutrition(image_path,dish_id):

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.ipc_collect()

    image = os.path.splitext(os.path.basename(image_path))[0] 
    panoramic_image_path = Path("outputs/" + image + "/panoramic_vis.png")

    # Resize to manageable size (e.g., 1024x1024 max)
    MAX_SIZE = 256
    img = Image.open(image_path)
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    #img.save(image_path)  # Overwrite or save separately

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
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check return code
    if result.returncode == 0:
        print("FoodSAM completed successfully!")
    else:
        print("Error"+result.stderr)
        exit(1)

    # Load FoodSAM CSV
    file_path = Path("outputs/" + image + "/sam_mask_label/sam_mask_label.txt")
    df = pd.read_csv(file_path)
    df_objects = df[df["category_name"] != "background"].copy()
    detected_items = df_objects[["category_name", "category_count_ratio", "mask_count_ratio"]].to_dict(orient="records")
    detected_text = f"Detected items from FoodSAM: {detected_items}"

    prompt = f"""
    You are an expert in food recognition and nutritional analysis.

    You will be provided with:
    - An original image and a panoramic image of a dish.
    - Mask labels and detected objects have already been applied to these images.

    From the provided images, extract the nutritional information of the entire dish.

    IMPORTANT: Return ONLY a JSON object in the following format:

    {{
        "dish_id": "{dish_id}",
        "calories": float,
        "mass": float,
        "fat": float,
        "carb": float,
        "protein": float
    }}

    Do not include any extra commentary or explanation.
    """

    while True:
        try:
            # Get response from Ollama
            prediction = chat_with_ollama(prompt, image_path, panoramic_image_path)
            
            # Try extracting valid JSON
            prediction = extract_json(prediction)
            
            # Success
            return prediction

        except Exception as e:
            print(f"Error extracting JSON, retrying... Error: {e}")
            # Retry immediately
            continue

def predict_nutrition_groq(image_path,dish_id):

    image = os.path.splitext(os.path.basename(image_path))[0] 
    panoramic_image_path = Path("outputs/" + image + "/panoramic_vis.png")

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

    prompt = f"""
    You are an expert in food recognition and nutritional analysis.

    You will be provided with:
    - An original image and a panoramic image of a dish.
    - Mask labels and detected objects have already been applied to these images.

    From the provided images, extract the nutritional information of the entire dish.

    IMPORTANT: Return ONLY a JSON object in the following format:

    {{
        "dish_id": "{dish_id}",
        "calories": float,
        "mass": float,
        "fat": float,
        "carb": float,
        "protein": float
    }}

    Do not include any extra commentary or explanation.
    """

    # Getting the base64 string
    base64_image_og = encode_image(image_path)
    base64_image_segmented = encode_image(panoramic_image_path)

    while True:
        try:
            # Get response from Ollama
            prediction = chat_with_groq(prompt, base64_image_og, base64_image_segmented)
            
            # Try extracting valid JSON
            prediction = extract_json(prediction)
            
            # Success
            return prediction

        except Exception as e:
            print(f"Error extracting JSON, retrying... Error: {e}")
            # Retry immediately
            continue

def chat_with_groq(prompt, base64_image_og, base64_image_segmented):
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

    result = chat_completion.choices[0].message.content

    return result
    


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

def kill_ollama_process():
    try:
        # Find all PIDs of ollama processes using the full path
        result = subprocess.run(
            ["pgrep", "-f", "/usr/local/bin/ollama"],
            capture_output=True, text=True, check=True
        )

        pids = result.stdout.strip().split("\n")

        for pid in pids:
            if pid:
                print(f"Killing Ollama process with PID {pid}")
                subprocess.run(["sudo", "kill", "-9", pid], check=True)

    except subprocess.CalledProcessError:
        print("No Ollama process found.")


sample_ids = []
with open("nutrition5k_dataset/dish_ids/splits/sample_dish_ids.txt") as f:
    sample_ids = [line.strip() for line in f]

results = []  # list of (dish_id, calories, mass, fat, carb, protein)

for dish_id in sample_ids:
    image_path = f"nutrition5k_dataset/imagery/realsense_overhead/{dish_id}/rgb.png" 
    print(f"Processing dish ID: {dish_id}")

    if not Path(image_path).exists():
        print(f"Skipping {dish_id} — file not found.")
        continue

    time.sleep(10) 
    kill_ollama_process()
    
    prediction = predict_nutrition_groq(image_path,dish_id)
    
    row = [
        dish_id,
        prediction["calories"],
        prediction["mass"],
        prediction["fat"],
        prediction["carb"],
        prediction["protein"]
    ]
    print(f"Prediction for {dish_id}: {row}")
    results.append(row)

# Save results to CSV
with open("predictions_sample_prompt_groq.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dish_id", "calories", "mass", "fat", "carb", "protein"])
    writer.writerows(results)


