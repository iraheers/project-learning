import pandas as pd
from pathlib import Path
from PIL import Image
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
import ollama
import subprocess
import json
import base64
import torch
import csv
from groq import Groq
import json
import requests
from transformers import LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from PIL import Image
import gc
import ast
from qwen_vl_utils import process_vision_info
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
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

def convert_to_list(s):
    start = s.find('[')
    end = s.rfind(']')
    
    if start != -1 and end != -1 and end > start:
        cleaned = s[start:end+1].strip()
        lst = ast.literal_eval(cleaned)  # parse string into list
        lst = flatten_if_nested(lst)
        return lst
    else:
        print("No valid list found in the string.")
        return None
    
def flatten_if_nested(lst):
    if isinstance(lst, list) and len(lst) == 1 and isinstance(lst[0], list):
        return lst[0]
    return lst

def predict_nutrition(image_path,dish_id, raw_image_path):

    label_path = Path("outputs/" + dish_id + "/rgb/sam_mask_label/sam_mask_label.txt")
    label_df = pd.read_csv(label_path)  # replace with the actual path if needed
    unique_categories = label_df[label_df['category_name'] != 'background']['category_name'].unique()
    labels = ' '.join(unique_categories)

    prompt = f"""
        You are an expert in food recognition and nutritional analysis.
        You are provided with a panoramic segmentation image from FoodSAM.
        The identified food items in the image are: {labels}.
        dish_id: {dish_id}

        Your task is to estimate the total nutritional information of the dish in the image based on the identified food items.

        IMPORTANT: Return ONLY a python list in the following format:
        ["dish_id" as string,"calories" as float,"mass" as float,"fat" as float,"carb" as float,"protein" as float] 
        

        Do NOT include any extra comments or explanations.
        """

    while True:
        try:
            # Get response from Ollama
            
            #prediction = chat_with_ollama(prompt, image_path, raw_image_path)

            kill_ollama_process()
            prediction = chat_with_llava_onevision(prompt, image_path, raw_image_path)
            
            # Try extracting valid JSON
            print(f"Raw prediction: {prediction}")
            prediction = convert_to_list(prediction)

            
            # Success
            return prediction

        except Exception as e:
            print(f"Error extracting JSON, retrying... Error: {e}")
            # Retry immediately
            continue

def chat_with_ollama(prompt, image_path,raw_image_path):
    res = ollama.chat(
        #model="aimenmalik/diet_coach:latest",
        #model="SISTCA-Team4/SISTCA-Team4-Nutrition",
        model="llava:latest",
        messages=[
            {'role': 'user', 
            'content': prompt,
            'images': [image_path]},
        ]
    )

    return res['message']['content']

def chat_with_llava_onevision(prompt, image_path,raw_image_path):

    torch.cuda.empty_cache()
    gc.collect()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

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
                {"type": "text", "text": prompt},
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

    return result

def predict_nutrition_food_qwen(image_path,dish_id, raw_image_path):

    label_path = Path("outputs/" + dish_id + "/rgb/sam_mask_label/sam_mask_label.txt")
    label_df = pd.read_csv(label_path)  # replace with the actual path if needed
    unique_categories = label_df[label_df['category_name'] != 'background']['category_name'].unique()
    labels = ' '.join(unique_categories)

    prompt = f"""
    You are provided with a panoramic segmentation image from FoodSAM.
    Here are the labels in the image: {labels}.
    dish_id = {dish_id}

    Your task is to estimate the total nutritional information of the dish in the image based on the identified food items.
    
    Return  a python list in the following format: ["dish_id","calories" as float,"mass" as float,"fat" as float,"carb" as float,"protein" as float]
    
    Do NOT include any extra comments or explanations.
    """

    try:
        kill_ollama_process()
        prediction = chat_with_food_qwen(prompt, image_path, raw_image_path)
        
        # Try extracting valid JSON
        print(f"Raw prediction: {prediction}")
        prediction = convert_to_list(prediction)
        prediction[0] = dish_id  # Ensure dish_id is included
        
        # Success
        return prediction

    except Exception as e:
        print(f"Error extracting JSON, retrying... Error: {e}")
        # Retry immediately

def chat_with_ollama(prompt, image_path,raw_image_path):
    res = ollama.chat(
        #model="aimenmalik/diet_coach:latest",
        #model="SISTCA-Team4/SISTCA-Team4-Nutrition",
        model="llava:latest",
        messages=[
            {'role': 'user', 
            'content': prompt,
            'images': [image_path]},
        ]
    )

    return res['message']['content']

def chat_with_food_qwen(prompt, image_path,raw_image_path):

    torch.cuda.empty_cache()
    gc.collect()

    MODEL_PATH = "lordChipotle/nutrition-label-detector" # Or your Hugging Face Hub repo name
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True # Qwen models may require this
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # NOTE: For AdaMLLM, always place the image at the beginning of the input instruction in the messages.
    print(f"Using image path: {image_path}")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result


def predict_nutrition_groq(image_path,dish_id,raw_image_path):

    label_path = Path("outputs/" + dish_id + "/rgb/sam_mask_label/sam_mask_label.txt")
    label_df = pd.read_csv(label_path)  # replace with the actual path if needed
    unique_categories = label_df[label_df['category_name'] != 'background']['category_name'].unique()
    labels = ' '.join(unique_categories)

    prompt = f"""
    You are an expert in food recognition and nutritional analysis.
        You are provided with a panoramic segmentation image from FoodSAM.
        The identified food items in the image are: {labels}.
        dish_id: {dish_id}

        Your task is to estimate the total nutritional information of the dish in the image based on the identified food items.

        IMPORTANT: Return ONLY a python list in the following format:
        ["dish_id" as string,"calories" as float,"mass" as float,"fat" as float,"carb" as float,"protein" as float] 
        
        Do NOT include any extra comments or explanations.
    """

    # Getting the base64 string
    base64_image_og = encode_image(image_path)
    base64_image_raw = encode_image(raw_image_path)

    while True:
        try:
            # Get response from Ollama
            prediction = chat_with_groq(prompt, base64_image_og, base64_image_raw)
            
            # Try extracting valid JSON
            #prediction = extract_json(prediction)
            prediction = convert_to_list(prediction)
            
            # Success
            return prediction

        except Exception as e:
            print(f"Error extracting JSON, retrying... Error: {e}")
            # Retry immediately
            continue

def chat_with_groq(prompt, base64_image_og, base64_image_raw):
    client = Groq(api_key=API_KEY)

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
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    result = chat_completion.choices[0].message.content

    return result

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

def run_predictions():
    sample_ids = [] # Load dish IDs from file
    results = []   

    with open("nutrition5k_dataset/dish_ids/splits/sample_dish_ids.txt") as f:
        sample_ids = [line.strip() for line in f]


    for dish_id in sample_ids:
        image_path = Path(f"outputs/{dish_id}/rgb/panoramic_vis.png")
        raw_image_path = Path("outputs/" + dish_id + "/rgb/input.jpg")
        print(f"Processing dish ID: {dish_id}")

        if not Path(image_path).exists():
            print(f"Skipping {dish_id} — file not found.")
            continue

        row = predict_nutrition_food_qwen(image_path,dish_id,raw_image_path)

        print(f"Prediction for {dish_id}: {row}")
        results.append(row)

    return results

def save_predictions(results, run_name):
    # Save results to CSV
    with open(f"evaluations/predictions_{run_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        #writer.writerow(["dish_id", "calories", "mass", "fat", "carb", "protein"])
        writer.writerows(results)

def run_eval(name):
    groundtruth_file = "groundtruth_sample.csv"
    predictions_file = f"evaluations/predictions_{name}.csv"
    output_file = f"evaluations/output_{name}.json"

    # Define the command as a list
    command = [
        "python",
        "compute_eval_statistics.py",
        groundtruth_file,
        predictions_file,
        output_file
    ]

    # Run the command
    subprocess.run(command, check=True)

results = run_predictions()
#print(results)

run_name = "nutrition-label-detector"
save_predictions(results, run_name)
run_eval(run_name)
    



