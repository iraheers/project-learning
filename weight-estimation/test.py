import torch
import gc
from transformers import BitsAndBytesConfig, AutoProcessor
from PIL import Image
from transformers import LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import ast
from qwen_vl_utils import process_vision_info
import re

#no use for now
def calculate_volumes_with_individual_tracking(meshes_folder):
    """
    More detailed version that tracks individual files and totals
    """
    volume_dict = defaultdict(float)
    individual_volumes = {}
    
    obj_files = [f for f in os.listdir(meshes_folder) if f.endswith('.obj')]
    
    for obj_file in obj_files:
        try:
            # Extract food type
            food_type = re.sub(r'_\d+\.obj$', '', obj_file)
            
            obj_path = os.path.join(meshes_folder, obj_file)
            mesh = trimesh.load(obj_path)
            volume = mesh.volume
            
            # Store individual volume
            individual_volumes[obj_file] = {
                'food_type': food_type,
                'volume': volume
            }
            
            # Add to total
            volume_dict[food_type] += volume
            
        except Exception as e:
            print(f"Error processing {obj_file}: {e}")
            continue
    
    return dict(volume_dict), individual_volumes
#no use for now


def extract_numbers(text):
    """Extract the first number from text"""
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    if match:
        num_str = match.group()
        return float(num_str) if '.' in num_str else float(num_str)
    return None


def chat_with_llava_onevision(prompt, image_path):

    torch.cuda.empty_cache()
    gc.collect()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", 
                                                               device_map="auto",
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
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to("cuda:0", torch.float32)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=1000)

    input_token_len = inputs['input_ids'].shape[1]
    response_token_ids = output[0][input_token_len:]
    result = processor.decode(response_token_ids, skip_special_tokens=True)

    return result

def find_reference_object(image_path, reference_objects):
    """Step 1: Identify which reference object is in the image"""

    prompt = f"""Look at this image and identify the best item to use as a reference for volume estimation from the following list: 
    {', '.join(reference_objects)}. 
    Respond with ONLY the name of the object if found, or 'none' if none are present."""
    
    response = chat_with_llava_onevision(prompt, image_path)
    
    # Clean and normalize the response
    response_lower = response.lower().strip()
    
    # Check for each reference object
    for obj in reference_objects:
        if obj in response_lower:
            return obj
    
    return None


def estimate_reference_volume(image_path, reference_object):
    """Step 2: Estimate the volume of the reference object from the image"""
    prompt = f"""Look at the {reference_object} in this image. Estimate its approximate volume in cm³. 
    Consider its size relative to the image and provide a numerical volume estimate. 
    Respond with ONLY a number representing the estimated volume."""
    
    response = chat_with_llava_onevision(prompt, image_path)
    estimated_volume = extract_numbers(response)
    
    return estimated_volume

def estimate_density(image_path, reference_objects):
    """Estimate density for each item individually by looping through the dictionary"""
    
    density_results = {}
    
    for item in reference_objects:
        print(f"Estimating density for: {item}")
        
        prompt = f"""Look at this image and estimate the density of {item} in g/cm³.
        Return ONLY the number, no additional text."""
        
        response = chat_with_llava_onevision(prompt, image_path)
        density = extract_numbers(response)
        
        # If no density found, use reasonable default based on item typpw
        
        density_results[item] = density
    
    return density_results

if __name__ == "__main__":

    image_path = "images/_meal.png"

    volumes =  {
        "lettuce": 2.629155,
        "onion": 0.264777,
        "bread": 16.740599,
        "apple": 3.509062,
        "cheese": 0.419600,
        "butter": 0.419600,
        "cheese butter": 0.419600,
        "steak": 0.246867
    }

    reference_object = find_reference_object(image_path, list(volumes.keys()))
    print(f"Identified reference item: {reference_object}")

    ref_estimated_volume = estimate_reference_volume(image_path, reference_object)
    print(f"Estimated volume of {reference_object}: {ref_estimated_volume} cm³")

    actual_volume = volumes[reference_object]
    scale_factor =  ref_estimated_volume / actual_volume
    print(f"Scale factor (mesh units to cm³): {scale_factor}")

    scaled_volumes = {}
    for food_item, volume in volumes.items():
        scaled_volumes[food_item] = volume * scale_factor
    print(f"\nEstimated Real-World Volumes cm³: {scaled_volumes}")

    densities = estimate_density(image_path, list(volumes.keys()))
    print(f"Densities: {densities}")

    estimated_weights = {}
    for food_item, volume in scaled_volumes.items():
        density = densities.get(food_item, 1.0)  # Default density if not found
        estimated_weights[food_item] = volume * density

    print(f"Estimated weights (g): {estimated_weights}")

    final_prompt = f'''
    Here is a breakdown of estimated weights for each food items in the image in grams: {estimated_weights}.
    Estimate the calories, mass, fat, carb, protein for each item and total.
    '''

    nutritional_info = chat_with_llava_onevision(final_prompt, image_path)
    print(f"Nutritional Information:\n{nutritional_info}")