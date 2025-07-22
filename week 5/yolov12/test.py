import torch

from transformers import LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoProcessor

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", 
                                                               quantization_config=quantization_config, 
                                                               device_map="auto",
                                                               torch_dtype=torch.float16,
                                                               )
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

# prepare image and text prompt, using the appropriate prompt template
url = "test_images/fruits.jpg"
template = f"""
You are an expert in food recognition and nutritional information.

An image of food items with bounding boxes and segmentation masks is provided.

Your response should have 4 sections:
1. Detected Food Items: List all the food items detected in the image.
2. Quantity Estimation: Estimate the quantity of each food item based on the bounding boxes and masks.
3. Nutritional Information: Provide the nutritional information for each detected food item, including calories, protein, fat, and carbohydrates.
4. Total Nutritional Summary: Calculate the total calories, protein, fat, and carbohydrates for all detected food items.
"""

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url},
            {"type": "text", "text": template},
        ],
    },
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
inputs = inputs.to("cuda:0", torch.float16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
