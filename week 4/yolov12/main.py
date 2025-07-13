from ultralytics import YOLO

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

file_name = "fruits.jpg"

model_yolo = YOLO("yolov12s-seg.pt")
results = model_yolo.predict("test_images/"+file_name, save=True, task="segment")

results[0].show()
saved_path = results[0].save_dir + "/"+file_name

detected_items = []
names = results[0].names

print("Starting loop")
# get masks
masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None

for i, (box, cls, conf) in enumerate(zip(
    results[0].boxes.xyxy.cpu().numpy(),
    results[0].boxes.cls.cpu().numpy(),
    results[0].boxes.conf.cpu().numpy()
)):
    mask_area = None
    if masks is not None:
        mask = masks[i]          # shape: (H, W)
        mask_area = int(mask.sum())   # total number of foreground pixels

    item = {
        "class_name": names[int(cls)],
        "confidence": float(conf),
        "box": box.tolist(),
        "mask_area_pixels": mask_area
    }
    detected_items.append(item)
    
print("Ending loop")

detected_text = f"Detected items from YOLO: {detected_items}"

# Step 3: Build the full prompt
template = f"""
You are an expert in food recognition and nutritional information.

An image of food items with bounding boxes and segmentation masks is provided.
{detected_text}

Your response should have 4 sections:
1. Detected Food Items: List all the food items detected in the image.
2. Quantity Estimation: Estimate the quantity of each food item based on the bounding boxes and masks.
3. Nutritional Information: Provide the nutritional information for each detected food item, including calories, protein, fat, and carbohydrates.
4. Total Nutritional Summary: Calculate the total calories, protein, fat, and carbohydrates for all detected food items.
"""


messages = [
    HumanMessage(
        content=[
            {
                "type": "text",
                "text": template
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

from pathlib import Path

save_dir = Path(results[0].save_dir)
output_file = save_dir / "llava_output.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)