import requests
import torch
from transformers import pipeline
from PIL import Image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
pipeline = pipeline(
    task="depth-estimation",
    model="Intel/zoedepth-nyu-kitti",
    dtype=torch.float16,
    device=0
)
results = pipeline(image)
#print(results[].keys())
#results['predicted_depth'].save("outputs/cat-depth.png") 
results['depth'].save("outputs/cat-depth.png") 

print(results['predicted_depth'])
print(results['depth'])