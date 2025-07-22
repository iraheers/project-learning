from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model_id = "llava-hf/llava-1.5-7b-hf"
model_id = "liuhaotian/llava-v1.5-7b"

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image_file = "images/000000039769.jpg"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
     model_id,
     torch_dtype=torch.float16
).to(device)


raw_image = Image.open(image_file)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))