import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif


ckpt_id = "openai/shap-e"
pipe = ShapEPipeline.from_pretrained(ckpt_id).to("cuda")


guidance_scale = 15.0
prompt = "a shark"
images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    size=256,
).images

gif_path = export_to_gif(images, "shark_3d.gif")