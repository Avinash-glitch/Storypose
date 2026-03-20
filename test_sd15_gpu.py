import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

MODEL_PATH = r"E:\stable-diffusion-webui\models\Stable-diffusion\v1-5-pruned-emaonly.safetensors"

print("Exists:", os.path.exists(MODEL_PATH))
print("Is file:", os.path.isfile(MODEL_PATH))
print("Path:", MODEL_PATH)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

print("Device:", device)
print("Dtype:", dtype)

pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=dtype,
    safety_checker=None,
    local_files_only=True,
)

pipe = pipe.to(device)

prompt = "a soft animated children's storybook illustration of a happy lion in a forest, warm colors, gentle lighting"

result = pipe(
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    width=512,
    height=512,
)

image = result.images[0]
image.save("out_base.png")
print("Saved out_base.png")