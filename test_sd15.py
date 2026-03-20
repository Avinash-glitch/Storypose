import os
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import numpy as np

MODEL_PATH = r"E:\stable-diffusion-webui\models\Stable-diffusion\v1-5-pruned-emaonly.safetensors"
# If it's inside a models folder, use:
# MODEL_PATH = r"E:\storypose\models\v1-5-pruned-emaonly.safetensors"

print("Exists:", os.path.exists(MODEL_PATH))
print("Is file:", os.path.isfile(MODEL_PATH))
print("Path:", MODEL_PATH)

# Force float32 to avoid fp16 precision NaNs on some GPUs
# (especially older/nvidia 16xx series and some driver combinations).
dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Dtype:", dtype)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=dtype,
    safety_checker=None,
    local_files_only=True,
)

# Try a different scheduler to avoid NaNs during denoising
try:
    scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    print("Using LMSDiscreteScheduler")
except Exception as e:
    print("Could not switch scheduler:", e)

pipe = pipe.to(device)

# Scan key submodules for NaN/Inf in parameters (StableDiffusionPipeline has no named_parameters())
def scan_module(name, module):
    for pname, param in module.named_parameters():
        if not torch.isfinite(param).all():
            print(f"Bad parameter detected: {name}.{pname} -> has NaN/Inf")
            return True
    return False

has_bad = False
for mod_name, mod in [
    ("unet", pipe.unet),
    ("vae", pipe.vae),
    ("text_encoder", pipe.text_encoder),
]:
    if scan_module(mod_name, mod):
        has_bad = True
        break

if not has_bad:
    print("No NaN/Inf found in model parameters")

# Inspect a couple of weights to ensure the model loaded correctly
unet_weight = pipe.unet.conv_in.weight
print("UNet conv_in stats -> min", float(unet_weight.min()), "max", float(unet_weight.max()), "mean", float(unet_weight.mean()))

vae_weight = pipe.vae.decoder.conv_in.weight
print("VAE conv_in stats -> min", float(vae_weight.min()), "max", float(vae_weight.max()), "mean", float(vae_weight.mean()))

prompt = "a soft animated children's storybook illustration of a happy lion in a forest, warm colors, gentle lighting"

generator = torch.Generator(device=device).manual_seed(42)

# Generate output as numpy array so we can inspect before any casting
output = pipe(
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    width=512,
    height=512,
    generator=generator,
    output_type="np",
)

np_images = output.images
print("Output images type:", type(np_images), "len:", len(np_images))
print("Per-image dtype:", np_images[0].dtype)

arr = np_images[0]
print("Raw output stats -> min", arr.min(), "max", arr.max(), "mean", arr.mean())
print("Has NaN:", np.isnan(arr).any())
print("Has Inf:", np.isinf(arr).any())

# Convert to PIL and save
from PIL import Image
pil = Image.fromarray(np.clip(arr * 255, 0, 255).astype("uint8"))
pil.save("out_base.png")
print("Saved out_base.png")