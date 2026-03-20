import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe = pipe.to("cuda")

image = pipe(
    "a soft animated children's storybook illustration of a happy lion in a forest",
    num_inference_steps=10,
    guidance_scale=6.5,
    width=384,
    height=384,
).images[0]

image.save("hf_sd15_test.png")