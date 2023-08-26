from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMPipeline
from diffusers import PNDMScheduler, UNet2DConditionModel, AutoencoderKL

PRETRAINED_MODEL = "google/ddpm-cat-256"
model = DDPMPipeline.from_pretrained(PRETRAINED_MODEL)

image = model(num_inference_steps = 25).images[0]
image.show()
# PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"

# image_encoder = AutoencoderKL.from_pretrained(PRETRAINED_MODEL, subfolder = 'vae')
# scheduler = PNDMScheduler.from_pretrained(PRETRAINED_MODEL, subfolder = 'scheduler')
# unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL, subfolder = 'unet')
# tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL, subfolder = 'tokenizer')
# text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL, subfolder = 'text_encoder')

# prompt = ["A cat on the table"]
# scheduler.set_timesteps(10)
# guidance_scale = 0.75
# batch_size = len(prompt)

# text_input = tokenizer(
#     prompt,
#     padding = 'max_length',
#     max_length = tokenizer.model_max_length,
#     truncation = True,
#     return_tensors = 'pt'
# )

# print(text_input)

# with torch.no_grad():
#     text_embedding = text_encoder(text_input.input_ids)[0]

# xt = torch.randn((1, 3, 256, 256))

# for t in scheduler.timesteps:
#     noise = unet(xt, t).sample
#     xt = scheduler.step(noise, t, xt).prev_sample

# def to_PIL_image(tensor):
#     image = (tensor / 2 + 0.5).clamp(0, 1).squeeze()
#     image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
#     image = Image.fromarray(image)

#     return image

# x0 = to_PIL_image(xt)   
# x0.show()
