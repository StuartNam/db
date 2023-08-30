import torch
import os
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

print(f"main(): Creating './model/...' for saving checkpoints ...")

def make_structured_dir(root_path, structured_dir_info):
    """
        1. Args:
        - root_path: str
            A path where the dir should be created

        - structured_dir_info: dict
            Represents the structure of the dir
    """

    for dir, sub_dir in structured_dir_info.items():
        dir_path = os.path.join(root_path, dir)
        os.makedirs(dir_path, exist_ok = True)
        make_structured_dir(dir_path, sub_dir)

result_dir_info = {
    'result': {
        'db': {}
    }
}

make_structured_dir(
    root_path = './',
    structured_dir_info = result_dir_info
)


def get_appropriate_pretrained(checkpoint_no):
    if checkpoint_no == 0:
        return PRETRAINED_MODEL_NAME
    
    needed_checkpoint = f'checkpoint-{checkpoint_no}'
    checkpoints = os.listdir('./model/checkpoints/text_encoder/')
    if needed_checkpoint not in checkpoints:
        raise RuntimeError(f"get_appropriate_pretrained(): checkpoint-{checkpoint_no} doesnot exist")
    
    return needed_checkpoint

pretrained_model = get_appropriate_pretrained(1000)

PRETRAINED_MODEL_NAME = 'stabilityai/stable-diffusion-2-1-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"main(): Device: {device}")

# - Tokenizer
print(f"- Loading <tokenizer> from {PRETRAINED_MODEL_NAME}")
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
    subfolder = 'tokenizer'
)

# - Scheduler
print(f"- Loading <scheduler> from '{PRETRAINED_MODEL_NAME}', subfolder 'scheduler'")
scheduler = DDPMScheduler.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
    subfolder = 'scheduler'
)

print(f"- Loading <vae> from '{PRETRAINED_MODEL_NAME}', subfolder 'vae'")
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
    subfolder = 'vae'
).to(device)

vae.requires_grad_(False)

# - Whole pipeline as a prior model
print(f"- Loading <prior_model> from '{PRETRAINED_MODEL_NAME}'")
prior_model = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME
).to(device)

# - Text encoder: 
print(f"- Loading <text_encoder> from '{pretrained_model}'")
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path = './model/checkpoints/text_encoder/' + pretrained_model
).to(device)

# if not FINETUNE_TEXT_ENCODER:
#     text_encoder.requires_grad_(False)

# - UNet
print(f"- Loading <unet> from '{pretrained_model}'")
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path = './model/checkpoints/unet/' + pretrained_model
).to(device)

model = StableDiffusionPipeline(
    vae = vae,
    text_encoder = text_encoder,
    tokenizer = tokenizer,
    unet = unet,
    scheduler = scheduler,
    safety_checker = None,
    feature_extractor = None
)

prompt = ['A photo of sks person']
image = model(prompt = prompt, num_inference_steps = 30).images[0]

save_path = './output/db/'
image.save(save_path)