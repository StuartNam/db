import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, AutoTokenizer
import os
from PIL import Image
import torchvision.transforms as transforms
import random
import torch.nn.functional as F 

FINETUNE_TEXT_ENCODER = True
PRETRAINED_MODEL_NAME = 'stabilityai/stable-diffusion-2-1-base'
INSTANCE_FOLDER_PATH = './data/valid/instances/'
PROMPT = 'A portrait of a sks person'
LRATE = 5e-6
WEIGHT_DECAY = 0
EPSILON = 0
NUM_EPOCHS = 50
PRIOR_LOSS_WEIGHT = 1
TEXT_ENCODER_CHECKPOINT_FOLDER_PATH = "./model/checkpoints/text_encoder/"
UNET_CHECKPOINT_FOLDER_PATH = "./model/checkpoints/unet/"

### 1. Prepare components of StableDiffusionPipeline

# - Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
    subfolder = 'tokenizer'
)

# - Text encoder: Used to encode input prompt into embedding. We can keep it fix or fine-tune it along with unet
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
    subfolder = 'text_encoder'
)

if not FINETUNE_TEXT_ENCODER:
    text_encoder.requires_grad_(False)

# - Scheduler
scheduler = DDPMScheduler.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
    subfolder = 'scheduler'
)

# - Autoencoder: Used to encode images into latent space. Keep it fix!
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
    subfolder = 'vae'
)
vae.requires_grad_(False)

# - Unet: Noise predictor. The heart of Diffusion-based generative models.
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
    subfolder = 'unet'
)

prior_model = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME
)

### 2. Prepare datasets
class LatentsDataset(Dataset):
    def __init__(self, instances_folder_path, instance_prompt, prior_model, image_encoder, size):
        def get_prior_class_prompt(instance_prompt, identifier):
            try:
                assert identifier in instance_prompt
                return instance_prompt.replace(identifier, "")
            except AssertionError as e:
                raise RuntimeError("at get_prior_prompt(): no -identifier- in -prompt-")

        image_encoder.eval()

        print("Preparing dataset")
        print("- Load tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
            subfolder = 'tokenizer'
        )

        print("- Tokenize <instance_prompt>")
        self.instance_prompt = instance_prompt

        self.instance_prompt_ids = tokenizer(
            self.instance_prompt,
            truncation = True,
            padding = "max_length",
            max_length = tokenizer.model_max_length,
            return_tensors = "pt",
        ).input_ids

        pre_process = transforms.Compose(
            [
                transforms.Resize(size, interpolation = transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        print("- Load instances")
        instances = []
        files = os.listdir(instances_folder_path)
        for file in files:
            file_path = os.path.join(instances_folder_path, file)
            image = Image.open(file_path)
            instances.append(pre_process(image))

        self.instances = instances
        self.num_instances = len(self.instances)

        print("- Encode instances to latent space")
        with torch.no_grad():
            self.latent_dists = [image_encoder.encode(instance.unsqueeze(0)).latent_dist for instance in self.instances]
        
        
        self.prior_class_prompt = get_prior_class_prompt(instance_prompt, 'sks')
        
        print("- Tokenize <prior_class_prompt>")
        self.prior_class_prompt_ids = tokenizer(
            self.prior_class_prompt,
            truncation = True,
            padding = "max_length",
            max_length = tokenizer.model_max_length,
            return_tensors = "pt",
        ).input_ids

        print("- Generate and encode prior_class_instances to latent space for prior preservation training")
        NUM_PRIOR_IMAGES = 1
        prior_class_images = prior_model([self.prior_class_prompt] * NUM_PRIOR_IMAGES, num_inference_steps = 15).images
        prior_class_images[0].show()

        self.prior_class_instances = [pre_process(image) for image in prior_class_images]
        with torch.no_grad():
            self.prior_latent_dicts = [image_encoder.encode(instance.unsqueeze(0)).latent_dist for instance in self.prior_class_instances]

        self.num_prior_class_instances = len(self.prior_class_instances)

        
    def __getitem__(self, index):
        prior_index = random.randint(0, self.num_prior_class_instances - 1)
        return {
            'latent': self.latent_dists[index].sample() * 0.18125,
            'instance_prompt_ids': self.instance_prompt_ids,
            'prior_latent': self.prior_latent_dicts[prior_index].sample() * 0.18125,
            'prior_class_prompt_ids': self.prior_class_prompt_ids
        }

    def __len__(self):
        return self.num_instances
    
dataset = LatentsDataset(
    instances_folder_path = INSTANCE_FOLDER_PATH,
    instance_prompt = PROMPT,
    prior_model = prior_model,
    image_encoder = vae,
    size = 512
)

### 3. Prepare training components

# - Optimizer(s): Unet optimizer and Text encoder optimizer if we are training Text encoder along with

unet_optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr = LRATE,
    weight_decay = WEIGHT_DECAY,
    eps = 1e-8
)

text_encoder_optimizer = torch.optim.AdamW(
    text_encoder.parameters(),
    lr = LRATE,
    weight_decay = WEIGHT_DECAY,
    eps = 1e-8
) if FINETUNE_TEXT_ENCODER else None

# - Dataloader
def collate_fn(batches):
    latents = []
    prior_latents = []

    for batch in batches:
        latents.append(batch['latent'])
        prior_latents.append(batch['prior_latent'])

    latents = torch.cat(latents)
    prior_latents = torch.cat(prior_latents)

    return {
        'latents': latents,
        'instance_prompt_ids': batches[0]['instance_prompt_ids'],
        'prior_latents': prior_latents,
        'prior_class_prompt_ids': batches[0]['prior_class_prompt_ids']
    }
    
data_loader = DataLoader(
    dataset = dataset,
    batch_size = 2,
    shuffle = False,
    collate_fn = lambda batches: collate_fn(batches)
)

### 4. Train

# Steps to train:
# 1. Get noisy latent vector using <scheduler>
# 2. Predict the noise added using <unet>

def train(start_from_epoch_no): 
    for epoch_no in tqdm.tqdm(range(start_from_epoch_no, NUM_EPOCHS), desc = "Training process", unit = ' epoch'):
        print()
        print(f"Start Epoch {epoch_no}")
        for batch_no, batch in tqdm.tqdm(enumerate(data_loader), desc = f"Epoch {epoch_no + 1}", unit = ' batch'):
            encoded_instance_prompt = text_encoder(batch['instance_prompt_ids'])[0]

            x0s = batch['latents']
            
            # - Sample white noise <epss> to add to x0
            epss = torch.randn_like(
                input = x0s,
                dtype = x0s.dtype,
                device = x0s.device
            )

            # - Sample <timesteps> T randomly
            ### Warning: <low> maybe 0
            ### 
            ###
            timesteps = torch.randint(
                low = 1,
                high = scheduler.config.num_train_timesteps,
                size = (x0s.shape[0], ),
                dtype = torch.long,
                device = x0s.device
            )

            xTs = scheduler.add_noise(x0s, epss, timesteps)
            predicted_epss = unet(xTs, timesteps, encoded_instance_prompt).sample
            instance_loss = F.mse_loss(predicted_epss, epss)
            
            prior_x0s = batch['prior_latents']

            encoded_prior_class_prompt = text_encoder(batch['instance_prompt_ids'])[0]

            prior_epss = torch.randn_like(
                input = prior_x0s,
                dtype = prior_x0s.dtype,
                device = prior_x0s.device
            )

            prior_xTs = scheduler.add_noise(prior_x0s, prior_epss, timesteps)
            predicted_prior_epss = unet(prior_xTs, timesteps, encoded_prior_class_prompt).sample
            prior_loss = F.mse_loss(predicted_prior_epss, prior_epss)

            loss = instance_loss + PRIOR_LOSS_WEIGHT * prior_loss

            text_encoder_optimizer.zero_grad()
            unet_optimizer.zero_grad()

            loss.backward()

            text_encoder_optimizer.step()
            unet_optimizer.step()

            if batch_no % 100 == 0:
                print(f"- Batch {batch_no}: Loss = {loss.detach().item()}" )

        if (epoch_no + 1) % 50 == 0:
            text_encoder.save_pretrained(TEXT_ENCODER_CHECKPOINT_FOLDER_PATH)
            unet.save_pretrained(UNET_CHECKPOINT_FOLDER_PATH)

train(start_from_epoch_no = 0)

if __name__ == "__main__":
    pass
