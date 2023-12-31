import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import os
from PIL import Image
import torchvision.transforms as transforms
import random
import torch.nn.functional as F 

# Argument list
FINETUNE_TEXT_ENCODER = True
PRETRAINED_MODEL_NAME = 'stabilityai/stable-diffusion-2-1-base'
INSTANCE_FOLDER_PATH = './data/valid/instances/'
PROMPT = 'A photo of a sks person\'s face'
LRATE = 5e-6
WEIGHT_DECAY = 0
EPSILON = 0
NUM_EPOCHS = 1000
PRIOR_LOSS_WEIGHT = 1
TEXT_ENCODER_CHECKPOINT_FOLDER_PATH = "./model/checkpoints/text_encoder/"
UNET_CHECKPOINT_FOLDER_PATH = "./model/checkpoints/unet/"
START_FROM_EPOCH_NO = 0
BATCH_SIZE = 1

# Dataset definition
from torch.utils.data import Dataset

class LatentsDataset(Dataset):
    def __init__(self, latents, prompt_ids):
        self.latents = latents
        self.prompt_ids = prompt_ids

    def __getitem__(self, index):
        return {
            'latent': self.latents[index],
            'prompt_ids': self.prompt_ids
        }

    def __len__(self):
        return len(self.latents)

class DBDataset(Dataset):
    def __init__(self, instance_dataset, prior_dataset):
        self.instance_dataset = instance_dataset
        self.prior_dataset = prior_dataset

    def __getitem__(self, index):
        import random

        prior_index = random.randint(0, len(self.prior_dataset) - 1)
        return {
            'latent': self.instance_dataset[index]['latent'],
            'instance_prompt_ids': self.instance_dataset[index]['prompt_ids'],
            'prior_latent': self.prior_dataset[prior_index]['latent'],
            'prior_prompt_ids': self.prior_dataset[prior_index]['prompt_ids']
        }
    
    def __len__(self):
        return len(self.instance_dataset)


def main():
    """
        1. Create accelerator object
    """
    from accelerate import Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = 3,
        mixed_precision = 'fp16'
    )

    """
        2. Create necessary directories for saving data and checkpoints
            - On main process only
            - If they already exist, no changes
    """
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

    if accelerator.is_main_process:
        accelerator.print(f"main_process - main(): Creating './model/...' for saving checkpoints ...")

        model_dir_info = {
            'model': {
                'checkpoints': {
                    'text_encoder': {},
                    'unet': {}
                }
            }
        }

        make_structured_dir(
            root_path = './',
            structured_dir_info = model_dir_info
        )

        accelerator.print(f"main_process - main(): Creating './data/...' for storing data ...")

        data_dir_info = {
            'data': {
                'valid': {
                    'instances': {},
                    'prior_class_instances': {}
                },
                'invalid': {}
            }
        }

        make_structured_dir(
            root_path = './',
            structured_dir_info = data_dir_info
        )

        accelerator.print(f"- Note: Put your instances into './data/valid/instances/' for training if you haven't. The invalid ones will be moved to './data/invalid/' automatically.")

        accelerator.print()

    accelerator.wait_for_everyone()

    """
        3. Prepare dataset on main process only
    """  
    accelerator.print(f"main_process - main(): Preparing dataset ...")
    def get_prior_class_prompt(instance_prompt, identifier):
        try:
            assert identifier in instance_prompt
            return instance_prompt.replace(identifier, "")
        except AssertionError as e:
            raise RuntimeError("at get_prior_prompt(): no -identifier- in -prompt-")
        
    if accelerator.is_main_process:
        accelerator.print(f"- Loading <tokenizer> from {PRETRAINED_MODEL_NAME}, subfolder 'tokenizer'")     
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
            subfolder = 'tokenizer'
        )

        accelerator.print(f"- Loading <vae> from '{PRETRAINED_MODEL_NAME}', subfolder 'vae'")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
            subfolder = 'vae'
        ).to(accelerator.device)

        vae.requires_grad_(False)

        print(f"- Loading <prior_model> from '{PRETRAINED_MODEL_NAME}'")
        prior_model = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path = PRETRAINED_MODEL_NAME
        ).to(accelerator.device)

        pre_process = transforms.Compose(
            [
                transforms.Resize(512, interpolation = transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        scaling_factor = vae.config.scaling_factor

        accelerator.print(f"- Loading and processing instances for training in './data/valid/instances'") 
        instance_prompt_ids = tokenizer(
            PROMPT,
            truncation = True,
            padding = "max_length",
            max_length = tokenizer.model_max_length,
            return_tensors = "pt",
        ).to(accelerator.device).input_ids        

        instances = []
        files = os.listdir(INSTANCE_FOLDER_PATH)
        for file in files:
            file_path = os.path.join(INSTANCE_FOLDER_PATH, file)
            image = Image.open(file_path)
            processed_image = pre_process(image).to(accelerator.device)
            instances.append(processed_image)

        with torch.no_grad():
            latent_dists = [vae.encode(instance.unsqueeze(0)).latent_dist for instance in instances]

        latents = [latent_dist.sample() * scaling_factor for latent_dist in latent_dists]

        instance_dataset = LatentsDataset(
            latents = latents,
            prompt_ids = instance_prompt_ids
        )

        accelerator.print(f"- Generating and processing prior_class_instances for training")
        prior_class_prompt = get_prior_class_prompt(PROMPT, 'sks')
        prior_class_prompt_ids = tokenizer(
            prior_class_prompt,
            truncation = True,
            padding = "max_length",
            max_length = tokenizer.model_max_length,
            return_tensors = "pt",
        ).to(accelerator.device).input_ids
        
        prior_class_images = []
        NUM_PRIOR_IMAGES = 1
        for i in range(NUM_PRIOR_IMAGES):
            prior_class_images += prior_model([prior_class_prompt], num_inference_steps = 25).images
        prior_class_images[0].show()

        prior_class_instances = [pre_process(image).to(accelerator.device) for image in prior_class_images]
        with torch.no_grad():
            prior_latent_dicts = [vae.encode(instance.unsqueeze(0)).latent_dist for instance in prior_class_instances]

        prior_latents = [prior_latent_dict.sample() * scaling_factor for prior_latent_dict in prior_latent_dicts]

        prior_dataset = LatentsDataset(
            latents = prior_latents,
            prompt_ids = prior_class_prompt_ids
        )

        accelerator.print(f"- Cleaning up pipelines and models")
        del vae, prior_model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        accelerator.print(f"- Finishing the dataset")
        dataset = DBDataset(
            instance_dataset = instance_dataset,
            prior_dataset = prior_dataset
        )

    accelerator.wait_for_everyone()

    """
        1. Setting up: 
        1.1. Prepare the folders if not yet done for training and inference, including:
        - './model/...':
            Used to store the saved checkpoints
        
        - './data/...':
            Used to store the data for training

        1.2. Prepare the components of Stable Diffusion, including:
        - Fixed components:
            . Tokenizer: 
                Used to tokenize input prompts for conditioning

            . Scheduler:
                Used to produce a noise schedule for fast inference

            . Autoencoder:
                Used to encode input images into latent space as latent vectors

            . Whole pipeline:
                Used to generate prior class samples for prior preservation training

        - Going-to-be-trained components:
            . UNet:
                Used to denoise the images followed the scheduler

            . Text encoder:
                Used to encode tokenized input prompts into embedding vectors
        
        1.3. Prepare dataset

        1.4. Prepare training components, including:
        - DataLoader
        - Text_encoder and UNet optimizers
        - Accelerator:
            Used to ...
    """

    # 1.
    # 1.1.
    

    # 1.2. 
    accelerator.print(f"main(): Preparing pipeline ...")

    def get_appropriate_pretrained(start_from_epoch_no):
        if start_from_epoch_no == 0:
            return PRETRAINED_MODEL_NAME
        
        needed_checkpoint = f'checkpoint-{start_from_epoch_no}'
        checkpoints = os.listdir('./model/checkpoints/text_encoder/')
        if needed_checkpoint not in checkpoints:
            raise RuntimeError(f"get_appropriate_pretrained(): checkpoint-{start_from_epoch_no} doesnot exist")
        
        return os.path.join('./model/checkpoints/', needed_checkpoint)

    pretrained_model = get_appropriate_pretrained(START_FROM_EPOCH_NO)

    # - Scheduler
    accelerator.print(f"- Loading <scheduler> from '{PRETRAINED_MODEL_NAME}', subfolder 'scheduler'")
    scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path = PRETRAINED_MODEL_NAME,
        subfolder = 'scheduler'
    )

    # - Text encoder: 
    accelerator.print(f"- Loading <text_encoder> from '{pretrained_model}'")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path = pretrained_model,
        subfolder = 'text_encoder'
    ).to(accelerator.device)

    # - UNet
    accelerator.print(f"- Loading <unet> from '{pretrained_model}', subfolder 'unet'")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path = pretrained_model,
        subfolder = 'unet'
    ).to(accelerator.device)

    # 1.4.

    # - Optimizers
    from torch.optim import AdamW
    unet_optimizer = AdamW(
        unet.parameters(),
        lr = LRATE,
        weight_decay = WEIGHT_DECAY,
        eps = 1e-8
    )

    text_encoder_optimizer = AdamW(
        text_encoder.parameters(),
        lr = LRATE,
        weight_decay = WEIGHT_DECAY,
        eps = 1e-8
    )

    # - DataLoader
    from torch.utils.data import DataLoader

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
            'prior_prompt_ids': batches[0]['prior_prompt_ids']
        }
        
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        collate_fn = lambda batches: collate_fn(batches)
    )

    (unet, text_encoder, dataloader, unet_optimizer, text_encoder_optimizer) = accelerator.prepare(
        unet,text_encoder, dataloader, unet_optimizer, text_encoder_optimizer
    )

    """
        2. Train DreamBooth for a personalized Stable Diffusion
            Steps to train:
                2.1. Encode <instance_prompt_ids>
                2.2. Sample <timesteps>
                2.2. For <latents: x0s> and <prior_latents: prior_x0s> :
                    - Sample <epss>
                    - Sample <xTs>
                    - Get <predicted_epss> by forwarding through <unet>
                    - Compute loss
                2.3. Compute loss by adding <instance_loss> and weighted <prior_loss>
                2.4. loss.backward() and optimizer.step()
    """

    # Set up
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)

    text_encoder.train()
    unet.train()

    # Train

    progress_bar = tqdm.tqdm(
        range(START_FROM_EPOCH_NO, NUM_EPOCHS), 
        desc = "Training process", 
        unit = ' epoch',
        disable = not accelerator.is_local_main_process
    )

    for epoch_no in progress_bar:    
        for batch_no, batch in enumerate(dataloader):
            with accelerator.accumulate(unet) and accelerator.accumulate(text_encoder):
                text_encoder_optimizer.zero_grad()
                unet_optimizer.zero_grad()
                
                instance_prompt_ids = batch['instance_prompt_ids']
                prior_class_prompt_ids = batch['prior_prompt_ids']
                latents = batch['latents']
                prior_latents = batch['prior_latents']

                encoded_instance_prompt = text_encoder(instance_prompt_ids)[0]
                encoded_instance_prompts = torch.cat([encoded_instance_prompt] * latents.shape[0], dim = 0)
                x0s = latents
                
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

                predicted_epss = unet(xTs, timesteps, encoded_instance_prompts).sample
                instance_loss = F.mse_loss(predicted_epss, epss)
                
                prior_x0s = prior_latents

                encoded_prior_class_prompt = text_encoder(prior_class_prompt_ids)[0]
                encoded_instance_prompts = torch.cat([encoded_prior_class_prompt] * BATCH_SIZE, dim = 0)

                prior_epss = torch.randn_like(
                    input = prior_x0s,
                    dtype = prior_x0s.dtype,
                    device = prior_x0s.device
                )

                prior_xTs = scheduler.add_noise(prior_x0s, prior_epss, timesteps)
                predicted_prior_epss = unet(prior_xTs, timesteps, encoded_prior_class_prompt).sample
                prior_loss = F.mse_loss(predicted_prior_epss, prior_epss)

                loss = instance_loss + PRIOR_LOSS_WEIGHT * prior_loss

                accelerator.backward(loss)

                text_encoder_optimizer.step()
                unet_optimizer.step()

                if batch_no % 3 == 0:
                    accelerator.print(f"- Batch {batch_no}: Loss = {loss.detach().item()}" )

        # Handle checkpoint saving
        if (epoch_no + 1) % 250 == 0:
            checkpoints_folder_path = './model/checkpoints/'
            text_encoder_checkpoint_path = os.path.join(checkpoints_folder_path, f'text_encoder/checkpoint-{epoch_no + 1}')
            unet_checkpoint_path = os.path.join(checkpoints_folder_path, f'unet/checkpoint-{epoch_no + 1}')

            text_encoder.save_pretrained(text_encoder_checkpoint_path)
            unet.save_pretrained(unet_checkpoint_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Finish training")

    accelerator.end_training()

if __name__ == "__main__":
    main()
