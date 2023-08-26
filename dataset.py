import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import random

PRETRAINED_MODEL_NAME = 'stabilityai/stable-diffusion-2-1-base'

