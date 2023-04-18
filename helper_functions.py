# Imports
import torch
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
import math
import imageio
import torchvision
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image, ImageOps
import time
import datetime
import torch
import sys
import os
import pickle
from my_half_diffusers import AutoencoderKL, UNet2DConditionModel
from my_half_diffusers.schedulers.scheduling_utils import SchedulerOutput
from my_half_diffusers import LMSDiscreteScheduler, PNDMScheduler, DDPMScheduler, DDIMScheduler
import random
from tqdm.auto import tqdm
from torch import autocast
torch_dtype = torch.float16 
np_dtype = np.float16 

# Shoutout to  bloc97's https://github.com/bloc97/CrossAttentionControl


# alpha and beta for DDIM
def get_alpha_and_beta(t, scheduler):
    # want to run this for both current and previous timnestep
    if t<0:
        return scheduler.final_alpha_cumprod.item(), 1 - scheduler.final_alpha_cumprod.item()
    
    if t.dtype==torch.long  or (t==t.long()):
        alpha = scheduler.alphas_cumprod[t.long()]
        return alpha.item(), 1-alpha.item()
    

    
    low = t.floor().long()
    high = t.ceil().long()
    rem = t - low
    
    low_alpha = scheduler.alphas_cumprod[low]
    high_alpha = scheduler.alphas_cumprod[high]
    interpolated_alpha = low_alpha * rem + high_alpha * (1-rem)
    interpolated_beta = 1 - interpolated_alpha
    return interpolated_alpha.item(), interpolated_beta.item()
    

# forward DDIM step
def forward_step(
    self,
    model_output,
    timestep: int,
    sample,
) :
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps

    if timestep > self.timesteps.max():
        raise NotImplementedError("Need to double check what the overflow is")
  

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)
    
    
    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev)**0.5)
    first_term =  (1./alpha_quotient) * sample
    second_term = (1./alpha_quotient) * (beta_prod_t ** 0.5) * model_output
    third_term = ((1 - alpha_prod_t_prev)**0.5) * model_output
    return first_term - second_term + third_term
                
# reverse ddim step
def reverse_step(
    self,
    model_output,
    timestep: int,
    sample,
) :
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps
   
    if timestep > self.timesteps.max():
        raise NotImplementedError
    else:
        alpha_prod_t = self.alphas_cumprod[timestep]
        
    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)
    
    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev)**0.5)
    
    first_term =  alpha_quotient * sample
    second_term = ((beta_prod_t)**0.5) * model_output
    third_term = alpha_quotient * ((1 - alpha_prod_t_prev)**0.5) * model_output
    return first_term + second_term - third_term  
 


def prep_image_for_return(image): # take torch image and return PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image

def center_crop(im): # PIL center_crop
    width, height = im.size   # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim)/2
    top = (height - min_dim)/2
    right = (width + min_dim)/2
    bottom = (height + min_dim)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_into_format_from_path(im_path): # From path get formatted PIL image
    return center_crop(ImageOps.exif_transpose(Image.open(im_path)) if isinstance(im_path, str) else im_path).resize((512,512))


# CLIP model for text conditioning
model_path_clip = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch_dtype)
clip = clip_model.text_model

# HF authentication
with open('hf_auth', 'r') as f:
    auth_token = f.readlines()[0].strip()

# Using SD 1.4
model_path_diffusion = "CompVis/stable-diffusion-v1-4"

# Initialize SD models
unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch_dtype)
vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch_dtype)

# Push to device
device = 'cuda'
unet.to(device)
vae.to(device)
clip.to(device)

print("Loaded all models")

