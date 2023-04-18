from typing import Callable, List, Tuple, Union
from torch import nn
from helper_functions import *
import memcnn

from transformers import AutoProcessor, AutoModel
import sys
from torchvision import transforms
from fgvc_ws_dan_helpers.inception_bap import inception_v3_bap

from os.path import expanduser  
from urllib.request import urlretrieve  
import open_clip


torch.autograd.set_detect_anomaly(False)
torch.set_grad_enabled(False)

class SteppingLayer(nn.Module):
    """
    This is a layer that performs DDIM stepping that will be wrapped
    by memcnn to be invertible
    """
    
    def __init__(self, unet,
                 embedding_uc,
                 embedding_c,
                 scheduler=None,
                 num_timesteps=50,
                 guidance_scale=7.5,
                 clip_cond_fn=None,
                 single_variable=False
                ):
        super(SteppingLayer, self).__init__()
        self.unet = unet
        self.e_uc = embedding_uc
        self.e_c = embedding_c
        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale
        if scheduler is None:
            self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                         beta_schedule="scaled_linear",
                                      num_train_timesteps=1000,
                                         clip_sample=False,
                                      set_alpha_to_one=False)
        else:
            self.scheduler = scheduler
        self.scheduler.set_timesteps(num_timesteps)
        
        self.clip_cond_fn = clip_cond_fn
        
        self.single_variable = single_variable
        
        
    def forward(self, i, t, latent_pair,
               reverse=False):
        """
        Run an EDICT step
        """
        for base_latent_i in range(2): 
            # Need to alternate order compatibly forward and backward
            if reverse:
                orig_i = self.num_timesteps - (i+1) 
                offset = (orig_i+1) % 2
                latent_i = (base_latent_i + offset) % 2
            else:
                offset = i%2
                latent_i = (base_latent_i + offset) % 2

            # leapfrog steps/run baseline logic hardcoded here
            latent_j = ((latent_i+1) % 2)
            
            latent_i = latent_i.long()
            latent_j = latent_j.long()
            
            if self.single_variable:
                # If it's the single variable baseline then just operate on one tensor
                latent_i = torch.zeros(1, dtype=torch.long).to(device)
                latent_j = torch.zeros(1, dtype=torch.long).to(device)

            # select latent model input
            if base_latent_i==0:
                latent_model_input = latent_pair.index_select(0, latent_j)
            else:
                latent_model_input = first_output
            latent_base = latent_pair.index_select(0, latent_i)

            #Predict the unconditional noise residual
            noise_pred_uncond = self.unet(latent_model_input, t[0], 
                                     encoder_hidden_states=self.e_uc).sample

            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = self.unet(latent_model_input, t[0], 
                                   encoder_hidden_states=self.e_c).sample
            # Get classifier free guidance term
            grad = (noise_pred_cond - noise_pred_uncond)
            noise_pred = noise_pred_uncond + self.guidance_scale * grad
            
            # incorporate classifier guidance if applicable
            if self.clip_cond_fn is not None:
                clip_grad = self.clip_cond_fn(latent_model_input, t.long(), 
                                             scheduler=self.scheduler)
                alpha_prod_t, beta_prod_t = get_alpha_and_beta(t.long(), self.scheduler)
                fac = beta_prod_t ** 0.5 
                noise_pred = noise_pred - fac * clip_grad 


            # Going forward or backward?
            step_call = reverse_step if reverse else forward_step
            # Step
            new_latent = step_call(self.scheduler,
                                      noise_pred,
                                        t[0].long(),
                                        latent_base)
            new_latent = new_latent.to(latent_base.dtype)

            
            # format outputs using index order
            if self.single_variable:
                combined_outputs = torch.cat([new_latent, new_latent])
                break
            
            if base_latent_i == 0: # first pass
                first_output = new_latent
            else: # second pass
                second_output = new_latent
                if latent_i==1: # so normal order
                    combined_outputs = torch.cat([first_output, second_output])
                else: # Offset so did in reverse
                    combined_outputs = torch.cat([second_output, first_output])
        
        return i.clone(), t.clone(), combined_outputs
    
    def inverse(self, i, t, latent_pair):
        # Inverse method for memcnn
        output = self.forward(i, t, latent_pair, reverse=True)
        return output
    
    
    
class MixingLayer(nn.Module):
    """
    This does the mixing layer of EDICT 
    https://arxiv.org/abs/2211.12446
    Equations 12/13
    """
    
    def __init__(self, mix_weight=0.93):
        super(MixingLayer, self).__init__()
        self.p = mix_weight
        
    def forward(self, input_x):
        input_x0, input_x1 = input_x[:1], input_x[1:]
        x0 = self.p*input_x0 + (1-self.p)*input_x1
        x1 = (1-self.p)*x0 + self.p*input_x1
        return torch.cat([x0, x1])
    
    def inverse(self, input_x):
        input_x0, input_x1 = input_x.split(1)
        x1 = (input_x1 - (1-self.p)*input_x0) / self.p
        x0 = (input_x0 - (1-self.p)*x1) / self.p
        return torch.cat([x0, x1])
        


class MakeCutouts(nn.Module):
    """
    boiler plate multicrop for model guidance
    https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py
    """
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)
    
def spherical_dist_loss(x, y): 
    """
    spherical distance for classifier guidance
    https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    dist =  (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    if dist.numel() > 1: # gave matrix of features
        return dist.min()
    return dist






def clip_loss_fn(prompt=None,
                 clip_viz_paths=None,
                 use_cutouts=False,
                       num_cuts=64,
                       cut_power=0.6,
                 grad_scale=0,
                clip_model_str='vit_b_32',
                 weights=[1], # order is b/l/g, not related to above str order
                ):
    """
    CLIP conditioning function
    
    prompt: text prompt to match (Str)
    clip_viz_paths: list of paths (str) to images to use as visual guidance
    use_cutouts / num_cuts / cut_power: Multicrop settings
    grad_scale: Scale of guidance
    clip_model_str: CLIP model to guide with, options below
    weights (list of flloats): Weighting of clip models if multiple
    """

    # Get clip models and tokenizers
    model_paths_clip = []
    if clip_model_str=='all':
        clip_model_str = 'vit_b_32 + vit_l_14 + vit_g_14'

    if 'vit_b_32' in clip_model_str:
        model_paths_clip.append('laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    elif 'vit_l_14' in clip_model_str:
        model_paths_clip.append('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    elif 'vit_g_14' in clip_model_str:
        model_paths_clip.append('laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    else:
        raise NotImplementedError
    guiding_clip_tokenizers = [AutoProcessor.from_pretrained(model_path_clip).tokenizer
                               for model_path_clip in model_paths_clip]
    guiding_clip_models = [AutoModel.from_pretrained(model_path_clip, torch_dtype=torch_dtype).to(device)
                           for model_path_clip in model_paths_clip]

    # Make data processing
    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    if use_cutouts:
        cut = MakeCutouts(cut_size=target_size,
                          cut_power=cut_power)
        
    with torch.no_grad():
        if prompt is not None:
            if len(guiding_clip_models) > 1:
                raise NotImplementedError # for text only guide with one model, could easily be changed
            guiding_clip_model = guiding_clip_models[0]
            assert clip_viz_paths is None # don't have multiple guidance
            prompts = [prompt]
            text_inputs = [guiding_clip_tokenizer(prompts,
                                       padding='max_length',     
                                     max_length=clip_tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors='pt')
                           for guiding_clip_tokenizer in guiding_clip_tokenizers]
            feed_text_inputs = [text_input.input_ids.to(guiding_clip_model.device)# .long()
                               for text_input in text_inputs]
            with autocast(device):
                z_ts = [guiding_clip_model.get_text_features(feed_text_input)[0]
                       for feed_text_input in feed_text_inputs]
            z_ts = [z_t / z_t.norm(p=2, dim=-1, keepdim=True)
                    for z_t in z_ts]
        elif clip_viz_paths is not None: # visual guidance
            load_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(target_size),
                                                 transforms.CenterCrop(target_size),
                                                 normalize])
            # collect guiding ims
            ims = []
            for p in clip_viz_paths:
                im = Image.open(p)
                if len(im.getbands())==1:
                    print(f"Skipping grayscale at {p}")
                    continue
                ims.append(load_transform(im).to(guiding_clip_models[0].dtype))
            
            ims = torch.stack(ims).to(device)
            
            z_ts = [guiding_clip_model.get_image_features(ims)
                   for guiding_clip_model in guiding_clip_models]
            z_ts = [z_t / z_t.norm(p=2, dim=-1, keepdim=True)
                    for z_t in z_ts]
            z_ts = [z_t.mean(dim=0, keepdim=True)
                    for z_t in z_ts]
            z_ts = [z_t / z_t.norm(p=2, dim=-1, keepdim=True)
                    for z_t in z_ts]

    # gets vae decode as input           
    def loss_fn(im_pix):
        # Prep image
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        if use_cutouts:
            x_var = cut(im_pix, num_cuts)
        else:
            x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(vae.dtype)
        # Image features
        z_is = [guiding_clip_model.get_image_features(x_var)
                for guiding_clip_model in guiding_clip_models]
        z_is = [z_i / z_i.norm(p=2, dim=-1, keepdim=True)
                for z_i in z_is]
        # Distances --> losses for each model
        dists_arr = [spherical_dist_loss(z_i, z_t)
                     for z_i,z_t in zip(z_is, z_ts)]
        losses = [dists.mean() for dists in dists_arr]
        loss = sum([l*w for l,w in zip(losses,weights)]) / sum(weights)
        return loss * grad_scale
    
    return loss_fn
        

def symmetry_loss_fn(symmetry_type, grad_scale=0):
    """
    Fun throw-in, works sometimes. Make generation have horizontal symmetry
    Probably want perturb_grad_scale=0 for this one and high grad_scale
    Didn't do formal experimentation with
    """
    def loss_fn(im_pix):
        
        if symmetry_type == 'horizontal':
            loss = F.mse_loss(im_pix, im_pix.flip(-1)) # could make L1
        else:
            raise NotImplementedError

        return grad_scale * loss

    return loss_fn



def fgvc_loss_fn(dataset,
                   class_idx=0,
                 use_cutouts=False,
                       num_cuts=64,
                       cut_power=0.6,
                 grad_scale=0):
    """
    Guide with FGVC model from https://github.com/wvinzh/WS_DAN_PyTorch
    
    dataset (str): 'aircraft' (FGVC-Aircraft), 'bird' (CUB), 'dog' (Stanford Dogs)
    """
    # Models take in high resolution
    target_size = 512
    
    ### number of classes in each dataset
    num_class_dict = {'aircraft':100,
                      'bird':200,
                      'dog':120}
    
    # Get network 
    net = inception_v3_bap(pretrained=True, aux_logits=False)
    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=num_class_dict[dataset])
    net.fc_new = new_linear
    net = net.to(device)
    ckpt = torch.load(f'fgvc_ws_dan_helpers/checkpoint/{dataset}.pth.tar')
    sd = {k.replace('module.',''):v for k,v in ckpt['state_dict'].items()}
    net.load_state_dict(sd)
    net.eval()
    


    # expected transform/multicrop
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
    if use_cutouts:
        cut = MakeCutouts(cut_size=target_size,
                          cut_power=cut_power)
        
    # gets vae decode as input           
    def loss_fn(im_pix):
        # prep image
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        if use_cutouts:
            x_var = cut(im_pix, num_cuts)
        else:
            x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(vae.dtype)
        
        with autocast(device):
            # last output is actual classification output, basic recognition objective
            _, _, output = net(x_var)
        # target is class idx
        target = (class_idx * torch.ones(output.size(0))).to(output.device).long()
        # Simple cross entropy loss
        loss = torch.nn.functional.cross_entropy(output, target)
        return grad_scale * loss

    return loss_fn
    

def get_generic_cond_fn(loss_fn,
                         embedding_unconditional,
                         embedding_conditional,
                         sd_guidance_scale=7.5,
                          steps=50,
                          mix_weight=0.93,
                        one_step_baseline=False,
                        keep_input=False, 
                       ):
    """
    Generic method that returns a function which given torch latents
    * Denoises them to images
    * Calculates loss on images
    * Backpropagates loss through denoising process to get grad w.r.t latents
    
    Args:
    * loss_fn: function with torch pixels in --> scalaar loss out
    * embedding_(un)conditional: SD embeddings as standard in diffusers
    * sd_guidance_scale: classifier free guidance scale
    * steps: Number diffusion timesteps
    * mix_weight: EDICT mixing layer weight
    * one_step_baseline: Whether to run only one step of denoising, this is standard classifier guidance
    * keep_input: A memcnn performance arg, recommend not fiddling with
    """
    # Initialize mixing layer and wrap in Memcnn wrapper for constant-memory backprop
    internal_mix = MixingLayer(mix_weight)
    internal_mix = memcnn.InvertibleModuleWrapper(internal_mix, keep_input=keep_input,
                                                  keep_input_inverse=keep_input,
                                               num_bwd_passes=2)
    # Initialize stepping layer and warp in MeMcnn
    internal_s = SteppingLayer(unet,
                      embedding_unconditional,
                      embedding_conditional,
                      guidance_scale=sd_guidance_scale,
                     num_timesteps=steps,
                     clip_cond_fn=None) # Want this to just be normal diffusion
    internal_s = memcnn.InvertibleModuleWrapper(internal_s, keep_input=keep_input,
                                                  keep_input_inverse=keep_input,
                                               num_bwd_passes=2)
    # Get timesteps from stepping layer
    timesteps = internal_s._fn.scheduler.timesteps 
  


    def diffuse_latents_from_t_to_0(input_latent, start_i):
        """
        Given noisy latents at time (start_i) denoise fully to x_0
        """
        # Format latent input
        if isinstance(input_latent, list):
            latent_pair = torch.cat(input_latent).clone()
        elif input_latent.shape[0]==2: # have both in here
            latent_pair = input_latent.clone()
        else: 
            # If just a single latent then double it
            # Could probably do repeat here? I think I was just hedging against grad implications
            latent_pair = torch.cat([input_latent.clone(), input_latent.clone()])
            
        # Just run diffusion
        with autocast(device):
            for i, t in tqdm(enumerate(timesteps[start_i:]), total=len(timesteps)-start_i,
                            disable=True):
                # get timestep indexing
                i = torch.tensor([i], dtype=torch_dtype, device=latent_pair.device)
                t = torch.tensor([t], dtype=torch_dtype, device=latent_pair.device)
                # take an EDICT denoise step
                i, t, latent_pair = internal_s(i, t, latent_pair)
                # take an EDICT averaging step
                latent_pair = internal_mix(latent_pair)
        return latent_pair
    
    # Define the conditioning function to return
    def cond_fn(x, t,
                scheduler):
        """
        Function that will give gradient for latents
        
        x: latents
        t: timestep
        scehduler: ddim scheduler
        """
        
        with torch.enable_grad(): # turn backprop back on
            # Clone latent and turn on grad
            l = x.clone().detach().requires_grad_(True)
            # timestep indexing
            start_i = (scheduler.timesteps==t.item()).nonzero()[0].item()
            
            
            if one_step_baseline: # standard classifier guidance
                # https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py#L143
                # make a noise pred
                internal_noise_pred = unet(l, t, 
                                       encoder_hidden_states=embedding_unconditional).sample
                # get alpha and beta
                alpha_prod_t, beta_prod_t = get_alpha_and_beta(t.long(), scheduler)
                fac = beta_prod_t ** 0.5
                # take a forward step
                pred_original_sample = (l - (fac)*internal_noise_pred) / (alpha_prod_t**0.5)
                sample = (pred_original_sample * (fac)) + (l * (1 - fac))
                pred_original_pair = sample.repeat(2, 1, 1, 1)
            else: # DOODL
                # call out to full-chain diffusion function we defined
                # Normally would take way too much memory but invertible + memcnn!
                pred_original_pair = diffuse_latents_from_t_to_0(l, start_i)
                
                
            grads = [] # EDICT tracks two latents, take grads w.r.t. both 
            for sample in pred_original_pair.chunk(2):
                # get image pixels
                im_pix = vae.decode(sample.to(vae.dtype) / 0.18215).sample
                # handle pixel normalization/transformation in the loss functions
                loss = loss_fn(im_pix) 
                loss.backward()
                if one_step_baseline: break # only have 1 image we're using
            # Trying to decrease loss so negative of gradient, divide by number of copies
            # grad is same shape as latents
            return (-1 if one_step_baseline else -0.5) * l.grad 
        
    return cond_fn
                
    
# from https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md
def get_aesthetic_model(clip_model="vit_l_14"):
    """
    Get an aesthetic scoring model based off of clip vit_l_14 or clip vit_b_32
    
    """
    # Download to cache folder
    # Aesthetic model is simple linear layer on top of CLIP stem
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def aesthetic_score(image, model, amodel):
    """
    Get aesthetic score of image (possibly stack of images from multicrop)
    Inputs:
    * image (bs, 3, 224, 224) tensor
    * model: clip feature extractor
    * amodel: linear head
    
    Output:
    * Single scalar score
    """
    with autocast(device):
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = amodel(image_features).clip(1, 10).mean()
        return prediction

def aesthetic_loss_fn(aesthetic_target=None,
                      use_cutouts=False,
                       num_cuts=64,
                       cut_power=0.6,
                 grad_scale=0,
                     clip_model_str='vit_l_14',
                     weights=[1,1]):
    """
    Loss function for aesthetics
    
    Inputs:
    * aesthetic value to target in 1-10. If None will maximize aesthetic vlaue
    * use_cutouts / num_cuts / cut_power: Multicrop boiler plate
    * grad_scale: Scale of loss gradient
    * clip_model_str: vit_l_14 or vit_b_32 or 'both' , which aesthetic model to use
    * weights (list of floats): Weights of vit_b_32 vs vit_l_14 if using 'both'
    """
    # https://github.com/LAION-AI/aesthetic-predictor
    
    # Image processing
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    if use_cutouts:
        cut = MakeCutouts(cut_size=target_size,
                          cut_power=cut_power)
        
    # Create normal clip model stems
    if clip_model_str == 'both':
        model_l, _, _ = open_clip.create_model_and_transforms('ViT-L-14',
                                                            pretrained='openai')
        model_l = model_l.cuda()
        amodel_l = get_aesthetic_model(clip_model='vit_l_14').cuda()
        amodel_l.eval()
        model_b, _, _ = open_clip.create_model_and_transforms('ViT-B-32',
                                                            pretrained='openai')
        model_b = model_b.cuda()
        amodel_b = get_aesthetic_model(clip_model='vit_b_32').cuda()
        amodel_b.eval()
        models = [model_l, model_b]
        amodels = [amodel_l, amodel_b]
    else:
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14' if clip_model_str=='vit_l_14' else 'ViT-B-32',
                                                            pretrained='openai')
        model = model.cuda()
        amodel = get_aesthetic_model(clip_model=clip_model_str).cuda()
        amodel.eval()
        models = [model]
        amodels = [amodel]
        
    # gets vae decode as input  
    def loss_fn(im_pix): # Loss function on aesthetic value
        # Process pixels and multicrop
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        if use_cutouts:
            x_var = cut(im_pix, num_cuts)
        else:
            x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(vae.dtype)

        # Get predicted scores from model(s)
        predictions = [aesthetic_score(x_var, model, amodel)
                       for model,amodel in zip(models,amodels)]
        # Average predictions across models
        prediction = sum([w*p for w,p in zip(weights,predictions)]) / len(predictions)
        if aesthetic_target is None: # default maximization
            loss = -1 * prediction
        else:
            # using L1 to keep on same scale
            loss = abs(prediction - aesthetic_target)

        return loss * grad_scale

    return loss_fn



def gen(
    cond_prompt = 'a dog',
    model_guidance_type = 'CLIP', # CLIP or imagenet or .... or .. 
    model_guidance_dict = {'clip_prompt':'a dog', 'clip_viz_paths':None},
    save_str = 'tmp.png',
    grad_scale = 1, 
    sd_guidance_scale = 7.5,
    use_cutouts = True,
    num_cuts = 16,
    cut_power = 0.3,
    latent_seed = None, # for starting point only
    seed = 0, # more general seed that both cuts and noise for traversal off of
    steps = 50, #20
    mix_weight = 0.93, #.7
    null_prompt='',
    single_variable=False,
    latent_traversal=True,
    num_traversal_steps=50,
    save_interval=1,
    one_step_baseline=False,
    tied_latents=True, 
    use_momentum = True,
    use_nesterov = False,
    renormalize_latents=True,
    optimize_first_edict_image_only=False,
    opt_t=None,
    perturb_grad_scale=1e-4,
    clip_grad_val=1e-3,
    source_im=None
    ):
    """
    Our main function which takes in all of the conditioning and hyperparameters
    It saves a series of images to the ims/ directory
    
    cond_prompt (str): text conditioning for stable diffusion
    model_guidance_type(str): type of classifier guidance (fgvc, CLIP, aesthetic) 
    model_guidance_dict (dict): kwargs for model_guidance_type (these are used in the below code)
    save_str (str): Save path in ims/ sequentially optimized images will replace '.png' with '_0.png' _1.png' etc
    sd_guidance_scale (float): stablediffusion guidance scale
    grad_scale (float): Amount loss is multiplied by before calculating gradients, basically learning rate
    use_cutouts (bool): Whether to use crops to determine guidance
    num_cuts (int): Number of crops
    cut_power (float): Strength of crops
    latent_seed = (int): Seed for initial latent
    seed (int): Seed for operations (noisy traversal, cuts, etc)
    steps (int): Num DDIM/EDICT steps
    mix_weight (float): EDICT mixing layer parameter
    null_prompt (str): Null conditioning (UC prompt) for stable diffusion
    single_variable (bool): Run single variable generation (not EDICT)
    latent_traversal (Bool): Optimize latent instead of guiding diffusion process (DOODL)
    num_traversal_steps (int): how many traversal (optimization) steps to run
    save_interval (int): Interval to save generations from traversal
    one_step_baseline ( bool ): Take only 1 denoising step, this is normal classifier guidance baseline
    tied_latents (bool): Average coupled latents together after each traversal step to avoid drift
    use_momentum / use_nesterov: Momentum args for SGD
    renormalize_latents (bool): Renormalize latents to original norm after each step
    optimize_first_edict_image_only (bool): only compute grad for first EDICT image
    opt_t (int): Timestep to optimize at (t=T if None)
    perturb_grad_scale (float): Amount of noise to add in each optimization step
    clip_grad_val (float): Max gradient element magnitude
    source_im (str): Path to image to edit 
    """
    # SGD logic
    if use_nesterov:
        use_momentum = True
    # Mem cnn arg
    keep_input = True
    # Latent setup
    if latent_seed is None:
        latent_seed = seed
    generator = torch.cuda.manual_seed(latent_seed)
    
    if source_im is None: # novel generation, not editing
        latent = torch.randn((1, 4, 64, 64),
                                generator=generator,
                                device=device,
                               dtype=torch_dtype,
                            requires_grad=True)
        latent_pair = torch.cat([latent.clone(), latent.clone()])
    else: 
        assert not renormalize_latents
    if renormalize_latents: # if renormalize_latents then get original norm value
        orig_norm = latent.norm().item()
   
    # random generator boilerplate
    generator = torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Text embedding setup
    with autocast(device):
        tokens_unconditional = clip_tokenizer(null_prompt, padding="max_length",
                                              max_length=clip_tokenizer.model_max_length,
                                              truncation=True, return_tensors="pt", 
                                              return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        tokens_conditional = clip_tokenizer(cond_prompt, padding="max_length", 
                                            max_length=clip_tokenizer.model_max_length,
                                            truncation=True, return_tensors="pt", 
                                            return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

        embedding_unconditional = embedding_unconditional.to(torch_dtype)
        embedding_conditional = embedding_conditional.to(torch_dtype)

     # model guidance
    if grad_scale != 0:
        # Logic by guidance type
        if model_guidance_type=='CLIP':  # text or visual
            
            # Get loss function with args from model_guidance_dict
            loss_fn = clip_loss_fn(prompt=model_guidance_dict.get('clip_prompt', None),
                                     clip_viz_paths=model_guidance_dict.get('clip_viz_paths', None),
                                     use_cutouts=use_cutouts,
                                   num_cuts=num_cuts,
                                   cut_power=cut_power,
                                     grad_scale=grad_scale,
                                  clip_model_str=model_guidance_dict.get('clip_model_str', 'vit_g_14'),
                                 )
   
        elif model_guidance_type=='fgvc': # FGVC 
            loss_fn = fgvc_loss_fn(model_guidance_dict['dataset'],
                                   class_idx=model_guidance_dict.get('class', 0),
                                     use_cutouts=use_cutouts,
                                           num_cuts=num_cuts,
                                           cut_power=cut_power,
                                     grad_scale=grad_scale)
       
        elif model_guidance_type=='aesthetic': # easthetic
            loss_fn = aesthetic_loss_fn(grad_scale=grad_scale,
                                       aesthetic_target=model_guidance_dict.get('aesthetic_target'),
                                       clip_model_str=model_guidance_dict.get('clip_model_str', 'vit_l_14'),
                                     use_cutouts=use_cutouts,
                                           num_cuts=num_cuts,
                                           cut_power=cut_power,
                                       weights=model_guidance_dict.get('weights', [1,1]),
                                       )
    
        elif model_guidance_type=='symmetry': # Not in paper but can symmetrize gens
            loss_fn = symmetry_loss_fn(symmetry_type=model_guidance_dict.get('symmetry_type',
                                                                                  'horizontal'),
                                          grad_scale=grad_scale)
        else:
            raise NotImplementedError
        
        # Construct conditioning function
        cond_fn = get_generic_cond_fn(loss_fn,
                                        embedding_unconditional,
                                        embedding_conditional,
                                        sd_guidance_scale=sd_guidance_scale,
                                        steps=steps,
                                        mix_weight=mix_weight,
                                        one_step_baseline=one_step_baseline,
                                        keep_input=keep_input,
                                        )
    else: # if no guidance, just generation
        cond_fn = None

    # Make layers for diffusion process
    
    # EDICT mixing layers
    if single_variable:
        mix = torch.nn.Identity()
    else:
        mix = MixingLayer(mix_weight)
        mix = memcnn.InvertibleModuleWrapper(mix, keep_input=keep_input,
                                             
                                           keep_input_inverse=keep_input,
                         num_bwd_passes=1)

    if latent_traversal: 
        if single_variable:
            raise NotImplementedError # Doesn't work in memory, could try one-step traversal but would be bad

        # make diffusion steps with no model conditioning function (Still has text conditioning tho)
        s = SteppingLayer(unet,
                          embedding_unconditional,
                          embedding_conditional,
                          guidance_scale=sd_guidance_scale,
                         num_timesteps=steps,
                         clip_cond_fn=None,
                         single_variable=single_variable)
        s = memcnn.InvertibleModuleWrapper(s, keep_input=keep_input,
                                           keep_input_inverse=keep_input,
                         num_bwd_passes=1)
        timesteps = s._fn.scheduler.timesteps

        # SGD boiler plate
        if use_momentum: prev_b_arr = [None, None]
        
        if source_im is not None: # image editng
            assert opt_t is None # assert we're optimizing at x_T
            
            # Image loading
            source_im = load_im_into_format_from_path(source_im)
            source_im = source_im.resize((512, 512), resample=Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            source_im = np.array(source_im) / 255.0 * 2.0 - 1.0
            source_im = torch.from_numpy(source_im[np.newaxis, ...].transpose(0, 3, 1, 2))
            if source_im.shape[1] > 3:
                source_im = source_im[:, :3] * source_im[:, 3:] + (1 - source_im[:, 3:])
            source_im = source_im.to(device).to(unet.dtype)
            
            # Peform reverse diffusion process
            with autocast(device):
                init_latent = vae.encode(source_im).latent_dist.sample(generator=generator) * 0.18215
 
                latent_pair = init_latent.repeat(2, 1, 1, 1)
                # Iterate through reversed tiemsteps using EDICT
                for i, t in tqdm(enumerate(timesteps.flip(0)), total=len(timesteps)):
                    i = torch.tensor([i],
                                     dtype=torch_dtype,
                                     device=latent_pair.device)
                    t = torch.tensor([t],
                                     dtype=torch_dtype,
                                     device=latent_pair.device)
                    latent_pair = mix.inverse(latent_pair)
                    i, t, latent_pair = s.inverse(i, t, latent_pair)
           
        elif opt_t is not None: # Partial optimization
            
            # Denoise T - opt_t diffusion timesteps before optimizing
            initial_timesteps = timesteps[:-1*opt_t] if opt_t!=0 else timesteps
            timesteps = timesteps[-1*opt_t:] if opt_t !=0 else []
            with autocast(device):
                for i, t in tqdm(enumerate(initial_timesteps),
                                 total=len(initial_timesteps)):
                    i = torch.tensor([i],
                                     dtype=torch_dtype,
                                     device=latent_pair.device)
                    t = torch.tensor([t],
                                     dtype=torch_dtype,
                                     device=latent_pair.device)
                    i, t, latent_pair = s(i, t, latent_pair)
                    latent_pair = mix(latent_pair)
            # will have drifted with above in EDICT, this might corrupt image at intermediate noise levvels
            if tied_latents: 
                combined_l = 0.5 * (latent_pair[0] + latent_pair[1])
                latent_pair = combined_l.repeat(2, 1, 1, 1)
            
            
        """
        PERFORM GRADIENT DESCENT DIRECTLY ON LATENTS USING GRADIENT CALCUALTED THROUGH WHOLE CHAIN
        """
        # turn on gradient calculation
        with torch.enable_grad(): # important b/c don't have on by default in module
            for m in range(num_traversal_steps): # This is # of optimization steps
                print(f"Optimization Step {m}")
                
                # Get clone of latent pair
                orig_latent_pair = latent_pair.clone().detach().requires_grad_(True)
                input_latent_pair = orig_latent_pair.clone()
                # Full-chain generation using EDICCT
                with autocast(device):
                    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                        i = torch.tensor([i],
                                         dtype=torch_dtype,
                                         device=latent_pair.device)
                        t = torch.tensor([t],
                                         dtype=torch_dtype,
                                         device=latent_pair.device)
                        i, t, input_latent_pair = s(i, t, input_latent_pair)
                        input_latent_pair = mix(input_latent_pair)
                    # Get the images that the latents yield
                    ims = [vae.decode(l.to(vae.dtype) / 0.18215).sample
                           for l in input_latent_pair.chunk(2)]

                # save images and compute loss
                # save image to ims/{save_str.replace(.png, _m.png) 
                losses = []
                for im_i,im in enumerate(ims): 
                    
                    # Get formatted images and save one of them
                    pil_im = prep_image_for_return(im.detach())
                    sub_str = f'_{m}.png' 
                    mod_save_str = save_str.replace('.png', sub_str)
                    
                    # save
                    if ( (m%save_interval)==0 or m==(num_traversal_steps-1) ) and im_i==0: 
                        pil_im.save(f'ims/{mod_save_str}')
                        
                    # If guiding then compute loss    
                    if grad_scale!=0:
                        loss = loss_fn(im)
                        losses.append(loss)
                        if optimize_first_edict_image_only: break
                sum_loss = sum(losses)
                
                # Backward pass
                sum_loss.backward()
                # Access latent gradient directly
                grad = -0.5 * orig_latent_pair.grad
                # Average gradients if tied_latents
                if tied_latents:
                    grad = grad.mean(dim=0, keepdim=True)
                    grad = grad.repeat(2, 1, 1, 1)

                new_latents = []
                # doing perturbation linked as well
                # Perturbation is just random noise added
                perturbation = perturb_grad_scale * torch.randn_like(orig_latent_pair[0]) if perturb_grad_scale else 0
                
                # SGD step (S=stochastic from multicrop, can also just be GD)
                # Iterate through latents/grads
                for grad_idx, (g, l) in enumerate(zip(grad.chunk(2), orig_latent_pair.chunk(2))):
                    
                    # Clip max magnitude
                    if clip_grad_val is not None:
                        g = g.clip(-clip_grad_val, clip_grad_val)
                        
                    # SGD code
                    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
                    if use_momentum: 
                        mom = 0.9
                        # LR is grad scale
                        # sticking with generic 0.9 momentum for now, no dampening
                        if m==0:
                            b = g
                        else:
                            b = mom * prev_b_arr[grad_idx] + g
                        if use_nesterov:
                            g = g + mom * b
                        else:
                            g = b
                        prev_b_arr[grad_idx] = b.clone()
                    new_l = l + g + perturbation
                    new_latents.append(new_l.clone())
                if tied_latents:  # don't think is needed with other tied_latent logic but just being safe
                    combined_l = 0.5 * (new_latents[0] + new_latents[1])
                    latent_pair = combined_l.repeat(2, 1, 1, 1)
                else:
                    latent_pair = torch.cat(new_latents)
                
                if renormalize_latents: # Renormalize latents 
                    for norm_i in range(2):
                        latent_pair[norm_i] = latent_pair[norm_i] * orig_norm / latent_pair[norm_i].norm().item()
            # Once we've run all of our optimization steps we're done
            return

        
    else:
        # not doing traversal (this is for baseline) or method halfway between baseline and DOODL 
        # where you get the gradient w.r.t. the final true generation and incorporate into denoising
        assert (not use_momentum) # don't re-use grad across nosie levels
        
        
        # Layer to perform EDICT steps
        # The conditioning function (cond_fn) internally performs the diffusion process to get grads
        s = SteppingLayer(unet,
                          embedding_unconditional,
                          embedding_conditional,
                          guidance_scale=sd_guidance_scale,
                         num_timesteps=steps,
                         clip_cond_fn=None,
                         single_variable=single_variable)
        s = memcnn.InvertibleModuleWrapper(s, keep_input=keep_input,
                                           keep_input_inverse=keep_input,)
        timesteps = s._fn.scheduler.timesteps
        
        
        
        if source_im is not None: # same initialization logic as above
            if opt_t is not None:
                timesteps = timesteps[opt_t:]
            source_im = load_im_into_format_from_path(source_im)
            source_im = source_im.resize((512, 512), resample=Image.Resampling.LANCZOS)
            source_im = np.array(source_im) / 255.0 * 2.0 - 1.0
            source_im = torch.from_numpy(source_im[np.newaxis, ...].transpose(0, 3, 1, 2))
            if source_im.shape[1] > 3:
                source_im = source_im[:, :3] * source_im[:, 3:] + (1 - source_im[:, 3:])
            source_im = source_im.to(device).to(unet.dtype)
            with autocast(device):
                init_latent = vae.encode(source_im).latent_dist.sample(generator=generator) * 0.18215
 
                latent_pair = init_latent.repeat(2, 1, 1, 1)
                for i, t in tqdm(enumerate(timesteps.flip(0)), total=len(timesteps)):
                    i = torch.tensor([i],
                                     dtype=torch_dtype,
                                     device=latent_pair.device)
                    t = torch.tensor([t],
                                     dtype=torch_dtype,
                                     device=latent_pair.device)
                    # latent_pair = mix.inverse(latent_pair)
                    i, t, latent_pair = s.inverse(i, t, latent_pair)
                    # print(latent_pair)

                    
        
        s = SteppingLayer(unet,
                          embedding_unconditional,
                          embedding_conditional,
                          guidance_scale=sd_guidance_scale,
                         num_timesteps=steps,
                         clip_cond_fn=cond_fn,
                         single_variable=single_variable)
        s = memcnn.InvertibleModuleWrapper(s, keep_input=keep_input,
                                           keep_input_inverse=keep_input,)
        # simply do the diffusion process, the conditioning function of s incorporates guidance
        with autocast(device):
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                i = torch.tensor([i], dtype=torch_dtype, device=latent_pair.device)
                t = torch.tensor([t], dtype=torch_dtype, device=latent_pair.device)
                i, t, latent_pair = s(i, t, latent_pair)
                latent_pair = mix(latent_pair)
        # decode and save
        decoded_latent = vae.decode(latent_pair[:1].to(vae.dtype) / 0.18215).sample
        pil_im = prep_image_for_return(decoded_latent.detach())
        if save_str: pil_im.save(f'ims/{save_str}')
    return 

