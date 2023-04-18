# Official Implementation of DOODL (End-to-End Diffusion Latent Optimization Improves Classifier Guidance)

[Arxiv](https://arxiv.org/abs/2303.13703)



# What is DOODL?

DOODL (Direct Optimization of Diffusion Latents) is a variant of classifier guidance that directly optimizes diffusion latents `x_T` instead of using model-based gradients to guide denoising. This is done be leveraging the [EDICT](https://arxiv.org/abs/2211.12446) algorithm and [MemCNN](https://github.com/silvandeleemput/memcnn) library to construct a diffusion process that can be backpropagated through with constant memory cost w.r.t the number of diffusion steps without significant runtime increase. The control of this optimization allows a variety of guidance modes to be incorporated. Check out our [paper](https://arxiv.org/abs/2303.13703) for more details and don't hesitate to reach out with questions!


# Setup

## HF Auth token

Paste a copy of a suitable [HF Auth Token](https://huggingface.co/docs/hub/security-tokens) into [hf_auth](hf_auth) with no new line (to be read by the following code in `edict_functions.py`)
```
with open('hf_auth', 'r') as f:
    auth_token = f.readlines()[0].strip()
    
```

Example file at `./hf_auth`
```
abc123abc123
```

## Environment

Run  `conda env create -f environment.yaml`, activate that conda env (`conda activate doodl`). Run jupyter with that conda env active

## FGVC models

FGVC models can be downloaded from the [WS-DAN repo](https://github.com/wvinzh/WS_DAN_PyTorch#result) and saved at `fgvc_ws_dan_helpers/checkpoints/`

# Experimentation

Check out [this notebook](demo.ipynb) for examples of how to use DOODL.



# Other Files

* [doodl.py](doodl.py) has the core functionality of DOODL
* [my_half_diffusers](my_half_diffusers) is a very slightly changed version of the [HF Diffusers repo](https://github.com/huggingface/diffusers)
* [fgvc_ws_dan_helpers/](fgvc_ws_dan_helpers/) gives access to the [WSDAN Model](https://github.com/wvinzh/WS_DAN_PyTorch).
* [memcnn/](memcnn) is a very lightly modified version of the excellent [MemCNN](https://github.com/silvandeleemput/memcnn) library. Thank you to the original MemCNN authors!
* 


# Citation

If you find our work useful in your research, please cite the following works:

```
@misc{wallace2023endtoend,
      title={End-to-End Diffusion Latent Optimization Improves Classifier Guidance}, 
      author={Bram Wallace and Akash Gokul and Stefano Ermon and Nikhil Naik},
      year={2023},
      eprint={2303.13703},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{wallace2022edict,
  title={EDICT: Exact Diffusion Inversion via Coupled Transformations},
  author={Wallace, Bram and Gokul, Akash and Naik, Nikhil},
  journal={arXiv preprint arXiv:2211.12446},
  year={2022}
}
```

# License

Our code is BSD-3 licensed. See LICENSE.txt for details.

