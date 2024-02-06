import torch 
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting

seed = 1307
generator=torch.Generator(device='cuda').manual_seed(seed)
rgb_model = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                       torch_dtype=torch.float16, variant="fp16").to("cuda")

prompt = "a photo of cabin in the woods, realistic"
neg_prompt = ""
in_img = 'cabin.png'

img = Image.open(f'./imgs/{in_img}').convert('RGB')
w_in, h_in = img.size
#import pdb; pdb.set_trace()
img_ = np.array(img)
img_[:,int(w_in/2):w_in,:] = 0
img = Image.fromarray(np.array(img_,dtype=np.uint8)).convert('RGB')

mask_in = np.ones((h_in, w_in, 3), dtype=np.uint8)
mask_in[:,:int(w_in/2),:] = 0

mask = Image.fromarray(np.round((mask_in)*255.).astype(np.uint8))
mask.save(f"./imgs/masks/mask_{in_img}")
out_img = rgb_model(
                prompt=prompt,
                negative_prompt=neg_prompt,
                generator=generator,
                strength=1.0,
                guidance_scale=8.0,
                num_inference_steps=30,
                image=img,
                mask_image=mask
            ).images[0]
out_img.save(f"./imgs/masks/inpaint_{in_img}")