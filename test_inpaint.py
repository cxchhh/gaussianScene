import torch 
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting
import sys
import torch.nn.functional as F

seed = 1307
generator=torch.Generator(device='cuda').manual_seed(seed)
PATH = '/home/vrlab/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots/115134f363124c53c7d878647567d04daf26e41e'
sys.path.append(PATH)
rgb_model = AutoPipelineForInpainting.from_pretrained(
        PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        variant="fp16").to("cuda")
#rgb_model.set_progress_bar_config(disable=True)

prompt = "Autumn park, realistic, photography"
neg_prompt = ""
in_img = 'snapshot_1400.png'

img = Image.open(f'./imgs/{in_img}').convert('RGB')
w_in, h_in = img.size
#import pdb; pdb.set_trace()
img_ = np.array(img)
#img_[:,int(w_in/2):w_in,:] = 0
img = Image.fromarray(np.array(img_,dtype=np.uint8)).convert('RGB')

mask_in = np.ones((h_in, w_in, 3), dtype=np.uint8)
#mask_in[:,:int(w_in/2),:] = 0

mask = Image.fromarray(np.round((mask_in)*255.).astype(np.uint8))
#mask.save(f"./imgs/mask_{in_img}")
out_img = rgb_model(
                prompt=prompt,
                negative_prompt=neg_prompt,
                generator=generator,
                strength=0.3,
                guidance_scale=10.0,
                num_inference_steps=15,
                image=img,
                mask_image=mask
            ).images[0]
raw_img = torch.from_numpy(np.array(out_img)).float()
down_img = F.interpolate(raw_img.permute(2,0,1).unsqueeze(0), scale_factor=0.5).squeeze(0)
final_img = Image.fromarray(down_img.byte().permute(1,2,0).numpy())
final_img.save(f"./imgs/inpaint_{in_img}")