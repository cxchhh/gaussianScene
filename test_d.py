import torch, os
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

#d_model = torch.hub.load('intel-isl/MiDaS', 'DPT_BEiT_L_512', pretrained=True,trust_repo=True).to('cuda')
in_img = 'inp.png'

img = Image.open("./"+in_img).convert('RGB')
# prepare image for the model
D_PATH = os.environ['HOME']+'/.cache/huggingface/hub/models--prs-eth--marigold-lcm-v1-0/snapshots/773825ffad4318356efcd14e3ff89d7812e5a0ab'
pipe = DiffusionPipeline.from_pretrained(
    D_PATH,
    local_files_only=True,
    custom_pipeline="marigold_depth_estimation",
    torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
    variant="fp16",                           # (optional) Use with `torch_dtype=torch.float16`, to directly load fp16 checkpoint
).to("cuda")

# interpolate to original size
depth = pipe(
    img
)
# depth = depth.reshape(*img.size[::-1])
# depth = depth.repeat(3,1,1).permute(1,2,0)
depth['depth_colored'].save('d.png')