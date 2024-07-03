import os
import numpy as np
import torch
from PIL import Image
from dav2.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def save_d_img(d, save_path):
    d_max = torch.max(torch.from_numpy(d))
    d_img = Image.fromarray((d/d_max.numpy()*255).astype(np.uint8))
    d_img.save(save_path)

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
D_PATH = os.environ['HOME']+'/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small/snapshots/14cf9f3d82acd6b6c9b43fa50b79a639a4e69c8d'
model.load_state_dict(torch.load(f'{D_PATH}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


raw_img = np.array(Image.open(f'./pano_lr.png').convert('RGB'))
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
depth = depth.max() - depth + depth.min()
save_d_img(depth,"d.png")