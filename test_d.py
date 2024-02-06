import torch
import numpy as np
from PIL import Image
d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
in_img = 'room.jpg'

img = Image.open(f'./imgs/{in_img}').convert('RGB')

d = d_model.infer_pil(img)

d_max = torch.max(torch.from_numpy(d))
d = d/d_max.numpy()
d = (d*255).astype(np.uint8)
d_img = Image.fromarray(d)
d_img.save(f'./imgs/d_{in_img}')