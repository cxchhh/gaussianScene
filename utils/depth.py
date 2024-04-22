import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512",local_files_only=True)
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512",local_files_only=True)


def infer_depth(image_pil,h_in,w_in) -> torch.Tensor:
    inputs = processor(images=image_pil, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h_in,w_in),
        mode="bicubic",
        align_corners=False,
    )
    depth = depth.reshape(h_in,w_in) / 500.
    depth = (depth.max() - depth) + depth.min()
    #depth = depth.repeat(3,1,1).permute(1,2,0)
    return depth