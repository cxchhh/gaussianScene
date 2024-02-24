import numpy as np
import torch
from PIL import Image

from utils.graphics import focal2fov, fov2focal
from scene.cameras import Camera
from arguments import CameraParams

# kernel = np.array([[ 0.05472157, 0.11098164, 0.05472157],
#                   [0.11098164, 0.22508352, 0.11098164],
#                   [0.05472157, 0.11098164, 0.05472157]])

kernel = np.array([[ 0.0947416, 0.118318, 0.0947416],
                  [0.118318, 0.147761, 0.118318],
                  [0.0947416, 0.118318, 0.0947416]])

yz_reverse = torch.tensor([[1,0,0], [0,-1,0], [0,0,-1]],dtype=torch.float32)

def pose2cam(render_pose, idx, white_background=False):
    ### Transform world to pixel
    Rw2i = render_pose[:3,:3]
    Tw2i = render_pose[:3,3:4]

    # Transfrom cam2 to world + change sign of yz axis
    Ri2w = torch.matmul(yz_reverse, Rw2i).T
    Ti2w = -torch.matmul(Ri2w, torch.matmul(yz_reverse, Tw2i))
    Pc2w = torch.concat((Ri2w, Ti2w), axis=1)
    Pc2w = torch.concat((Pc2w, torch.tensor([[0,0,0,1]])), axis=0)

    transform_matrix = Pc2w.tolist()

    c2w = np.array(transform_matrix)
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    cam = CameraParams()

    null_image = Image.fromarray(np.zeros([cam.H,cam.W,3]).astype(np.uint8))
    im_data = np.array(null_image.convert("RGBA"))

    bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

    norm_data = im_data / 255.0
    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
    loaded_mask = np.ones_like(norm_data[:, :, 3:4])

    fovx = cam.fov[0]
    fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
    FovY = fovy 
    FovX = fovx

    image = torch.Tensor(arr).permute(2,0,1)
    loaded_mask = None #torch.Tensor(loaded_mask).permute(2,0,1)
    
    return Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, 
                            gt_alpha_mask=loaded_mask, image_name='', uid=idx, data_device='cuda')