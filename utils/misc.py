from random import randint
from re import X
import numpy as np
import torch
from PIL import Image
import cv2
from utils.graphics import BasicPointCloud, focal2fov, fov2focal
from scene.cameras import Camera
from arguments import CameraParams

# kernel = np.array([[ 0.05472157, 0.11098164, 0.05472157],
#                   [0.11098164, 0.22508352, 0.11098164],
#                   [0.05472157, 0.11098164, 0.05472157]])

kernel = np.array([[ 0.0947416, 0.118318, 0.0947416],
                  [0.118318, 0.147761, 0.118318],
                  [0.0947416, 0.118318, 0.0947416]])
x_kernel = np.array([[0,1,0],
                     [1,1,1],
                     [0,1,0]],np.uint8)

h_kernel = np.array([[0,0,0],
                     [1,1,1],
                     [0,0,0]],np.uint8)

v_kernel = np.array([[0,1,0],
                     [0,1,0],
                     [0,1,0]],np.uint8)


def pose2cam(render_pose, idx, white_background=False):
    ### Transform world to pixel
    Rw2i = render_pose[:3,:3]
    Tw2i = render_pose[:3,3:4]

    # Transfrom cam2 to world
    Ri2w = Rw2i.T
    Ti2w = -torch.matmul(Ri2w,  Tw2i)
    Pc2w = torch.concat((Ri2w, Ti2w), axis=1)
    Pc2w = torch.concat((Pc2w, torch.tensor([[0,0,0,1]])), axis=0)

    transform_matrix = Pc2w.tolist()

    c2w = np.array(transform_matrix)
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)

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

def add_new_pose(max_range, iter, max_iters):
    factor = int(iter*100 / max_iters)/100
    depth_to_c = 0 + max_range/3 * factor

    new_pose = torch.zeros([3,4],dtype=torch.float32)
    th = randint(0,360)
    th_rad = th / 180 * np.pi
    phi = randint(-10, 10)
    phi_rad = phi / 180 * np.pi
    Rot_H = torch.tensor([[np.cos(th_rad),0,-np.sin(th_rad)],[0,1,0],[np.sin(th_rad),0,np.cos(th_rad)]],dtype=torch.float32)
    Rot_V = torch.tensor([[1,0,0],[0,np.cos(phi_rad),-np.sin(phi_rad)],[0,np.sin(phi_rad),np.cos(phi_rad)]],dtype=torch.float32)
    new_pose[:3,:3] = Rot_H
    
    trans = torch.tensor([0,(max_range-depth_to_c)*np.tan(phi_rad) / 2,-depth_to_c],dtype=torch.float32).reshape(3,1)
    new_pose[:3,3:4] = trans

    th_2 = (randint(0,360))
    th_2_rad = th_2 / 180 * np.pi
    Rot_H_2 = torch.tensor([[np.cos(th_2_rad),0,-np.sin(th_2_rad)],[0,1,0],[np.sin(th_2_rad),0,np.cos(th_2_rad)]],dtype=torch.float32)
    LookArd = torch.matmul(Rot_V, Rot_H_2)
    new_pose[:3,:3] = torch.matmul(LookArd, new_pose[:3,:3])
    new_pose[:3,3:4] = torch.matmul(torch.linalg.inv(LookArd), new_pose[:3,3:4])

    return new_pose

def create_bottom(r, b, dense)-> BasicPointCloud:
    bx, bz = torch.meshgrid(torch.arange(dense, dtype=torch.float32), torch.arange(dense, dtype=torch.float32), indexing='xy')
    bx = bx.unsqueeze(-1) / dense * r * 2 - r
    bz = bz.unsqueeze(-1) / dense * r * 2 - r
    by = b * torch.ones_like(bx)

    pts = torch.concat([bx,by,bz],dim=-1).reshape(-1,3)
    pts = pts[pts.T[0]**2 + pts.T[2]**2 <= r**2]

    colors = torch.ones_like(pts) * 0.5

    pcd = BasicPointCloud(pts, colors, normals=None)

    return pcd


def add_anns(anns, origin_rgb):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    bg = origin_rgb.astype(np.float32)/255
    global_mask = np.zeros((bg.shape[0],bg.shape[1], 3),dtype=np.float32)
    alpha = np.zeros((bg.shape[0],bg.shape[1]))
    for ann in sorted_anns:
        m = ann['segmentation']
        mask_img = np.ones((m.shape[0], m.shape[1], 3),dtype=np.float32)
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            mask_img[:,:,i] = color_mask[i] * m * (1-alpha)
        global_mask = cv2.addWeighted(global_mask, 1, mask_img,1,0)
        alpha = np.logical_or(alpha, m)

    for i in range(3):
        bg[:,:,i] *= 0.65 * alpha + 1.0 * (1-alpha)
    result = cv2.addWeighted(bg,0.8, global_mask,0.35,0)
    return np.clip(result,0,1)

def save_d_img(d, save_path):
    d_max = torch.max(torch.from_numpy(d))
    d_img = Image.fromarray((d/d_max.numpy()*255).astype(np.uint8))
    d_img.save(save_path)