import os
from random import randint, uniform
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from arguments import CameraParams, GSParams
from scene.gaussian_model import GaussianModel
from diffusers import AutoPipelineForInpainting
from gaussian_renderer import render
from utils.graphics import BasicPointCloud
from utils.loss import l1_loss, ssim
from scene.dataset_readers import readDataInfo
from utils.misc import kernel
from tqdm import tqdm

from utils.trajectory import get_pcdGenPoses
opt = GSParams()
cam = CameraParams()
gaussian_model = GaussianModel(opt.sh_degree)
background = torch.tensor([0,0,0], dtype=torch.float32, device='cuda')

seed = 1307
generator=torch.Generator(device='cuda').manual_seed(seed)
rgb_model = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16").to("cuda")
rgb_model.set_progress_bar_config(disable=True)

d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')

def save_d_img(d, save_path):
    d_max = torch.max(torch.from_numpy(d))
    d_img = Image.fromarray((d/d_max.numpy()*255).astype(np.uint8))
    d_img.save(save_path)

prompt = "Chinese garden, realistic, photography"
neg_prompt = ""
h_in, w_in = 512, 512
prompt_path = prompt.replace(" ","_")
os.system(f"mkdir -p ./gs_checkpoints/{prompt_path}")

N = 15 # camera pose nums
render_poses = torch.zeros(N,3,4)
yz_reverse = torch.tensor([[1,0,0], [0,-1,0], [0,0,-1]],dtype=torch.float32)
frames = []
for i in range(N):
    th = i*360/N
    th_rad = th/180*np.pi
    render_poses[i,:3,:3] = torch.tensor([[np.cos(th_rad),0,-np.sin(th_rad)],[0,1,0],[np.sin(th_rad),0,np.cos(th_rad)]],dtype=torch.float32)
    render_poses[i,:3,3:4] = torch.tensor([0,0,0],dtype=torch.float32).reshape(3,1)

    ### Transform world to pixel
    Rw2i = render_poses[i,:3,:3]
    Tw2i = render_poses[i,:3,3:4]

    # Transfrom cam2 to world + change sign of yz axis
    Ri2w = torch.matmul(yz_reverse, Rw2i).T
    Ti2w = -torch.matmul(Ri2w, torch.matmul(yz_reverse, Tw2i))
    Pc2w = torch.concat((Ri2w, Ti2w), axis=1)
    Pc2w = torch.concat((Pc2w, torch.tensor([[0,0,0,1]])), axis=0)

    frames.append({
        'image': Image.fromarray(np.zeros([h_in,w_in,3]).astype(np.uint8)), # just a placeholder
        'transform_matrix': Pc2w.tolist(),
    })


H,W,K = cam.H,cam.W,cam.K
x, y = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy') # pixels

traindata = {
            'camera_angle_x': cam.fov[0],
            'W': W,
            'H': H,
            'pcd_points': torch.zeros((3,1),dtype=torch.float),
            'pcd_colors':  torch.zeros((1,3),dtype=torch.float),
            'frames': frames,
        }

# set training cameras
info = readDataInfo(traindata, opt.white_background)
cameras_extent = info.nerf_normalization["radius"]
gt_images = [None] * N

#init gaussians
gaussian_model.create_from_pcd(info.point_cloud, cameras_extent)


viewpoint_stack = info.train_cameras.copy()
for i in tqdm(range(N),desc="creating point cloud from poses"):
    cam_i = viewpoint_stack[i]

    render_pkg = render(cam_i, gaussian_model, opt, background)
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
    
    del gaussian_model # for saving VRAM
    
    #import pdb; pdb.set_trace()
    image_u8 = (255*image.clone().detach().cpu().permute(1,2,0)).byte()
    image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
    mask = image_u8.float()
    mask_l = mask[:,:,0]*0.299+mask[:,:,1]*0.587+mask[:,:,2]*0.114
    mask[mask_l<1.]=255.
    mask[mask_l>=1.]=0.
    mask_u8 = cv2.dilate(mask.byte().numpy(), kernel, iterations=5)
    
    mask_pil = Image.fromarray(mask_u8).convert('RGB')
    
    inpainted_image = rgb_model(
                prompt=prompt,
                negative_prompt = neg_prompt,
                image=image_pil,
                mask_image=mask_pil,
                guidance_scale=8.0,
                strength=1.0, 
                generator=generator,
                num_inference_steps=30
            ).images[0]
    
    # manually downsample from 1024 to 512, because the params (height,width)=(512,512) to the pipeline give bad result
    raw_img = torch.from_numpy(np.array(inpainted_image))
    down_img = F.interpolate(raw_img.permute(2,0,1).unsqueeze(0), scale_factor=0.5).squeeze(0).permute(1,2,0)
    inpainted_image = Image.fromarray(down_img.numpy())

    gt_images[i] = (torch.from_numpy(np.array(inpainted_image))/255.).permute(2,0,1).float()

    depth_np = d_model.infer_pil(inpainted_image)
    depth = torch.from_numpy(depth_np)

    #import pdb; pdb.set_trace()
    Ri, Ti = render_poses[i,:3,:3], render_poses[i,:3,3:4]
    pts_coord_cam_i = torch.matmul(torch.linalg.inv(K), torch.stack((x*depth, y*depth, 1*depth), axis=0).reshape(3,-1))

    pts_coord_world_curr = (torch.linalg.inv(Ri).matmul(pts_coord_cam_i) - torch.linalg.inv(Ri).matmul(Ti).reshape(3,1)).float()
    pts_colors_curr = (torch.from_numpy(np.array(inpainted_image)).reshape(-1,3).float()/255.)

    new_pts_coord_world = pts_coord_world_curr.T[mask_u8.reshape(-1,3) > 0].reshape(-1,3)
    new_pts_colors = pts_colors_curr[mask_u8.reshape(-1,3) > 0].reshape(-1,3)

    if i == 0 :
        new_global_pts = new_pts_coord_world.cpu()
        new_global_colors = new_pts_colors.cpu()
    else:
        new_global_pts = torch.concat([new_global_pts, new_pts_coord_world.cpu()], axis=0)
        new_global_colors = torch.concat([new_global_colors, new_pts_colors.cpu()], axis=0)

    new_global_pcd = BasicPointCloud(new_global_pts, new_global_colors,normals=None)
    
    gaussian_model= GaussianModel(opt.sh_degree)
    gaussian_model.create_from_pcd(new_global_pcd, cameras_extent)

    # import pdb; pdb.set_trace()
    
#del rgb_model # for saving VRAM
mask_all = np.ones((h_in, w_in, 3), dtype=np.uint8)
mask_all_pil = Image.fromarray(np.round((mask_all)*255.).astype(np.uint8))

# train gaussians
gaussian_model.training_setup(opt)

for iteration in tqdm(range(1, opt.iterations+1), desc="training gaussians"):#
    gaussian_model.update_learning_rate(iteration)

    if iteration % 1000 == 0:#
        gaussian_model.oneupSHdegree()
    
    # Pick a Camera
    viewpoint_stack = info.train_cameras.copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    
    # Render
    render_pkg = render(viewpoint_cam, gaussian_model, opt, background)
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
    
    # get refined G.T.
    if iteration % 50 == 0:
        image_u8 = (255*image.clone().detach().cpu().permute(1,2,0)).byte()
        image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
        new_gt_pil =  rgb_model(
                prompt=prompt,
                negative_prompt=neg_prompt,
                generator=generator,
                strength=0.7,
                guidance_scale=5,
                num_inference_steps=30,
                image=image_pil,
                mask_image=mask_all_pil
            ).images[0]

        # manually downsample from 1024 to 512, because the params height, width=512 to the pipeline give bad result
        raw_img = torch.from_numpy(np.array(new_gt_pil)).float()
        down_img = F.interpolate(raw_img.permute(2,0,1).unsqueeze(0), scale_factor=0.5).squeeze(0)
        new_gt = down_img/255.
        gt_images[viewpoint_cam.uid] = new_gt

        image_pil.save(f"scene_imgs/{iteration}.png")
        new_gt_pil.save(f"scene_imgs/gt_{iteration}.png")


    gt_image = gt_images[viewpoint_cam.uid].to("cuda")

    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss.backward()

    with torch.no_grad():
        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussian_model.max_radii2D[visibility_filter] = torch.max(
                gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussian_model.densify_and_prune(
                    opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
            
            if (iteration % opt.opacity_reset_interval == 0 
                or (opt.white_background and iteration == opt.densify_from_iter)
            ):
                gaussian_model.reset_opacity()

        # Optimizer step
        if iteration < opt.iterations:
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none = True)
    
    if (iteration) % 500 == 0:
        #save gaussians
        gaussian_model.save_ply(f'./gs_checkpoints/{prompt_path}/{prompt}.ply')

        #save snapshot
        image_u8 = (255*image.clone().detach().cpu().permute(1,2,0)).byte()
        image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
        imgpath=f'./gs_checkpoints/{prompt_path}/{prompt}_snapshot_{(iteration)}.png'
        # print(imgpath)
        image_pil.save(imgpath)

gaussian_model.save_ply(f'./gs_checkpoints/{prompt_path}/{prompt}.ply')
