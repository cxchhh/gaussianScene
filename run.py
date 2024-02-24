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
from utils.convert import save_splat
from utils.misc import kernel, pose2cam
from tqdm import tqdm

from utils.trajectory import get_pcdGenPoses
opt = GSParams()
cam = CameraParams()
gaussian_model = GaussianModel(opt.sh_degree)
background = torch.tensor([0,0,0], dtype=torch.float32, device='cuda')

seed = 1307
generator=torch.Generator(device='cuda').manual_seed(seed)
import sys
PATH = '/home/vrlab/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots/115134f363124c53c7d878647567d04daf26e41e'
sys.path.append(PATH)
rgb_model = AutoPipelineForInpainting.from_pretrained(
        PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        variant="fp16").to("cuda")
rgb_model.set_progress_bar_config(disable=True)

d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')

prompt = "Autumn park, realistic, photography"
neg_prompt = "people, text"
h_in, w_in = cam.H, cam.W
prompt_path = prompt.replace(" ","_")
gs_dir = f'./gs_checkpoints/{prompt_path}'
os.system(f"mkdir -p {gs_dir}")
output_path = f"{gs_dir}/out.splat"

def save_d_img(d, save_path):
    d_max = torch.max(torch.from_numpy(d))
    d_img = Image.fromarray((d/d_max.numpy()*255).astype(np.uint8))
    d_img.save(save_path)



N = 10 # camera pose nums
render_poses = torch.zeros(N,3,4)
yz_reverse = torch.tensor([[1,0,0], [0,-1,0], [0,0,-1]],dtype=torch.float32)
frames = []

render_poses[0,:3,:3] = torch.tensor([[1,0,0], [0,1,0], [0,0,1]],dtype=torch.float32)



H,W,K = cam.H,cam.W,cam.K
x, y = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy') # pixels
cameras_extent = 1.0

# create init scene
image_u8 = torch.zeros((h_in,w_in,3), dtype=torch.uint8)
image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
mask_u8 = torch.ones((h_in,w_in,3), dtype=torch.uint8) * 255
mask_pil = Image.fromarray(mask_u8.numpy()).convert('RGB')

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

depth_np = d_model.infer_pil(inpainted_image)
depth = torch.from_numpy(depth_np)
center_depth = torch.mean(depth[int(h_in/2)-2:int(h_in/2)+2,int(w_in/2)-2:int(w_in/2)+2])

R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
depth_pixels = torch.stack((x*depth, y*depth, 1*depth), axis=0)
pts_coord_cam_i = torch.matmul(torch.linalg.inv(K), depth_pixels.reshape(3,-1))
pts_coord_world_curr = (torch.linalg.inv(R0).matmul(pts_coord_cam_i) - torch.linalg.inv(R0).matmul(T0).reshape(3,1)).float()
pts_colors_curr = ((down_img).reshape(-1,3).float()/255.)

crop_mask = mask_u8
new_pts_coord_world = pts_coord_world_curr.T[crop_mask.reshape(-1,3) > 0].reshape(-1,3)
new_pts_colors = pts_colors_curr[crop_mask.reshape(-1,3) > 0].reshape(-1,3)

global_pts = new_pts_coord_world.cpu()
global_colors = new_pts_colors.cpu()
global_pcd = BasicPointCloud(global_pts, global_colors,normals=None)

gaussian_model= GaussianModel(opt.sh_degree)
gaussian_model.create_from_pcd(global_pcd, cameras_extent)



# render surroundings
train_cameras = []
gt_images = [None] * N

for i in tqdm(range(N),desc="creating point cloud from poses"):
    # if i < N/2:
    th = 360 * i / (N)
    th_rad = th / 180 * np.pi
    render_poses[i,:3,:3] = torch.tensor([[np.cos(th_rad),0,-np.sin(th_rad)],[0,1,0],[np.sin(th_rad),0,np.cos(th_rad)]],dtype=torch.float32)
    # t_rad = torch.tensor([0,0,-center_depth],dtype=torch.float32).reshape(3,1)
    # render_poses[i,:3,3:4] = torch.matmul(render_poses[i,:3,:3], t_rad ) - t_rad /2
    render_poses[i,:3,3:4] = torch.tensor([0,0,0],dtype=torch.float32).reshape(3,1)
    # else:
    #     th = 360 * (i % (N/2)) / (N/2)
    #     th_rad = th / 180 * np.pi
    #     phi = 0
    #     phi_rad = phi / 180 * np.pi
    #     Rot_H = torch.tensor([[np.cos(th_rad),0,-np.sin(th_rad)],[0,1,0],[np.sin(th_rad),0,np.cos(th_rad)]],dtype=torch.float32)
    #     Rot_V = torch.tensor([[1,0,0],[0,np.cos(phi_rad),np.sin(phi_rad)],[0,-np.sin(phi_rad),np.cos(phi_rad)]],dtype=torch.float32)
    #     render_poses[i,:3,:3] = torch.matmul(Rot_V, Rot_H)
    #     t_rad = torch.tensor([0,-0.2,0],dtype=torch.float32).reshape(3,1)
    #     render_poses[i,:3,3:4] = t_rad
    #
    train_cameras.append(pose2cam(render_poses[i],i)) 
    render_pkg = render(train_cameras[i], gaussian_model, opt, background)
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
    image_u8 = (255*image.clone().detach().cpu().permute(1,2,0)).byte()
    image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
    del gaussian_model # for saving VRAM
    
    Ri, Ti = render_poses[i,:3,:3], render_poses[i,:3,3:4]
    global_pts_cam_i = torch.matmul(K,(torch.matmul(Ri,global_pts.T)+Ti))
    global_pts_cam_i = global_pts_cam_i.T[global_pts_cam_i[2] > 0].T
    pts_pixels = torch.round(global_pts_cam_i[:2] / global_pts_cam_i[2]).int()
    mask_lo = pts_pixels >= torch.tensor([0,0]).reshape(2,1)
    mask_hi = pts_pixels < torch.tensor([w_in,h_in]).reshape(2,1)
    mask_inside = torch.logical_and(mask_lo[0] & mask_lo[1], mask_hi[0] & mask_hi[1])
    inside_pixels = pts_pixels.T[mask_inside]
    ref_d = global_pts_cam_i[2][mask_inside]
    ref_d_img = torch.zeros([512,512])
    ref_d_img[inside_pixels[:,1],inside_pixels[:,0]] = ref_d
    
    mask = torch.ones([512,512],dtype=torch.uint8)
    mask[inside_pixels.T.flip(0).tolist()] = 0
    mask *= 255
    mask = mask.repeat(3,1,1).permute(1,2,0).numpy()
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = torch.tensor(mask)

    mask_diff = torch.bitwise_xor(mask[:,:-1], mask[:,1:])
    mask_diff = cv2.dilate(mask_diff.numpy(), kernel, iterations=3)
    mask_diff = torch.tensor(mask_diff)
    mask_diff = mask_diff & (255 * (mask[:,:-1] == 0).byte())
    mask_diff = torch.cat([mask_diff, torch.zeros(h_in,1,3)],dim=1)
    mask_diff_pil = Image.fromarray(mask_diff.byte().numpy()).convert('RGB')
    # mask_diff_pil.save(f"./imgs/mask_diff_{i}.png")
    mask_diff = mask_diff.bool()


    mask_pil = Image.fromarray(mask.numpy()).convert('RGB')
    
    # image_pil.save(f"./imgs/img_{i}.png")
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

    # aligning the predicted depth
    sc = torch.nn.Parameter(torch.ones([h_in,2]).float())
    bi = torch.nn.Parameter(torch.zeros([h_in,2]).float())
    # else:
    #     sc = torch.ones([w_in]).float().requires_grad_(True)
    #     bi = torch.zeros([w_in]).float().requires_grad_(True)
    d_optimizer = torch.optim.Adam(params=[sc, bi], lr=0.005)
    mask_d = mask_diff[...,0]
    
    t = torch.linspace(0, 1, w_in, device=depth.device)
    for iter in range(100):
        sc_t = sc[:,:1] * (1-t) + sc[:,1:] * t
        bi_t = bi[:,:1] * (1-t) + bi[:,1:] * t
        trans_d = sc_t * depth + bi_t
        curr_d = trans_d[mask_d]
        ref_d = ref_d_img[mask_d]
        loss = torch.mean((ref_d - curr_d) ** 2)
        d_optimizer.zero_grad()
        loss.backward()
        d_optimizer.step()
    #import pdb; pdb.set_trace()
    # if i >= N/2:
    #     import pdb; pdb.set_trace()
    
    # add new points into global point cloud
    
    with torch.no_grad():
        sc_t = sc[:,:1] * (1-t) + sc[:,1:] * t
        bi_t = bi[:,:1] * (1-t) + bi[:,1:] * t
        depth = sc_t * depth + bi_t
        depth_pixels = torch.stack((x*depth, y*depth, 1*depth), axis=0)
        pts_coord_cam_i = torch.matmul(torch.linalg.inv(K), depth_pixels.reshape(3,-1))
        pts_coord_world_curr = (torch.linalg.inv(Ri).matmul(pts_coord_cam_i) - torch.linalg.inv(Ri).matmul(Ti).reshape(3,1)).float()
    
    pts_colors_curr = ((down_img).reshape(-1,3).float()/255.)

    new_pts_coord_world = pts_coord_world_curr.T[mask.reshape(-1,3) > 0].reshape(-1,3)
    new_pts_colors = pts_colors_curr[mask.reshape(-1,3) > 0].reshape(-1,3)

    global_pts = torch.concat([global_pts, new_pts_coord_world.cpu()], axis=0)
    global_colors = torch.concat([global_colors, new_pts_colors.cpu()], axis=0)
    global_pcd = BasicPointCloud(global_pts, global_colors,normals=None)
    
    gaussian_model= GaussianModel(opt.sh_degree)
    gaussian_model.create_from_pcd(global_pcd, cameras_extent)

mask_all = np.ones((h_in, w_in, 3), dtype=np.uint8)
mask_all_pil = Image.fromarray(np.round((mask_all)*255.).astype(np.uint8))
import pdb; pdb.set_trace()

# train gaussians
gaussian_model.training_setup(opt)


MAX_USAGE = 50
usage = [0] * N#



for iteration in tqdm(range(1, opt.iterations+1), desc="training gaussians"):#
    gaussian_model.update_learning_rate(iteration)

    if iteration % 1000 == 0:#
        gaussian_model.oneupSHdegree()
    
    
    # Pick a Camera
    randidx = randint(0, N - 1)
    viewpoint_cam = train_cameras[randidx]
    
    if usage[viewpoint_cam.uid] < MAX_USAGE:
        # Render
        render_pkg = render(viewpoint_cam, gaussian_model, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
    else:
        new_pose = torch.zeros([3,4],dtype=torch.float32)
        th = int(torch.rand(1)*360)
        th_rad = th / 180 * np.pi
        phi = 0
        phi_rad = phi / 180 * np.pi
        Rot_H = torch.tensor([[np.cos(th_rad),0,-np.sin(th_rad)],[0,1,0],[np.sin(th_rad),0,np.cos(th_rad)]],dtype=torch.float32)
        Rot_V = torch.tensor([[1,0,0],[0,np.cos(phi_rad),-np.sin(phi_rad)],[0,np.sin(phi_rad),np.cos(phi_rad)]],dtype=torch.float32)
        new_pose[:3,:3] = torch.matmul(Rot_V, Rot_H)
        ran = center_depth/2
        trans = torch.tensor([0,0,-ran * int(iteration*10 / opt.iterations)/10],dtype=torch.float32).reshape(3,1)
        new_pose[:3,3:4] = trans

        train_cameras[randidx] = pose2cam(new_pose, viewpoint_cam.uid)
        usage[viewpoint_cam.uid] = 0
        viewpoint_cam = train_cameras[randidx]

        # Render
        render_pkg = render(viewpoint_cam, gaussian_model, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

        # get refined G.T.
        image_u8 = (255*image.clone().detach().cpu().permute(1,2,0)).byte()
        image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
        strength_ctrlr = 0.8 - 0.1 * iteration / opt.iterations
        new_gt_pil =  rgb_model(
                prompt=prompt,
                negative_prompt=neg_prompt,
                generator=generator,
                strength=strength_ctrlr,
                guidance_scale=10,
                num_inference_steps=30,
                image=image_pil,
                mask_image=mask_all_pil
            ).images[0]

        # manually downsample from 1024 to 512, because the params (height, width)=512 to the pipeline give bad results
        raw_img = torch.from_numpy(np.array(new_gt_pil)).float()
        down_img = F.interpolate(raw_img.permute(2,0,1).unsqueeze(0), scale_factor=0.5).squeeze(0)
        new_gt = down_img/255.
        gt_images[viewpoint_cam.uid] = new_gt


    gt_image = gt_images[viewpoint_cam.uid].to("cuda")
    usage[viewpoint_cam.uid] += 1

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

        #save snapshot
        image_u8 = (255*image.clone().detach().cpu().permute(1,2,0)).byte()
        image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
        imgpath=f'{gs_dir}/snapshot_{(iteration)}.png'
        # print(imgpath)
        image_pil.save(imgpath)


#convert ply to gaussian splats

save_splat(gaussian_model, output_path)


