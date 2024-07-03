import sys
import os
from random import randint, uniform
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from arguments import CameraParams, GSParams
from scene.gaussian_model import GaussianModel
from diffusers import AutoPipelineForInpainting, DiffusionPipeline
from gaussian_renderer import render
from utils.graphics import BasicPointCloud
from utils.loss import l1_loss, ssim
from scene.dataset_readers import readDataInfo
from utils.convert import save_splat
from utils.misc import add_new_pose, create_bottom, kernel, pose2cam, x_kernel, v_kernel
from tqdm import tqdm

opt = GSParams()
cam = CameraParams()
gaussian_model = GaussianModel(opt.sh_degree)
background = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

multi_gpu = False

seed = 1307
generator = torch.Generator(device='cuda').manual_seed(seed)
PATH = os.environ['HOME']+'/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots/115134f363124c53c7d878647567d04daf26e41e'
sys.path.append(PATH)
rgb_model = AutoPipelineForInpainting.from_pretrained(
    PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda:1" if multi_gpu else "cuda")
rgb_model.set_progress_bar_config(disable=True)

D_PATH = os.environ['HOME']+'/.cache/huggingface/hub/models--prs-eth--marigold-lcm-v1-0/snapshots/773825ffad4318356efcd14e3ff89d7812e5a0ab'
d_model = DiffusionPipeline.from_pretrained(
    D_PATH,
    local_files_only=True,
    custom_pipeline="./pipelines/marigold_depth_estimation.py",
    torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
    variant="fp16",                           # (optional) Use with `torch_dtype=torch.float16`, to directly load fp16 checkpoint
).to("cuda:2" if multi_gpu else "cuda")
d_model.set_progress_bar_config(disable=True)

prompt = "A living room, high quality, 8k image, photorealistic"
neg_prompt = "text, photo frames"
h_in, w_in = cam.H, cam.W
prompt_path = prompt.replace(" ", "_")[:min(50, len(prompt))]
gs_dir = f'./gs_checkpoints/{prompt_path}'
os.system(f"mkdir -p {gs_dir}")
output_path = f"{gs_dir}/{prompt_path}.splat"


def save_d_img(d, save_path):
    d_max = torch.max(torch.from_numpy(d))
    d_img = Image.fromarray((d/d_max.numpy()*255).astype(np.uint8))
    d_img.save(save_path)


mask_all = torch.ones([h_in, w_in, 3], dtype=torch.uint8)*255
mask_all_pil = Image.fromarray(mask_all.numpy()).convert('RGB')

N = 10  # camera pose nums
N_2 = 0
render_poses = torch.zeros(N + N_2, 3, 4)


H, W, K = cam.H, cam.W, cam.K
x, y = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(
    H, dtype=torch.float32), indexing='xy')  # pixels
cameras_extent = 1.0
bg_r = 0.75

h_t, v_t =torch.meshgrid(torch.linspace(-1,1,W),torch.linspace(-1,1,H),indexing='xy')

# render surroundings
train_cameras = []
gt_images = [None] * (N + N_2)
center_depth = None
near_depth = None
global_pts = None
for i in tqdm(range(N + N_2), desc="creating point cloud from poses"):
    if i == 0:
        bg_r = 1
        render_poses[i, :3, :3] = torch.tensor([[1, 0, 0], 
                                                [0, 1, 0], 
                                                [0, 0, 1]], dtype=torch.float32)

        render_poses[i, :3, 3:4] = torch.tensor(
            [0, 0, 1], dtype=torch.float32).reshape(3, 1)
    elif i < N:
        alpha = 45
        t_rad = 2* np.pi * (i - 1) / (N - 1)
        th = alpha * np.sin(t_rad)
        th_rad = th / 180 * np.pi
        phi = -alpha * np.cos(t_rad) *0.3
        phi_rad = phi / 180 * np.pi
        Rot_H = torch.tensor([[np.cos(th_rad), 0, -np.sin(th_rad)], [0, 1, 0],
                             [np.sin(th_rad), 0, np.cos(th_rad)]], dtype=torch.float32)
        Rot_V = torch.tensor([[1, 0, 0], [0, np.cos(phi_rad), np.sin(phi_rad)], [
                             0, -np.sin(phi_rad), np.cos(phi_rad)]], dtype=torch.float32)
        render_poses[i, :3, :3] = torch.matmul(Rot_V, Rot_H)

        render_poses[i, :3, 3:4] = torch.tensor(
            [2* np.sin(t_rad), np.cos(t_rad), 2], dtype=torch.float32).reshape(3, 1)

    train_cameras.append(pose2cam(render_poses[i], i))
    Ri, Ti = render_poses[i, :3, :3], render_poses[i, :3, 3:4]

    if i == 0:  # first pose
        image = torch.zeros([H, W, 3], dtype=torch.uint8)
        image_pil = Image.fromarray(image.numpy()).convert('RGB')
        mask = mask_all
        mask_pil = mask_all_pil
    else:  # other poses
        render_pkg = render(train_cameras[i], gaussian_model, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
        image_u8 = (255*image.clone().detach().cpu().permute(1, 2, 0)).byte()
        image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
        # del gaussian_model # for saving VRAM

        global_pts = gaussian_model.get_xyz.detach().clone().cpu()
        global_pts_cam_i = torch.matmul(K, (torch.matmul(Ri, global_pts.T)+Ti))
        pos_z_mask = global_pts_cam_i[2] > 0
        global_pts_cam_i = global_pts_cam_i.T[pos_z_mask].T
        pts_pixels = torch.round(
            global_pts_cam_i[:2] / global_pts_cam_i[2]).int()
        mask_lo = pts_pixels >= torch.tensor([0, 0]).reshape(2, 1)
        mask_hi = pts_pixels < torch.tensor([w_in, h_in]).reshape(2, 1)
        mask_inside = torch.logical_and(
            mask_lo[0] & mask_lo[1], mask_hi[0] & mask_hi[1])
        inside_pixels = pts_pixels.T[mask_inside]
        ref_d = global_pts_cam_i[2][mask_inside]
        ref_d_img = torch.zeros([512, 512])
        ref_d_img[inside_pixels[:, 1], inside_pixels[:, 0]] = ref_d
        ref_d_img = torch.tensor(cv2.dilate(ref_d_img.numpy(),np.ones([3,3],np.uint8))) # dilate to eliminate stain spots

        mask = torch.ones([512, 512], dtype=torch.uint8)
        mask[inside_pixels.T.flip(0).tolist()] = 0
        mask *= 255
        mask = mask.repeat(3, 1, 1).permute(1, 2, 0).numpy()
        mask = cv2.erode(mask, kernel, iterations=8)
        mask_erode = torch.tensor(mask)
        mask = cv2.dilate(mask, kernel, iterations=9)
        mask = torch.tensor(mask)

        mask_diff = torch.bitwise_xor(mask[:, :-1], mask[:, 1:])
        mask_diff = cv2.dilate(mask_diff.numpy(), kernel, iterations=6)
        mask_diff = torch.tensor(mask_diff)
        mask_diff = mask_diff & (255 * (mask[:, :-1] == 0).byte())
        mask_diff = torch.cat([mask_diff, torch.zeros(h_in, 1, 3)], dim=1)
        mask_diff_pil = Image.fromarray(
            mask_diff.byte().numpy()).convert('RGB')

        mask_diff = mask_diff.bool()

        mask_pil = Image.fromarray(mask.numpy()).convert('RGB')

    # image_pil.save(f"./imgs/img_{i}.png")
    inpainted_image = rgb_model(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=10.0,
        strength=1.0,
        generator=generator,
        num_inference_steps=30
    ).images[0]

    # manually downsample from 1024 to 512, because the params (height,width)=(512,512) to the pipeline give bad result
    raw_img = torch.from_numpy(np.array(inpainted_image))
    down_img = F.interpolate(raw_img.permute(2, 0, 1).unsqueeze(
        0), scale_factor=H/1024).squeeze(0).permute(1, 2, 0)
    inpainted_image = Image.fromarray(down_img.numpy())

    gt_images[i] = (torch.from_numpy(
        np.array(inpainted_image))/255.).permute(2, 0, 1).float()

    depth_pkg = d_model(inpainted_image)
    depth_np = depth_pkg['depth_np'] * 10
    depth = torch.from_numpy(depth_np)
    
    # if i > 0:
    #     import pdb;pdb.set_trace()

    if i == 0:
        depth = depth + depth.median()
        # calc radius of the scene
        center_depth = torch.mean(
            depth[int(h_in/2)-2:int(h_in/2)+2, int(w_in/2)-2:int(w_in/2)+2])
    else:
        depth = depth + depth.median()
    if i > 0:
        # aligning the predicted depth
        
        sc = torch.nn.Parameter(torch.ones([1]).float())
        bi = torch.nn.Parameter(torch.zeros([1]).float())
        d_optimizer = torch.optim.Adam(params=[sc, bi], lr=0.01)
        mask_d_edge = mask_diff[..., 0]
        mask_d = mask[..., 0] <= 0
        
        
        #import pdb; pdb.set_trace()
        for iter in range(1000):
            sc_map = sc
            trans_d = (sc_map * depth) + bi
            curr_d = trans_d[mask_d_edge]
            gt_d = ref_d_img[mask_d_edge]
            loss = torch.mean((gt_d - curr_d) ** 2)
            d_optimizer.zero_grad()
            loss.backward()
            d_optimizer.step()

        # add new points into global point cloud
        with torch.no_grad():
            #import pdb; pdb.set_trace()
            depth = (sc_map * depth) + bi
        # t_map = torch.zeros_like(depth,dtype=torch.float32)
        # mask_d_erode = mask_erode[..., 0] > 0 if i < N else mask_erode[..., 0] <= 0
        # md = (mask_d_erode).float()
        # md_arr = [(mask_d_erode)]
        # while True:
        #     new_md = torch.tensor(cv2.dilate(md.numpy(),
        #                             x_kernel if i < N else v_kernel,
        #                             iterations=5))
        #     new_md_mask = (new_md - md) > 0
        #     if not new_md_mask.max():
        #         break
        #     md_arr.append(new_md_mask.detach().clone())
        #     md = new_md
        # t = torch.linspace(1, 0, len(md_arr))
        # for j, md_mask in enumerate(md_arr):
        #     t_map[md_mask] = t[j]
        # if i<N:
        #     depth[mask_d] = (ref_d_img * (1 - t_map) + depth * t_map)[mask_d]
            

    depth_pixels = torch.stack((x*depth, y*depth, 1*depth), axis=0)
    pts_coord_cam_i = torch.matmul(
        torch.linalg.inv(K), depth_pixels.reshape(3, -1))
    pts_coord_world_i = (torch.linalg.inv(Ri).matmul(
        pts_coord_cam_i) - torch.linalg.inv(Ri).matmul(Ti).reshape(3, 1)).float()

    pts_colors_i = ((down_img).reshape(-1, 3).float()/255.)

    combined_mask = (mask.reshape(-1, 3) > 0) 
    # import pdb; pdb.set_trace()
    
    # if i > 0 and i < N:
    #     global_mask_inside = torch.zeros([global_pts.shape[0]]).bool()
    #     global_mask_inside[pos_z_mask] = mask_inside
    #     pts_coord_interpld = pts_coord_world_i.reshape(3,H,W)[..., inside_pixels[:, 1], inside_pixels[:, 0]]
    #     #import pdb; pdb.set_trace()
    #     global_pts.T[..., global_mask_inside] = pts_coord_interpld
        
    #     gaussian_model.change_global_pts(global_pts)
        
    # elif i >= N:
    #     global_mask_inside = torch.zeros([global_pts.shape[0]]).bool()
    #     global_mask_inside[pos_z_mask] = mask_inside
        
    #     edge_mask = mask_diff[..., 0][inside_pixels[:, 1],inside_pixels[:, 0]]
    #     edge_pixels=inside_pixels[edge_mask]
        
    #     global_edge_mask = torch.zeros_like(global_mask_inside)
    #     global_edge_mask[global_mask_inside] = edge_mask
    #     ref_global_coord = global_pts.T[..., global_edge_mask]
    #     ref_edge_coord = torch.zeros([3, H,W])
    #     ref_edge_coord[..., edge_pixels[:, 1], edge_pixels[:,0]]= ref_global_coord
    #     final_v_mask = torch.zeros(H,W).bool()
    #     final_v_mask[edge_pixels[:, 1], edge_pixels[:,0]] = 1
    #     final_v_mask = final_v_mask & mask_diff[..., 0]
        
    #     ref_v_coord = ref_edge_coord[..., final_v_mask] # 3 * M
    #     v_coord = pts_coord_world_i.reshape(3,H,W)[..., final_v_mask] # 3 * M
    #     u_coord = pts_coord_world_i.T[combined_mask[...,0]].T # 3 * N
    #     delta_v = ref_v_coord - v_coord
        
    #     delta_u = delta_v.mean(dim=1)
    #     # import pdb; pdb.set_trace()
    #     pts_coord_world_i.T[combined_mask[...,0]] += delta_u

    new_pts_coord_world = pts_coord_world_i.T[combined_mask].reshape(-1, 3)
    new_pts_colors = pts_colors_i[combined_mask].reshape(-1, 3)

    if i == 0:
        global_pts = new_pts_coord_world.cpu()
        near_depth = torch.sort(global_pts[:, 2]).values[:int(
            global_pts.shape[0]/100)].mean()
    else:
        global_pts = torch.concat(
            [global_pts, new_pts_coord_world.cpu()], axis=0)

    new_pcd = BasicPointCloud(
        new_pts_coord_world.cpu(), new_pts_colors.cpu(), normals=None)

    new_gaussian_model = GaussianModel(opt.sh_degree)
    new_gaussian_model.create_from_pcd(new_pcd, cameras_extent)
    
    gaussian_model.merge(new_gaussian_model)
    del new_gaussian_model
    # if i >= N:  
    #     import pdb; pdb.set_trace()
    
    # warm-up
    gt_image = gt_images[i].to("cuda")
    training_model = gaussian_model 
    training_model.training_setup(opt)
    training_iters = 50
    for iteration in range(training_iters):
        render_pkg = render(train_cameras[i], training_model, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + \
            opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        with torch.no_grad():
            gaussian_model.max_radii2D[visibility_filter] = torch.max(
                    gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussian_model.add_densification_stats(
                viewspace_point_tensor, visibility_filter)

            if  iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussian_model.densify_and_prune(
                    opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)

            training_model.optimizer.step()
            training_model.optimizer.zero_grad(set_to_none=True)


debug = 0
if debug:
    save_splat(gaussian_model, output_path)
    import pdb; pdb.set_trace()


# train gaussians
gaussian_model.training_setup(opt)


MAX_USAGE = 50000
usage = [0] * (N + N_2)
ADD_POSE_INTERVAL = 10
ADD_POSE_FROM_ITER = 1000
ADD_POSE_UNTIL_ITER = 3000


def get_new_GT(viewpoint_cam, rate: float):
    # Render
    render_pkg = render(viewpoint_cam, gaussian_model, opt, background)
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

    # get refined G.T.
    image_u8 = (255*image.clone().detach().cpu().permute(1, 2, 0)).byte()
    image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
    strength_ctrlr = max(0.3, 0.6 - 0.3 * rate)
    new_gt_pil = rgb_model(
        prompt=prompt,
        negative_prompt=neg_prompt,
        generator=generator,
        strength=strength_ctrlr,
        guidance_scale=8,
        num_inference_steps=30,
        image=image_pil,
        mask_image=mask_all_pil
    ).images[0]

    # manually downsample from 1024 to 512, because the params (height, width)=512 to the pipeline give bad results
    raw_img = torch.from_numpy(np.array(new_gt_pil)).float()
    down_img = F.interpolate(raw_img.permute(
        2, 0, 1).unsqueeze(0), scale_factor=H/1024).squeeze(0)
    new_gt = down_img/255.

    return new_gt


for iteration in tqdm(range(1, opt.iterations+1), desc="training gaussians"):
    gaussian_model.update_learning_rate(iteration)

    if iteration % 1000 == 0:
        gaussian_model.oneupSHdegree()

    # if iteration < ADD_POSE_UNTIL_ITER and iteration % ADD_POSE_INTERVAL == 0:
    #     new_pose = add_new_pose(center_depth, iteration, opt.iterations)
    #     new_cam = pose2cam(new_pose, len(train_cameras))
    #     train_cameras.append(new_cam)
    #     usage.append(0)
    #     gt_images.append(get_new_GT(new_cam, iteration / ADD_POSE_UNTIL_ITER))

    # Pick a Camera
    cam_idx = randint(0, len(train_cameras) - 1)
    viewpoint_cam = train_cameras[cam_idx]

    if usage[viewpoint_cam.uid] < MAX_USAGE or iteration < ADD_POSE_FROM_ITER:
        # Render
        render_pkg = render(viewpoint_cam, gaussian_model, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
    else:
        # Replace camera and Render
        import pdb;pdb.set_trace()
        new_pose = add_new_pose(center_depth, iteration, opt.iterations)

        train_cameras[cam_idx] = pose2cam(new_pose, viewpoint_cam.uid)
        usage[viewpoint_cam.uid] = 0
        viewpoint_cam = train_cameras[cam_idx]

        gt_images[viewpoint_cam.uid] = get_new_GT(viewpoint_cam, iteration / ADD_POSE_UNTIL_ITER)

    gt_image = gt_images[viewpoint_cam.uid].to("cuda")
    usage[viewpoint_cam.uid] += 1

    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + \
        opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss.backward(retain_graph=True)

    with torch.no_grad():
        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussian_model.max_radii2D[visibility_filter] = torch.max(
                gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussian_model.add_densification_stats(
                viewspace_point_tensor, visibility_filter)

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
            gaussian_model.optimizer.zero_grad(set_to_none=True)

    if (iteration) % 200 == 0:
        # save snapshot
        image_u8 = (255*image.clone().detach().cpu().permute(1, 2, 0)).byte()
        image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
        imgpath = f'{gs_dir}/snapshot_{(iteration)}.png'
        # print(imgpath)
        image_pil.save(imgpath)


# convert ply to gaussian splats

save_splat(gaussian_model, output_path)
