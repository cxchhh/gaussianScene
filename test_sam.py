import sys
import os
from random import randint, uniform
import torch
from torch import nn
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
from utils.misc import add_anns, add_new_pose, create_bottom, kernel, pose2cam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

opt = GSParams()
cam = CameraParams()
gaussian_model = GaussianModel(opt.sh_degree)
background = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

seed = 1307
generator = torch.Generator(device='cuda').manual_seed(seed)
PATH = os.environ['HOME']+'/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots/115134f363124c53c7d878647567d04daf26e41e'
sys.path.append(PATH)
rgb_model = AutoPipelineForInpainting.from_pretrained(
    PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
    variant="fp16").to('cuda')
rgb_model.set_progress_bar_config(disable=True)

rgb_model=nn.DataParallel(rgb_model).module


D_PATH = os.environ['HOME']+'/.cache/huggingface/hub/models--prs-eth--marigold-lcm-v1-0/snapshots/773825ffad4318356efcd14e3ff89d7812e5a0ab'
d_model = DiffusionPipeline.from_pretrained(
    D_PATH,
    local_files_only=True,
    custom_pipeline="marigold_depth_estimation",
    torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
    variant="fp16",                           # (optional) Use with `torch_dtype=torch.float16`, to directly load fp16 checkpoint
).to("cuda")
d_model.set_progress_bar_config(disable=True)

prompt = "A tranquil autumn forest with warm, vibrant leaves and a soft, diffused light, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by greg rutkowski and alphonse mucha"
neg_prompt = "people, text"
h_in, w_in = cam.H, cam.W
prompt_path = prompt.replace(" ", "_")[:min(50, len(prompt))]
gs_dir = f'./gs_checkpoints/{prompt_path}'
os.system(f"mkdir -p {gs_dir}")
output_path = f"{gs_dir}/out.splat"


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

# render surroundings
train_cameras = []
gt_images = [None] * (N + N_2)
center_depth = None
near_depth = None
global_pts = None
gauss_buff = []
for i in tqdm(range(N + N_2), desc="creating point cloud from poses"):
    if i < N:
        bg_r = 1
        th = 360 * i / N
        th_rad = th / 180 * np.pi
        render_poses[i, :3, :3] = torch.tensor([[np.cos(th_rad), 0, -np.sin(th_rad)], [
                                               0, 1, 0], [np.sin(th_rad), 0, np.cos(th_rad)]], dtype=torch.float32)

        render_poses[i, :3, 3:4] = torch.tensor(
            [0, 0, 0], dtype=torch.float32).reshape(3, 1)

    train_cameras.append(pose2cam(render_poses[i], i))
    Ri, Ti = render_poses[i, :3, :3], render_poses[i, :3, 3:4]

    if i == 0:  # first pose
        image = torch.zeros([512, 512, 3], dtype=torch.uint8)
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

        global_pts_cam_i = torch.matmul(K, (torch.matmul(Ri, global_pts.T)+Ti))
        global_pts_cam_i = global_pts_cam_i.T[global_pts_cam_i[2] > 0].T
        pts_pixels = torch.round(
            global_pts_cam_i[:2] / global_pts_cam_i[2]).int()
        mask_lo = pts_pixels >= torch.tensor([0, 0]).reshape(2, 1)
        mask_hi = pts_pixels < torch.tensor([w_in, h_in]).reshape(2, 1)
        mask_inside = torch.logical_and(
            mask_lo[0] & mask_lo[1], mask_hi[0] & mask_hi[1])
        inside_pixels = pts_pixels.T[mask_inside]
        ref_d = global_pts_cam_i[2][mask_inside]
        ref_d_img = torch.zeros([H, W])
        ref_d_img[inside_pixels[:, 1], inside_pixels[:, 0]] = ref_d

        mask = torch.ones([H, W], dtype=torch.uint8)
        mask[inside_pixels.T.flip(0).tolist()] = 0
        mask *= 255
        mask = mask.repeat(3, 1, 1).permute(1, 2, 0).numpy()
        mask = cv2.erode(mask, kernel, iterations=3)
        mask = cv2.dilate(mask, kernel, iterations=5)
        mask = torch.tensor(mask)

        mask_diff = torch.bitwise_xor(mask[:, :-1], mask[:, 1:])
        mask_diff = cv2.dilate(mask_diff.numpy(), kernel, iterations=3)
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
        guidance_scale=8.0,
        strength=1.0,
        generator=generator,
        num_inference_steps=15
    ).images[0]
    # manually downsample from 1024 to 512, because the params (height,width)=(512,512) to the pipeline give bad result
    raw_img = torch.from_numpy(np.array(inpainted_image))
    down_img = F.interpolate(raw_img.permute(2, 0, 1).unsqueeze(
        0), scale_factor=0.5).squeeze(0).permute(1, 2, 0)
    inpainted_image = Image.fromarray(down_img.numpy())
    inpainted_image_np = np.array(inpainted_image)
    

    depth_np = d_model(inpainted_image)['depth_np'] * 10
    depth = torch.from_numpy(depth_np)

    gt_images[i] = (torch.from_numpy(
        np.array(inpainted_image))/255.).permute(2, 0, 1).float()
    

    if i == 0:
        # calc radius of the scene
        center_depth = torch.mean(
            depth[int(h_in/2)-2:int(h_in/2)+2, int(w_in/2)-2:int(w_in/2)+2])
    else:
        # aligning the predicted depth
        if i< N:
            sc = torch.nn.Parameter(torch.ones([h_in, 2]).float())
            bi = torch.nn.Parameter(torch.zeros([h_in, 2]).float())
        p = torch.nn.Parameter(torch.Tensor([1.0]))
        d_optimizer = torch.optim.Adam(params=[sc, bi], lr=0.005)
        mask_d = mask_diff[..., 0]

        t = torch.linspace(0, 1, w_in, device=depth.device)
        for iter in range(100):
            if i < N:
                sc_t = 1 + (sc[:, :1] - 1) * (1-t)**p + (sc[:, 1:] - 1) * t**p
                bi_t = bi[:, :1] * (1-t)**p + bi[:, 1:] * t**p

            trans_d = sc_t * depth + bi_t
            curr_d = trans_d[mask_d]
            ref_d = ref_d_img[mask_d]
            loss1 = torch.mean((ref_d - curr_d) ** 2)
            if i < N:
                loss2 = torch.mean(((trans_d[1:, :] - trans_d[:-1, :])) ** 2)
                loss = loss1 + loss2 * 100

            d_optimizer.zero_grad()
            loss.backward()
            d_optimizer.step()
        
        with torch.no_grad():
            if i < N:
                sc_t = 1 + (sc[:, :1] - 1) * (1-t)**p + (sc[:, 1:] - 1) * t**p
                bi_t = bi[:, :1] * (1-t)**p + bi[:, 1:] * t**p
                depth = sc_t * depth + bi_t
    #import pdb; pdb.set_trace()

    # add new points into global point cloud
    depth_pixels = torch.stack((x*depth, y*depth, 1*depth), axis=0)
    pts_coord_cam_i = torch.matmul(
        torch.linalg.inv(K), depth_pixels.reshape(3, -1))
    pts_coord_world_curr = (torch.linalg.inv(Ri).matmul(
        pts_coord_cam_i) - torch.linalg.inv(Ri).matmul(Ti).reshape(3, 1)).float()

    pts_colors_curr = ((down_img).reshape(-1, 3).float()/255.)

    final_mask = mask.reshape(-1, 3) > 0

    new_pts_coord_world = pts_coord_world_curr.T[final_mask].reshape(-1, 3)
    new_pts_colors = pts_colors_curr[final_mask].reshape(-1, 3)

    if i == 0:
        global_pts = new_pts_coord_world.cpu()
        near_depth = torch.sort(global_pts[:, 2]).values[:int(
            global_pts.shape[0]/100)].mean()
    elif i < N:
        global_pts = torch.concat(
            [global_pts, new_pts_coord_world.cpu()], axis=0)

    new_pcd = BasicPointCloud(
        new_pts_coord_world.cpu(), new_pts_colors.cpu(), normals=None)

    new_gaussian_model = GaussianModel(opt.sh_degree)
    new_gaussian_model.create_from_pcd(new_pcd, cameras_extent)
    if i < N:
        gaussian_model.merge(new_gaussian_model)
        del new_gaussian_model
    # import pdb; pdb.set_trace()

    # gt_image = gt_images[i].to("cuda")
    # training_model = gaussian_model
    # training_model.training_setup(opt)
    # training_iters = 50
    # for iteration in range(training_iters):
    #     render_pkg = render(train_cameras[i], training_model, opt, background)
    #     image, viewspace_point_tensor, visibility_filter, radii = (
    #         render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

    #     Ll1 = l1_loss(image, gt_image)
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + \
    #         opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    #     loss.backward()

    #     training_model.optimizer.step()
    #     training_model.optimizer.zero_grad(set_to_none=True)

    break

debug = 1
if debug:
    save_splat(gaussian_model, output_path)