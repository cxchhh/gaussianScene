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
from dav2.depth_anything_v2.dpt import DepthAnythingV2

opt = GSParams()
cam = CameraParams()

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
multi_gpu = False

def load_rgb_model():
    PATH = os.environ['HOME']+'/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots/115134f363124c53c7d878647567d04daf26e41e'
    sys.path.append(PATH)
    rgb_model = AutoPipelineForInpainting.from_pretrained(
        PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda:1" if multi_gpu else "cuda")
    rgb_model.set_progress_bar_config(disable=True)
    return rgb_model

def load_depth_estimator():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

    d_model = DepthAnythingV2(**model_configs[encoder])
    D_PATH = os.environ['HOME']+'/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small/snapshots/14cf9f3d82acd6b6c9b43fa50b79a639a4e69c8d'
    d_model.load_state_dict(torch.load(f'{D_PATH}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    d_model = d_model.to(DEVICE).eval()
    return d_model

seed = 1307
generator = torch.Generator(device='cuda').manual_seed(seed)
rgb_model = load_rgb_model()


prompt = "Autumn park, high quality, 8k image, photorealistic"
neg_prompt = "text, photo frames"
prompt_path = prompt.replace(" ", "_")[:min(50, len(prompt))]

gs_dir = f'./gs_checkpoints/{prompt_path}'
os.system(f"mkdir -p {gs_dir}")
output_path = f"{gs_dir}/{prompt_path}.splat"


def save_d_img(d, save_path):
    d_max = torch.max(torch.from_numpy(d))
    d_img = Image.fromarray((d/d_max.numpy()*255).astype(np.uint8))
    d_img.save(save_path)


N = 10  # camera pose nums
assert N > 2
N_2 = 0
render_poses = torch.zeros(N + N_2, 3, 4)

H, W, K = cam.H, cam.W, cam.K
mask_all = torch.ones([H, W, 3], dtype=torch.uint8)*255
mask_all_pil = Image.fromarray(mask_all.numpy()).convert('RGB')

x, y = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(
    H, dtype=torch.float32), indexing='xy')  # pixels
cameras_extent = 1.0
bg_r = 0.75

h_t, v_t =torch.meshgrid(torch.linspace(-1,1,W),torch.linspace(-1,1,H),indexing='xy')

# render surroundings
train_cameras = []
gt_images = []
center_depth = None
near_depth = None
global_pts = None

pano_ratio = 2 * np.pi * cam.focal[1] / H
stride_rate = (pano_ratio - 1) / (N - 2)
stride = int(H * stride_rate)
margin = 0
# import pdb; pdb.set_trace()
assert stride < W - 10
pano_img = None
for i in tqdm(range(N + N_2), desc="creating panorama"):

    if i == 0:  # first pose
        image = torch.zeros([H, W, 3], dtype=torch.uint8)
        image_pil = Image.fromarray(image.numpy()).convert('RGB')
        mask = mask_all
        mask_pil = mask_all_pil
    else:  # other poses
        
        image = torch.ones([H, W, 3], dtype=torch.uint8)*255
       
        image[:, :(W - stride)] = pano_img[:, stride * i:]
        mask = mask_all
        mask[:, :(W - stride - margin) ] = 0

        if i == N - 1:
            end_width = max(W - stride, int(stride / 2))
            image[:, (W - end_width):] = pano_img[:, :end_width]
            mask[:, (W - end_width):] = 0

        image_pil = Image.fromarray(image.numpy()).convert('RGB')
        mask_pil = Image.fromarray(mask.numpy()).convert('RGB')

    # image_pil.save(f"./imgs/img_{i}.png")
    inpainted_image = rgb_model(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=6.0,
        strength=1.0,
        generator=generator,
        num_inference_steps=30
    ).images[0]

    # manually downsample from 1024 to 512, because the params (height,width)=(512,512) to the pipeline give bad result
    raw_img = torch.from_numpy(np.array(inpainted_image))
    down_img = F.interpolate(raw_img.permute(2, 0, 1).unsqueeze(
        0), scale_factor=H/1024).squeeze(0).permute(1, 2, 0)
    inpainted_image = Image.fromarray(down_img.numpy())

    inp_image = torch.from_numpy(np.array(inpainted_image))

    if i == 0:
        pano_img = inp_image
    else:
        pano_img = torch.concat([pano_img[:, :pano_img.shape[1] - margin],
                                  inp_image[:, (W - stride - margin):]], dim=1)
    

pano_pil = Image.fromarray(pano_img.numpy())
del rgb_model
torch.cuda.empty_cache()

# predict depth
d_model = load_depth_estimator()
depth = d_model.infer_image(pano_img.numpy())
del d_model
torch.cuda.empty_cache()

depth = torch.tensor((depth.max() - depth) + depth.min())
depth = depth + 2 * torch.exp(0.01 * depth ** 2)

pano_pil.save("pano.png")
save_d_img(depth.numpy(), "d.png")

P_s = pano_img.shape[1]

sc = torch.nn.Parameter(torch.ones([2]).float())
bi = torch.nn.Parameter(torch.zeros([1]).float())
d_optimizer = torch.optim.Adam(params=[sc, bi], lr=0.01)
t = torch.linspace(0, 1, pano_img.shape[1])

for iter in range(1000):
    bi_map = bi * t
    trans_d = (depth + bi_map)
    d_0 = trans_d[:, :end_width]
    d_1 = trans_d[:, -end_width:]
    loss = torch.mean((d_1 - d_0) ** 2)
    d_optimizer.zero_grad()
    loss.backward()
    d_optimizer.step()

with torch.no_grad():
    depth = (depth + bi_map)

# extract to 3d
pano = pano_img[:, :-end_width] # (H, P+s) -> (H, P)
pts_colors = ((pano).reshape(-1, 3).float()/255.) # (H*P, 3)

depth = depth[:, :-end_width]
center_depth = torch.mean(depth[int(H/2)-10:int(H/2)+10,:])

thetas = torch.linspace(0, 2 * np.pi, pano.shape[1]).unsqueeze(0).repeat(H, 1)
phis = torch.linspace((H / (2 * cam.focal[1])),
                       -(H / (2 * cam.focal[1])), H).unsqueeze(1).repeat(1, pano.shape[1])
pts_world = torch.zeros_like(pts_colors) # (H*P, 3)
pts_world[:, 0] = -torch.cos(thetas).reshape(-1) * depth.reshape(-1)
pts_world[:, 2] = torch.sin(thetas).reshape(-1) * depth.reshape(-1)
pts_world[:, 1] = -phis.reshape(-1) * depth.reshape(-1)
pts_world = pts_world 
pcd = BasicPointCloud(pts_world.cpu(), pts_colors.cpu(),normals=None)

gaussian_model = GaussianModel(opt.sh_degree)
gaussian_model.create_from_pcd(pcd, cameras_extent)
background = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
debug = 0
if debug:
    save_splat(gaussian_model, output_path)
    import pdb; pdb.set_trace()


# train gaussians
gaussian_model.training_setup(opt)

MAX_USAGE = 500
usage = []
ADD_POSE_INTERVAL = 200
ADD_POSE_FROM_ITER = 60000
ADD_POSE_UNTIL_ITER = 3000

rgb_model = load_rgb_model()

def get_new_GT(viewpoint_cam, rate: float):
    # Render
    render_pkg = render(viewpoint_cam, gaussian_model, opt, background)
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

    # get refined G.T.
    image_u8 = (255*image.clone().detach().cpu().permute(1, 2, 0)).byte()
    image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')
    strength_ctrlr = max(0.3, 0.8 - 0.3 * rate)
    new_gt_pil = rgb_model(
        prompt=prompt,
        negative_prompt=neg_prompt,
        generator=generator,
        strength=strength_ctrlr,
        guidance_scale=12.0,
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

pano2 = torch.concat([pano, pano], dim=1)
for i in tqdm(range(N), desc="warm-up gaussians from poses"):
    
    th = 360 * i / N
    th_rad = th / 180 * np.pi
    phi = 0
    phi_rad = phi / 180 * np.pi
    Rot_H = torch.tensor([[np.cos(th_rad), 0, -np.sin(th_rad)], [0, 1, 0],
                            [np.sin(th_rad), 0, np.cos(th_rad)]], dtype=torch.float32)
    Rot_V = torch.tensor([[1, 0, 0], [0, np.cos(phi_rad), np.sin(phi_rad)], [
                            0, -np.sin(phi_rad), np.cos(phi_rad)]], dtype=torch.float32)
    render_poses[i, :3, :3] = torch.matmul(Rot_V, Rot_H)

    render_poses[i, :3, 3:4] = torch.tensor([0, 0, 0], dtype=torch.float32).reshape(3, 1)

    train_cameras.append(pose2cam(render_poses[i], i))
    usage.append(0)
    viewpoint_cam = train_cameras[i]

    render_pkg = render(train_cameras[i], gaussian_model, opt, background)
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
    image_u8 = (255*image.clone().detach().cpu().permute(1, 2, 0)).byte()
    image_pil = Image.fromarray(image_u8.numpy()).convert('RGB')

    gt_u8 = pano2[:,703+int(i*(P_s/N-15)):703+int(+i*(P_s/N-15))+W]
    gt_images.append(gt_u8.permute(2,0,1).float()/255.)

    #import pdb; pdb.set_trace()
        
    # warm-up
    gt_image = gt_images[i].to("cuda")
    training_model = gaussian_model 
    training_model.training_setup(opt)
    training_iters = 100
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


for iteration in tqdm(range(1, opt.iterations+1), desc="training gaussians"):
    gaussian_model.update_learning_rate(iteration)

    if iteration % 1000 == 0:
        gaussian_model.oneupSHdegree()

    if iteration <= ADD_POSE_UNTIL_ITER and iteration > ADD_POSE_FROM_ITER and iteration % ADD_POSE_INTERVAL == 0 or len(usage) == 0:
        new_pose = add_new_pose(center_depth, iteration, opt.iterations)
        new_cam = pose2cam(new_pose, len(train_cameras))
        train_cameras.append(new_cam)
        usage.append(0)
        gt_images.append(get_new_GT(new_cam, iteration / ADD_POSE_UNTIL_ITER))

    # Pick a Camera
    cam_idx = randint(0, len(train_cameras) - 1)
    viewpoint_cam = train_cameras[cam_idx]

    if usage[viewpoint_cam.uid] < MAX_USAGE or iteration > ADD_POSE_UNTIL_ITER:
        # Render
        render_pkg = render(viewpoint_cam, gaussian_model, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
    else:
        # Replace camera and Render
        # import pdb;pdb.set_trace()
        new_pose = add_new_pose(center_depth, iteration, opt.iterations)

        train_cameras[cam_idx] = pose2cam(new_pose, viewpoint_cam.uid)
        usage[viewpoint_cam.uid] = 0
        viewpoint_cam = train_cameras[cam_idx]

        render_pkg = render(viewpoint_cam, gaussian_model, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

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
