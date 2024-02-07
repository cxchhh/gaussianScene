# You can use this to convert a .ply file to a .splat file programmatically in python
# Alternatively you can drag and drop a .ply file into the viewer at https://antimatter15.com/splat

from plyfile import PlyData,PlyElement
import numpy as np
import argparse
from io import BytesIO

from tqdm import tqdm


def process_ply_to_splat(plydata):
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in tqdm(sorted_indices,desc = "writing gaussians to file"):
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    return buffer.getvalue()

def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)


# splat_data = process_ply_to_splat("example.ply")
# output_file = "out.splat"
# save_splat_file(splat_data, output_file)
# print(f"Saved {output_file}")

def save_splat(gaussian_model, save_path):
    xyz = gaussian_model._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussian_model._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussian_model._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussian_model._opacity.detach().cpu().numpy()
    scale = gaussian_model._scaling.detach().cpu().numpy()
    rotation = gaussian_model._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in gaussian_model.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([el])

    splat_data = process_ply_to_splat(ply_data)
    save_splat_file(splat_data, save_path)
    print(f"Saved {save_path}")