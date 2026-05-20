import argparse
import os
import pickle

import numpy as np
import pydiffvg
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from sds_image_simplicity import load_512
from utils.common import init_diffvg, init_optimizer, load_config
from utils.img_process import (
    add_visual_paths_shade,
    get_cluster_img,
    merge_path,
    remove_lowquality_paths,
    rgba_to_rgb,
    svg_to_img,
)


def svg_optimize_img_visual_shade(
    device,
    albedo_img,
    shapes,
    shape_groups,
    target_img,
    file_save_path,
    is_opt_list,
    train_conf,
    base_lr_conf,
    count=0,
    struct_path_num=0,
    is_path_merging_phase=False,
):
    img_height, img_width = target_img.shape[:2]
    target_img = torch.tensor(target_img, device=device) / 255
    target_img = target_img.permute(2, 0, 1)

    svg_optimizer = init_optimizer(
        shapes,
        shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_visual_color"],
        is_opt_list,
        lr_base=base_lr_conf,
    )
    num_iters = 100 if is_path_merging_phase else train_conf["visual_opt_num_iters"]

    with tqdm(total=num_iters, desc="Shade visual optimization", unit="iter") as pbar:
        for _ in range(num_iters):
            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device)
            img = img * albedo_img
            loss = F.mse_loss(img, target_img)
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()
            pydiffvg.save_svg(f"{file_save_path}/{count}.svg", img_width, img_height, shapes, shape_groups)
            count += 1
            pbar.update(1)
    return shapes, shape_groups, count


def joint_svg_optimize_visual(
    device,
    albedo_shapes,
    albedo_shape_groups,
    shade_shapes,
    shade_shape_groups,
    init_img,
    albedo_file_save_path,
    shade_file_save_path,
    train_conf,
    base_lr_conf,
    alter_opt=False,
):
    img_height, img_width = init_img.shape[:2]
    target_img = torch.tensor(init_img, device=device) / 255
    target_img = target_img.permute(2, 0, 1)

    albedo_optimizer = init_optimizer(
        albedo_shapes,
        albedo_shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_joint_color"],
        lr_base=base_lr_conf,
        is_train_point=False,
    )
    shade_optimizer = init_optimizer(
        shade_shapes,
        shade_shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_joint_color"],
        lr_base=base_lr_conf,
        is_train_point=True,
    )
    num_iters = train_conf["joint_visual_opt_num_iters"]

    if alter_opt:
        with tqdm(total=num_iters * 2, desc="Joint optimization (alternating)", unit="iter") as pbar:
            for i in range(num_iters):
                albedo_img = svg_to_img(img_width, img_height, albedo_shapes, albedo_shape_groups, device)
                albedo_img = rgba_to_rgb(albedo_img, device)
                shade_img = svg_to_img(img_width, img_height, shade_shapes, shade_shape_groups, device)
                shade_img = rgba_to_rgb(shade_img, device)
                albedo_shade_img = shade_img * albedo_img
                loss_albedo = F.mse_loss(albedo_shade_img, target_img)

                albedo_optimizer.zero_grad()
                loss_albedo.backward()
                albedo_optimizer.step()
                pydiffvg.save_svg(
                    f"{albedo_file_save_path}/albedo_{i}.svg",
                    img_width,
                    img_height,
                    albedo_shapes,
                    albedo_shape_groups,
                )
                pbar.update(1)

                albedo_img = svg_to_img(img_width, img_height, albedo_shapes, albedo_shape_groups, device)
                albedo_img = rgba_to_rgb(albedo_img, device)
                shade_img = svg_to_img(img_width, img_height, shade_shapes, shade_shape_groups, device)
                shade_img = rgba_to_rgb(shade_img, device)
                albedo_shade_img = shade_img * albedo_img
                loss_shade = F.mse_loss(albedo_shade_img, target_img)

                shade_optimizer.zero_grad()
                loss_shade.backward()
                shade_optimizer.step()
                pydiffvg.save_svg(
                    f"{shade_file_save_path}/shade_{i}.svg",
                    img_width,
                    img_height,
                    shade_shapes,
                    shade_shape_groups,
                )
                pbar.update(1)
    else:
        with tqdm(total=num_iters, desc="Joint optimization", unit="iter") as pbar:
            for i in range(num_iters):
                albedo_img = svg_to_img(img_width, img_height, albedo_shapes, albedo_shape_groups, device)
                albedo_img = rgba_to_rgb(albedo_img, device)
                shade_img = svg_to_img(img_width, img_height, shade_shapes, shade_shape_groups, device)
                shade_img = rgba_to_rgb(shade_img, device)
                albedo_shade_img = shade_img * albedo_img
                loss = F.mse_loss(albedo_shade_img, target_img)

                albedo_optimizer.zero_grad()
                shade_optimizer.zero_grad()
                loss.backward()
                albedo_optimizer.step()
                shade_optimizer.step()

                pydiffvg.save_svg(
                    f"{albedo_file_save_path}/albedo_{i}.svg",
                    img_width,
                    img_height,
                    albedo_shapes,
                    albedo_shape_groups,
                )
                pydiffvg.save_svg(
                    f"{shade_file_save_path}/shade_{i}.svg",
                    img_width,
                    img_height,
                    shade_shapes,
                    shade_shape_groups,
                )
                pbar.update(1)

    return albedo_shapes, albedo_shape_groups, shade_shapes, shade_shape_groups


def layered_vectorization(args, device=None):
    print("Joint Visual Refinement...")

    joint_shade_svgs_save_path = f"./workdir/{args.file_save_name}/joint/shade_svgs"
    os.makedirs(joint_shade_svgs_save_path, exist_ok=True)
    joint_albedo_svgs_save_path = f"./workdir/{args.file_save_name}/joint/albedo_svgs"
    os.makedirs(joint_albedo_svgs_save_path, exist_ok=True)
    joint_applied_imgs_save_path = f"./workdir/{args.file_save_name}/joint/albedo&shade_imgs"
    os.makedirs(joint_applied_imgs_save_path, exist_ok=True)
    visual_svgs_save_path = f"./workdir/{args.file_save_name}/joint/struct&visual_svgs"
    os.makedirs(visual_svgs_save_path, exist_ok=True)

    init_img = load_512(args.init_image)
    albedo_img = load_512(args.albedo_image)
    img_height, img_width = init_img.shape[:2]

    with open(args.layered_albedo_masks, "rb") as f:
        layered_albedo_masks = pickle.load(f)

    canvas_width, canvas_height, albedo_shapes, albedo_shape_groups = pydiffvg.svg_to_scene(args.albedo_svg)
    _, _, shade_shapes, shade_shape_groups = pydiffvg.svg_to_scene(args.shade_svg)

    white_bg = torch.tensor([1.0, 1.0, 1.0], requires_grad=False, device=device)
    albedo_render = svg_to_img(img_width, img_height, albedo_shapes, albedo_shape_groups, device)
    albedo_render = rgba_to_rgb(albedo_render, device, white_bg)
    shade_render = svg_to_img(img_width, img_height, shade_shapes, shade_shape_groups, device)
    shade_render = rgba_to_rgb(shade_render, device, white_bg)
    Image.fromarray(
        ((shade_render * albedo_render).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    ).save(f"{joint_applied_imgs_save_path}/origin_applied.png")

    albedo_shapes, albedo_shape_groups, shade_shapes, shade_shape_groups = joint_svg_optimize_visual(
        device,
        albedo_shapes,
        albedo_shape_groups,
        shade_shapes,
        shade_shape_groups,
        init_img,
        joint_albedo_svgs_save_path,
        joint_shade_svgs_save_path,
        args.train,
        args.base_lr,
        alter_opt=False,
    )

    if args.color_fitting_type not in ["dominan", "mse"]:
        raise ValueError(
            f"color_fitting_type must be 'dominan' or 'mse', got {args.color_fitting_type}"
        )

    for group in albedo_shape_groups:
        if group.fill_color is not None:
            group.fill_color.data = torch.clamp(group.fill_color.data, 0.0, 1.0)

    albedo_img = svg_to_img(img_width, img_height, albedo_shapes, albedo_shape_groups, device)
    albedo_img = rgba_to_rgb(albedo_img, device).detach()

    target_img_cluster = get_cluster_img(init_img, args.is_cluster_target_img, args.kmeas_k)
    Image.fromarray(target_img_cluster).save(f"./workdir/{args.file_save_name}/cluster_img.png")

    print("Visual Refinement...")
    pseudo_struct_masks = [mask for sublist in layered_albedo_masks for mask in sublist]
    is_opt_list = [1 for _ in range(len(shade_shapes))]
    count = 0
    struct_path_num = 0

    for i in range(args.add_visual_path_num_iters):
        os.makedirs(f"{visual_svgs_save_path}/{i}_add_paths", exist_ok=True)
        if i == args.add_visual_path_num_iters - 1:
            remaining_path_num = args.max_path_num_limit - len(albedo_shapes) - len(shade_shapes)
        else:
            remaining_path_num = int(
                (args.max_path_num_limit - len(albedo_shapes) - len(shade_shapes)) * 0.6
            )

        shade_shapes, shade_shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = (
            add_visual_paths_shade(
                albedo_img,
                shade_shapes,
                shade_shape_groups,
                device,
                struct_path_num,
                target_img_cluster,
                pseudo_struct_masks,
                is_opt_list,
                epsilon=args.approxpolydp_epsilon,
                N=remaining_path_num,
            )
        )
        if struct_path_num == -1:
            print("No new paths to add.")
            break

        print("Add new path")
        shade_shapes, shade_shape_groups, count = svg_optimize_img_visual_shade(
            device,
            albedo_img,
            shade_shapes,
            shade_shape_groups,
            init_img,
            f"{visual_svgs_save_path}/{i}_add_paths",
            is_opt_list,
            args.train,
            args.base_lr,
            count,
            struct_path_num,
        )
        if i == args.add_visual_path_num_iters - 1:
            break

        shade_shapes, shade_shape_groups = remove_lowquality_paths(
            shade_shapes,
            shade_shape_groups,
            device,
            img_width,
            img_height,
            visual_difference_threshold=args.paths_remove_visual_threshold,
            struct_path_num=struct_path_num,
        )

        print("Path merging")
        os.makedirs(f"{visual_svgs_save_path}/{i}_merge_paths", exist_ok=True)
        shade_shapes, shade_shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = merge_path(
            shade_shapes,
            shade_shape_groups,
            device,
            img_width,
            img_height,
            struct_path_num,
            pseudo_struct_masks,
            is_opt_list,
            color_threshold=args.paths_merge_color_threshold,
            overlapping_area_threshold=args.paths_merge_distance_threshold,
        )
        shade_shapes, shade_shape_groups, count = svg_optimize_img_visual_shade(
            device,
            albedo_img,
            shade_shapes,
            shade_shape_groups,
            init_img,
            f"{visual_svgs_save_path}/{i}_merge_paths",
            is_opt_list,
            args.train,
            args.base_lr,
            count,
            struct_path_num,
            is_path_merging_phase=True,
        )

    pydiffvg.save_svg(
        f"./workdir/{args.file_save_name}/joint/joint_albedo.svg",
        canvas_width,
        canvas_height,
        albedo_shapes,
        albedo_shape_groups,
    )
    pydiffvg.save_svg(
        f"./workdir/{args.file_save_name}/joint/joint_shade.svg",
        canvas_width,
        canvas_height,
        shade_shapes,
        shade_shape_groups,
    )

    albedo_render = svg_to_img(img_width, img_height, albedo_shapes, albedo_shape_groups, device)
    albedo_render = rgba_to_rgb(albedo_render, device, white_bg)
    shade_render = svg_to_img(img_width, img_height, shade_shapes, shade_shape_groups, device)
    shade_render = rgba_to_rgb(shade_render, device, white_bg)

    Image.fromarray(
        ((shade_render * albedo_render).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    ).save(f"{joint_applied_imgs_save_path}/joint_applied.png")
    Image.fromarray(
        (albedo_render.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    ).save(f"{joint_applied_imgs_save_path}/joint_albedo.png")
    Image.fromarray(
        (shade_render.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    ).save(f"{joint_applied_imgs_save_path}/joint_shade.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint albedo+shade optimization")
    parser.add_argument("-c", "--config", type=str, default="./config/base_config.yaml")
    parser.add_argument("-iimg", "--init_image", default="./target_imgs/init/2-thing-2.png", type=str)
    parser.add_argument("-aimg", "--albedo_image", default="./target_imgs/albedo/2-thing-2.png", type=str)
    parser.add_argument("-asvg", "--albedo_svg", default="./workdir/2-thing-2.png/64_paths/albedo/final_albedo.svg", type=str)
    parser.add_argument("-ssvg", "--shade_svg", default="./workdir/2-thing-2.png/64_paths/shade/final_shade.svg", type=str)
    parser.add_argument("-fsn", "--file_save_name", default="2-thing-2.png/64_paths", type=str)
    args = parser.parse_args()
    args = load_config(args.config, args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    init_diffvg(device=device)
    layered_vectorization(args, device)
