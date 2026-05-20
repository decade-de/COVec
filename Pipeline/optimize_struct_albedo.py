import argparse
import os
import pickle

import pydiffvg
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from sds_image_simplicity import load_512
from utils.common import exclude_loss, init_diffvg, init_optimizer, load_config
from utils.img_process import (
    add_visual_paths,
    color_fitting,
    init_struct_target_imgs,
    init_svg_by_mask,
    merge_path,
    remove_lowquality_paths,
    rgba_to_rgb,
    svg_to_img,
)


def svg_optimize_img_struct(
    device,
    shapes,
    shape_groups,
    target_img,
    layerd_struct_masks,
    file_save_path,
    train_conf,
    base_lr_conf,
):
    struct_target_imgs, struct_colors_list = init_struct_target_imgs(layerd_struct_masks)
    struct_target_imgs = [x.to(device) for x in struct_target_imgs]
    struct_shape_groups_list = []
    for struct_colors in struct_colors_list:
        struct_shape_groups = []
        for i, color in enumerate(struct_colors):
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([i]),
                fill_color=torch.FloatTensor(color + [1]),
                stroke_color=torch.FloatTensor([0, 0, 0, 1]),
            )
            struct_shape_groups.append(path_group)
        struct_shape_groups_list.append(struct_shape_groups)

    transparent_shape_groups = []
    for i in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([i]),
            fill_color=torch.FloatTensor([0, 0, 0, 0.3]),
            stroke_color=torch.FloatTensor([0, 0, 0, 0.3]),
        )
        transparent_shape_groups.append(path_group)

    black_bg = torch.tensor([0.0, 0.0, 0.0], requires_grad=False, device=device)
    white_bg = torch.tensor([1.0, 1.0, 1.0], requires_grad=False, device=device)

    img_height, img_width = target_img.shape[:2]
    target_img = torch.tensor(target_img, device=device) / 255
    target_img = target_img.permute(2, 0, 1)

    svg_optimizer = init_optimizer(
        shapes,
        shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_struct_color"],
        lr_base=base_lr_conf,
    )

    total_iters = train_conf["struct_opt_num_iters"]
    with tqdm(total=total_iters, desc="Structural optimization", unit="iter") as pbar:
        for i in range(total_iters):
            loss_struct = 0
            loss_exclude = 0
            shape_index = 0
            for struct_i, struct_target_img in enumerate(struct_target_imgs):
                shape_index += len(layerd_struct_masks[struct_i])
                slice_start = shape_index - len(layerd_struct_masks[struct_i])
                struct_img = svg_to_img(
                    img_width,
                    img_height,
                    shapes[slice_start:shape_index],
                    struct_shape_groups_list[struct_i],
                    device,
                )
                struct_img = rgba_to_rgb(struct_img, device, black_bg)
                loss_struct += F.mse_loss(struct_img, struct_target_img)

                transparent_img = svg_to_img(
                    img_width,
                    img_height,
                    shapes[slice_start:shape_index],
                    transparent_shape_groups[: len(layerd_struct_masks[struct_i])],
                    device,
                )
                transparent_img = rgba_to_rgb(transparent_img, device, white_bg)
                loss_exclude += exclude_loss(transparent_img, scale=2e-7)

            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device, white_bg)
            loss_mse = F.mse_loss(img, target_img)

            loss = loss_mse * 0.02 + loss_exclude + loss_struct
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()
            pydiffvg.save_svg(f"{file_save_path}/{i}.svg", img_width, img_height, shapes, shape_groups)
            pbar.update(1)
    return shapes, shape_groups


def svg_optimize_img_visual(
    device,
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
    num_iters = 50 if is_path_merging_phase else train_conf["visual_opt_num_iters"]

    with tqdm(total=num_iters, desc="Visual optimization", unit="iter") as pbar:
        for _ in range(num_iters):
            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device)
            loss = F.mse_loss(img, target_img)
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()
            pydiffvg.save_svg(f"{file_save_path}/{count}.svg", img_width, img_height, shapes, shape_groups)
            count += 1
            pbar.update(1)
    return shapes, shape_groups, count


def layered_vectorization(args, device=None):
    struct_svgs_save_path = f"./workdir/{args.file_save_name}/albedo/albedo_svgs"
    os.makedirs(struct_svgs_save_path, exist_ok=True)

    target_img = load_512(args.albedo_image)
    if args.is_simple:
        with open(args.layered_combined_masks, "rb") as f:
            layerd_struct_masks = pickle.load(f)
    else:
        with open(args.layered_albedo_masks, "rb") as f:
            layerd_struct_masks = pickle.load(f)
    img_height, img_width = target_img.shape[:2]

    shapes, shape_groups = init_svg_by_mask(layerd_struct_masks, target_img, args.approxpolydp_epsilon)
    shapes, shape_groups = svg_optimize_img_struct(
        device,
        shapes,
        shape_groups,
        target_img,
        layerd_struct_masks,
        struct_svgs_save_path,
        args.train,
        args.base_lr,
    )

    if args.color_fitting_type not in ["dominan", "mse"]:
        raise ValueError(
            f"color_fitting_type must be 'dominan' or 'mse', got {args.color_fitting_type}"
        )

    if args.color_fitting_type == "dominan":
        shape_groups, target_img_cluster = color_fitting(
            shape_groups,
            target_img,
            layerd_struct_masks,
            args.is_cluster_target_img,
            args.kmeas_k,
        )
        Image.fromarray(target_img_cluster).save(
            f"./workdir/{args.file_save_name}/albedo/cluster_img.png"
        )
        pydiffvg.save_svg(
            f"./workdir/{args.file_save_name}/albedo/color-adjusted.svg",
            img_height,
            img_width,
            shapes,
            shape_groups,
        )

    if args.is_simple:
        visual_svgs_save_path = f"./workdir/{args.file_save_name}/albedo/struct&visual_svgs"
        os.makedirs(visual_svgs_save_path, exist_ok=True)

        print("Visual Refinement...")
        shapes, shape_groups = remove_lowquality_paths(
            shapes,
            shape_groups,
            device,
            img_width,
            img_height,
            visual_difference_threshold=args.paths_remove_visual_threshold,
            struct_path_num=0,
        )

        pseudo_struct_masks = [mask for sublist in layerd_struct_masks for mask in sublist]
        is_opt_list = []
        count = 0
        struct_path_num = len(shapes)

        for i in range(args.add_visual_path_num_iters):
            os.makedirs(f"{visual_svgs_save_path}/{i}_add_paths", exist_ok=True)
            if i == args.add_visual_path_num_iters - 1:
                remaining_path_num = args.max_path_num_limit - len(shapes)
            else:
                remaining_path_num = int((args.max_path_num_limit - len(shapes)) * 0.6)

            shapes, shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = add_visual_paths(
                shapes,
                shape_groups,
                device,
                struct_path_num,
                target_img_cluster,
                pseudo_struct_masks,
                is_opt_list,
                epsilon=args.approxpolydp_epsilon,
                N=remaining_path_num,
            )
            if struct_path_num == -1:
                print("No new paths to add.")
                break

            print("Add new path")
            shapes, shape_groups, count = svg_optimize_img_visual(
                device,
                shapes,
                shape_groups,
                target_img,
                f"{visual_svgs_save_path}/{i}_add_paths",
                is_opt_list,
                args.train,
                args.base_lr,
                count,
                struct_path_num,
            )
            if i == args.add_visual_path_num_iters - 1:
                break

            shapes, shape_groups = remove_lowquality_paths(
                shapes,
                shape_groups,
                device,
                img_width,
                img_height,
                visual_difference_threshold=args.paths_remove_visual_threshold,
                struct_path_num=struct_path_num,
            )

            print("Path merging")
            os.makedirs(f"{visual_svgs_save_path}/{i}_merge_paths", exist_ok=True)
            shapes, shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = merge_path(
                shapes,
                shape_groups,
                device,
                img_width,
                img_height,
                struct_path_num,
                pseudo_struct_masks,
                is_opt_list,
                color_threshold=args.paths_merge_color_threshold,
                overlapping_area_threshold=args.paths_merge_distance_threshold,
            )
            shapes, shape_groups, count = svg_optimize_img_visual(
                device,
                shapes,
                shape_groups,
                target_img,
                f"{visual_svgs_save_path}/{i}_merge_paths",
                is_opt_list,
                args.train,
                args.base_lr,
                count,
                struct_path_num,
                is_path_merging_phase=True,
            )

    pydiffvg.save_svg(
        f"./workdir/{args.file_save_name}/albedo/final_albedo.svg",
        img_height,
        img_width,
        shapes,
        shape_groups,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Albedo structure optimization")
    parser.add_argument("-c", "--config", type=str, default="./config/base_config.yaml")
    parser.add_argument("-timg", "--target_image", default="./target_imgs/albedo/2-thing-2.png", type=str)
    parser.add_argument("-fsn", "--file_save_name", default="2-thing-2.png/64_paths", type=str)
    args = parser.parse_args()
    args = load_config(args.config, args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    init_diffvg(device=device)
    layered_vectorization(args, device)
