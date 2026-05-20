import argparse
import os
import pickle

import pydiffvg
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sds_image_simplicity import load_512
from utils.common import exclude_loss, init_diffvg, init_optimizer, load_config
from utils.img_process import init_struct_target_imgs, init_svg_by_mask, rgba_to_rgb, svg_to_img


def svg_optimize_shade(
    device,
    shapes,
    shape_groups,
    init_img,
    albedo_img,
    layered_shade_masks,
    file_save_path,
    train_conf,
    base_lr_conf,
):
    struct_target_imgs, struct_colors_list = init_struct_target_imgs(layered_shade_masks)
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

    img_height, img_width = init_img.shape[:2]
    init_img = torch.tensor(init_img, device=device) / 255
    init_img = init_img.permute(2, 0, 1)

    albedo_img = torch.tensor(albedo_img, device=device) / 255
    albedo_img = albedo_img.permute(2, 0, 1)

    svg_optimizer = init_optimizer(
        shapes,
        shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_struct_color"],
        lr_base=base_lr_conf,
    )

    total_iters = train_conf["shade_struct_opt_num_iters"]
    with tqdm(total=total_iters, desc="Shade structural optimization", unit="iter") as pbar:
        for i in range(total_iters):
            loss_struct = 0
            loss_exclude = 0
            shape_index = 0
            for struct_i, struct_target_img in enumerate(struct_target_imgs):
                shape_index += len(layered_shade_masks[struct_i])
                slice_start = shape_index - len(layered_shade_masks[struct_i])
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
                    transparent_shape_groups[: len(layered_shade_masks[struct_i])],
                    device,
                )
                transparent_img = rgba_to_rgb(transparent_img, device, white_bg)
                loss_exclude += exclude_loss(transparent_img, scale=2e-7)

            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device, white_bg)
            shade_albedo_img = img * albedo_img
            loss_mse = F.mse_loss(shade_albedo_img, init_img)

            loss = loss_mse * 0.02 + loss_exclude + loss_struct
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()
            pydiffvg.save_svg(f"{file_save_path}/{i}.svg", img_width, img_height, shapes, shape_groups)
            pbar.update(1)
    return shapes, shape_groups


def layered_vectorization(args, device=None):
    shade_svgs_save_path = f"./workdir/{args.file_save_name}/shade/shade_svgs"
    os.makedirs(shade_svgs_save_path, exist_ok=True)

    init_img = load_512(args.init_image)
    albedo_img = load_512(args.albedo_image)
    img_height, img_width = init_img.shape[:2]

    with open(args.layered_shade_masks, "rb") as f:
        layered_shade_masks = pickle.load(f)

    shapes, shape_groups = init_svg_by_mask(layered_shade_masks, init_img, args.approxpolydp_epsilon)
    shapes, shape_groups = svg_optimize_shade(
        device,
        shapes,
        shape_groups,
        init_img,
        albedo_img,
        layered_shade_masks,
        shade_svgs_save_path,
        args.train,
        args.base_lr,
    )

    if args.color_fitting_type not in ["dominan", "mse"]:
        raise ValueError(
            f"color_fitting_type must be 'dominan' or 'mse', got {args.color_fitting_type}"
        )

    pydiffvg.save_svg(
        f"./workdir/{args.file_save_name}/shade/final_shade.svg",
        img_height,
        img_width,
        shapes,
        shape_groups,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shade structure optimization")
    parser.add_argument("-c", "--config", type=str, default="./config/base_config.yaml")
    parser.add_argument("-iimg", "--init_image", default="./target_imgs/init/2-thing-2.png", type=str)
    parser.add_argument("-aimg", "--albedo_image", default="./target_imgs/albedo/2-thing-2.png", type=str)
    parser.add_argument("-fsn", "--file_save_name", default="2-thing-2.png/64_paths", type=str)
    args = parser.parse_args()
    args = load_config(args.config, args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    init_diffvg(device=device)
    layered_vectorization(args, device)
