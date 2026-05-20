"""Post-process joint SVG outputs: clamp colors, extract highlights, merge layers."""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pydiffvg
import torch
from PIL import Image

from sds_image_simplicity import load_512
from utils.img_process import rgba_to_rgb, svg_to_img


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_viewbox_and_size(svg_text: str):
    match = re.search(r"<svg\b([^>]*)>", svg_text, flags=re.IGNORECASE | re.DOTALL)
    attrs = match.group(1) if match else ""

    def get_attr(name):
        attr_match = re.search(rf'{name}\s*=\s*["\']([^"\']+)["\']', attrs, flags=re.IGNORECASE)
        return attr_match.group(1) if attr_match else None

    return get_attr("width"), get_attr("height"), get_attr("viewBox")


def extract_inner(svg_text: str) -> str:
    match = re.search(r"<svg\b[^>]*>(.*)</svg\s*>", svg_text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else svg_text.strip()


def build_layered_svg(albedo_text, shade_text, light_text, out_path):
    width, height, viewbox = extract_viewbox_and_size(albedo_text)
    albedo_inner = extract_inner(albedo_text)
    shade_inner = extract_inner(shade_text)
    light_inner = extract_inner(light_text) if light_text is not None else ""

    attrs = []
    if width:
        attrs.append(f'width="{width}"')
    if height:
        attrs.append(f'height="{height}"')
    if viewbox:
        attrs.append(f'viewBox="{viewbox}"')
    root_attrs = " ".join(attrs) if attrs else 'viewBox="0 0 1024 1024"'

    out = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg {root_attrs} xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      #stack {{ isolation:isolate; }}
      #layer-albedo {{ mix-blend-mode:normal; }}
      #layer-shade  {{ mix-blend-mode:multiply; }}
      #layer-light  {{ mix-blend-mode:plus-lighter; }}
    </style>
  </defs>
  <g id="stack">
    <g id="layer-albedo">{albedo_inner}</g>
    <g id="layer-shade">{shade_inner}</g>
    <g id="layer-light">{light_inner}</g>
  </g>
</svg>
'''
    Path(out_path).write_text(out, encoding="utf-8")


def get_shape_average_color(shape, shape_group, reference_img, img_width, img_height, device):
    """Sample the mean color of a shape region from a reference image."""
    with torch.no_grad():
        transparent_bg = torch.tensor([0.0, 0.0, 0.0], device=device)
        mask_shape_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([0]),
            fill_color=torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            stroke_color=None,
        )
        mask_img = svg_to_img(img_width, img_height, [shape], [mask_shape_group], device=device)
        mask_img = rgba_to_rgb(mask_img, device, transparent_bg)
        mask = mask_img[1] > 0.5

        if mask.sum() == 0:
            if shape_group.fill_color is not None:
                return shape_group.fill_color.data.clone()
            return torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)

        mean_r = torch.mean(reference_img[0][mask]).unsqueeze(0)
        mean_g = torch.mean(reference_img[1][mask]).unsqueeze(0)
        mean_b = torch.mean(reference_img[2][mask]).unsqueeze(0)
        return torch.cat([mean_r, mean_g, mean_b, torch.tensor([1.0], device=device)])


def post_process(args, device=None):
    init_img_path = args.init_image
    albedo_svg_path = args.joint_albedo_svg
    shade_svg_path = args.joint_shade_svg

    _, _, albedo_shapes, albedo_shape_groups = pydiffvg.svg_to_scene(albedo_svg_path)
    img_width, img_height, shade_shapes, shade_shape_groups = pydiffvg.svg_to_scene(shade_svg_path)

    init_img = load_512(init_img_path)
    img_height, img_width = init_img.shape[0:2]
    if isinstance(init_img, np.ndarray):
        init_img = torch.from_numpy(init_img).permute(2, 0, 1).float() / 255.0
    init_img = init_img.to(device)

    remove_index_list = []

    for group in albedo_shape_groups:
        if group.fill_color is not None:
            if group.fill_color.data[0:3].min() < 0.0 or group.fill_color.data[0:3].max() > 1.0:
                group.fill_color.data = torch.clamp(group.fill_color.data, 0.0, 1.0)

    for i, group in enumerate(shade_shape_groups):
        if group.fill_color is not None:
            if group.fill_color.data[0:3].max() > 1.05:
                group.fill_color.data = torch.clamp(group.fill_color.data, 0.0, 1.0)
                remove_index_list.append(i)

    new_shade_shapes = [shape for i, shape in enumerate(shade_shapes) if i not in remove_index_list]
    new_shade_shape_groups = [group for i, group in enumerate(shade_shape_groups) if i not in remove_index_list]
    for i in range(len(new_shade_shape_groups)):
        new_shade_shape_groups[i].shape_ids = torch.tensor([i])

    highlight_shapes = [shape for i, shape in enumerate(shade_shapes) if i in remove_index_list]
    highlight_shape_groups = [group for i, group in enumerate(shade_shape_groups) if i in remove_index_list]
    for i in range(len(highlight_shape_groups)):
        highlight_shape_groups[i].shape_ids = torch.tensor([i])

    white_bg = torch.tensor([1.0, 1.0, 1.0], requires_grad=False, device=device)
    albedo_img = svg_to_img(img_width, img_height, albedo_shapes, albedo_shape_groups, device)
    albedo_img = rgba_to_rgb(albedo_img, device, white_bg)
    shade_img = svg_to_img(img_width, img_height, new_shade_shapes, new_shade_shape_groups, device)
    shade_img = rgba_to_rgb(shade_img, device, white_bg)
    albedo_shade_img = shade_img * albedo_img

    for shape, shape_group in zip(highlight_shapes, highlight_shape_groups):
        if shape_group.fill_color is not None:
            avg_color = get_shape_average_color(
                shape, shape_group, init_img - albedo_shade_img, img_width, img_height, device
            )
            shape_group.fill_color.data = avg_color.clone().detach()

    highlight_img = None
    if highlight_shapes:
        highlight_img = svg_to_img(img_width, img_height, highlight_shapes, highlight_shape_groups, device)
        highlight_img = rgba_to_rgb(highlight_img, device, torch.tensor([0.0, 0.0, 0.0], device=device))
        albedo_shade_img = torch.clamp(albedo_shade_img + highlight_img, 0.0, 1.0)

    post_svgs_save_path = f"./workdir/{args.file_save_name}/post_process"
    os.makedirs(post_svgs_save_path, exist_ok=True)

    preview = (albedo_shade_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(preview).save(f"{post_svgs_save_path}/post_applied.png")

    pydiffvg.save_svg(
        f"{post_svgs_save_path}/post_albedo.svg", img_width, img_height, albedo_shapes, albedo_shape_groups
    )
    pydiffvg.save_svg(
        f"{post_svgs_save_path}/post_shade.svg", img_width, img_height, new_shade_shapes, new_shade_shape_groups
    )
    if highlight_shapes:
        pydiffvg.save_svg(
            f"{post_svgs_save_path}/post_highlight.svg",
            img_width,
            img_height,
            highlight_shapes,
            highlight_shape_groups,
        )

    albedo_text = read_text(Path(f"{post_svgs_save_path}/post_albedo.svg"))
    shade_text = read_text(Path(f"{post_svgs_save_path}/post_shade.svg"))
    light_text = (
        read_text(Path(f"{post_svgs_save_path}/post_highlight.svg")) if highlight_shapes else None
    )
    build_layered_svg(albedo_text, shade_text, light_text, Path(f"./workdir/{args.file_save_name}/result.svg"))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--shade_svg", type=str, help="Shade SVG path")
    parser.add_argument("--albedo_svg", type=str, help="Albedo SVG path")
    parser.add_argument("--init_img", type=str, help="Init image path")
    args = parser.parse_args()
    post_process(args, device)
