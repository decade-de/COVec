"""Shared utilities for SVG optimization scripts."""

import os
from typing import List

import pydiffvg
import torch
import torch.nn.functional as F
import yaml


def init_diffvg(
    device: torch.device,
    use_gpu: bool = torch.cuda.is_available(),
    print_timing: bool = False,
):
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_print_timing(print_timing)


def init_optimizer(
    shapes,
    shape_groups,
    is_train_stroke: bool = False,
    is_train_color: bool = True,
    is_opt_list: List[int] = None,
    lr_base: dict = None,
    is_train_point: bool = True,
):
    if is_opt_list is None:
        is_opt_list = []
    if lr_base is None:
        lr_base = {}

    points_vars = []
    color_vars = []
    stroke_width_vars = []
    stroke_color_vars = []

    if len(is_opt_list) == 0:
        is_opt_list = [1 for _ in range(len(shapes))]

    for i, path in enumerate(shapes):
        if is_opt_list[i] == 1:
            path.id = i
            path.points.requires_grad = is_train_point
            points_vars.append(path.points)
            if is_train_stroke:
                path.stroke_width.requires_grad = True
                stroke_width_vars.append(path.stroke_width)

    if is_train_color:
        for i, group in enumerate(shape_groups):
            if is_opt_list[i] == 1:
                group.fill_color.requires_grad = True
                color_vars.append(group.fill_color)
                if is_train_stroke:
                    group.stroke_color.requires_grad = True
                    stroke_color_vars.append(group.stroke_color)

    params = {"point": points_vars}
    if is_train_color:
        params["color"] = color_vars
    if is_train_stroke:
        params["stroke_width"] = stroke_width_vars
        params["stroke_color"] = stroke_color_vars

    learnable_params = [
        {"params": params[key], "lr": lr_base[key], "_id": str(key)}
        for key in sorted(params.keys())
    ]
    return torch.optim.Adam(learnable_params, betas=(0.9, 0.9), eps=1e-6)


def exclude_loss(raster_img, scale=1):
    img = F.relu(178 / 255 - raster_img)
    return torch.sum(img) * scale


def load_config(file_path, args, parser=None):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    defaults = {}
    if parser is not None:
        defaults = {
            action.dest: action.default
            for action in parser._actions
            if action.dest not in ("help", "config")
        }

    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        # Keep explicit CLI flags (non-default values) higher priority than YAML.
        elif parser is None or getattr(args, key) == defaults.get(key):
            setattr(args, key, value)
    return args


def resolve_path(path: str, env_var: str | None = None) -> str:
    if env_var and os.environ.get(env_var):
        path = os.environ[env_var]
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path
