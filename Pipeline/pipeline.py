import argparse
import os
import pickle

import pydiffvg
import torch
from PIL import Image

from sds_image_simplicity import load_512, sds_based_simplification
from semantic_binarization_mask import create_semantic_binarization_mask, create_visualization_mask_fast
from utils.common import init_diffvg, load_config, resolve_path
from utils.img_process import get_struct_masks_by_area, layer_segmented_masks, sam_img_seq


def maybe_generate_albedo(args, device):
    if args.is_simple:
        return

    albedo_path = os.path.abspath(args.albedo_image)
    args.albedo_image = albedo_path
    args.init_image = os.path.abspath(args.init_image)

    if os.path.isfile(albedo_path) and not args.force_generate_albedo:
        print(f"Using existing albedo: {albedo_path}")
        return

    if not args.generate_albedo and args.run_all:
        print("Albedo not found; auto-enabling Intrinsic generation for --run_all.")
        args.generate_albedo = True

    if not args.generate_albedo:
        albedo_dir = os.path.dirname(albedo_path)
        existing = sorted(os.listdir(albedo_dir)) if os.path.isdir(albedo_dir) else []
        raise FileNotFoundError(
            f"Albedo image not found: {albedo_path}\n"
            f"Existing files in {albedo_dir}: {existing}\n"
            "Run with --generate_albedo or --run_all to create it via Intrinsic, "
            "or copy/rename your albedo PNG to the expected filename."
        )

    from utils.albedo_generator import ensure_albedo_image

    ensure_albedo_image(args, device)

    if not os.path.isfile(albedo_path):
        raise RuntimeError(
            f"Intrinsic albedo generation did not produce: {albedo_path}\n"
            "Check Intrinsic installation and GPU memory, then retry."
        )


def preprocess_image(args, device):
    print("=== [Preprocess] SDS simplification & SAM masks for Albedo ===")

    simp_img_seq_save_path = f"./workdir/{args.file_save_name}/preprocess/simplified_image_sequence"
    os.makedirs(simp_img_seq_save_path, exist_ok=True)

    all_simp_img_seq_save_path = "-1"
    if args.is_save_all_simp_img_seq:
        all_simp_img_seq_save_path = f"./workdir/{args.file_save_name}/preprocess/all_simplified_image_sequence"
        os.makedirs(all_simp_img_seq_save_path, exist_ok=True)

    masks_save_path = "-1"
    if args.is_save_masks:
        masks_save_path = f"./workdir/{args.file_save_name}/preprocess/all_masks"
        os.makedirs(masks_save_path, exist_ok=True)

    albedo_masks_save_path = f"./workdir/{args.file_save_name}/preprocess/albedo_masks"
    os.makedirs(albedo_masks_save_path, exist_ok=True)
    shade_masks_save_path = f"./workdir/{args.file_save_name}/preprocess/shade_masks"
    os.makedirs(shade_masks_save_path, exist_ok=True)
    resized_imgs_save_path = f"./workdir/{args.file_save_name}/preprocess/resized_imgs"
    os.makedirs(resized_imgs_save_path, exist_ok=True)

    if args.is_sds_simplify:
        print("SDS simplification...")
        simp_img_seq = sds_based_simplification(
            device,
            args.albedo_image,
            args.simp_img_seq_indexs,
            simp_img_seq_save_path,
            all_simp_img_seq_save_path,
        )
    else:
        print("No SDS simplification, using the input image directly.")
        albedo_img = load_512(args.albedo_image)
        simp_img_seq = [albedo_img]

    target_img = simp_img_seq[0]
    Image.fromarray(target_img).save(f"{resized_imgs_save_path}/albedo.png")

    print("SAM mask generation...")
    masks = sam_img_seq(device, simp_img_seq, masks_save_path, args.sam)
    print(f"Mask count: {len(masks)}")

    print("Layered structure segmentation...")
    layered_masks = layer_segmented_masks([[masks[0]]], masks[1:])
    layered_masks = get_struct_masks_by_area(
        layered_masks, int(args.max_path_num_limit * args.albedo_mask_rate)
    )

    for group_idx, mask_group in enumerate(layered_masks):
        for mask_idx, mask in enumerate(mask_group):
            create_visualization_mask_fast(
                target_img,
                mask=mask,
                save_path=f"{albedo_masks_save_path}/mask_group{group_idx}_mask{mask_idx}.png",
            )

    print("=== [Preprocess] Semantic binarization masks for Shade ===")

    init_img = load_512(args.init_image)
    Image.fromarray(init_img).save(f"{resized_imgs_save_path}/init.png")

    layered_shade_masks = create_semantic_binarization_mask(
        init_img, layered_masks, binary_method="mean"
    )
    layered_shade_masks = get_struct_masks_by_area(
        layered_shade_masks, int(args.max_path_num_limit * args.shade_mask_rate)
    )

    for group_idx, mask_group in enumerate(layered_shade_masks):
        for mask_idx, mask in enumerate(mask_group):
            create_visualization_mask_fast(
                init_img,
                mask=mask,
                save_path=f"{shade_masks_save_path}/mask_group{group_idx}_mask{mask_idx}.png",
            )

    with open(f"./workdir/{args.file_save_name}/layerd_albedo_masks.pkl", "wb") as f:
        pickle.dump(layered_masks, f)
    print("[Done] Albedo layered mask saved.")

    with open(f"./workdir/{args.file_save_name}/layerd_shade_masks.pkl", "wb") as f:
        pickle.dump(layered_shade_masks, f)
    print("[Done] Shade layered mask saved.")

    if args.is_simple:
        combined_masks_save_path = f"./workdir/{args.file_save_name}/preprocess/combined_masks"
        os.makedirs(combined_masks_save_path, exist_ok=True)

        flatten_layered_masks = [mask for sublist in layered_masks for mask in sublist]
        flatten_layered_shade_masks = [mask for sublist in layered_shade_masks for mask in sublist]
        combined_masks = flatten_layered_masks + flatten_layered_shade_masks
        layered_combined_masks = layer_segmented_masks([[combined_masks[0]]], combined_masks[1:])

        for group_idx, mask_group in enumerate(layered_combined_masks):
            for mask_idx, mask in enumerate(mask_group):
                create_visualization_mask_fast(
                    target_img,
                    mask=mask,
                    save_path=f"{combined_masks_save_path}/mask_group{group_idx}_mask{mask_idx}.png",
                )

        with open(f"./workdir/{args.file_save_name}/layerd_combined_masks.pkl", "wb") as f:
            pickle.dump(layered_combined_masks, f)
        print("[Done] Combined layered mask saved.")


def run_albedo_optimize(args, device):
    from optimize_struct_albedo import layered_vectorization as albedo_vectorization

    print("=== Step 1: Albedo Structure Optimization ===")
    albedo_vectorization(args, device)


def run_shade_optimize(args, device):
    from optimize_struct_shade import layered_vectorization as shade_vectorization

    print("=== Step 2: Shade Structure Optimization ===")
    shade_vectorization(args, device)


def run_joint_optimization(args, device):
    from optimize_joint import layered_vectorization as joint_vectorization

    print("=== Step 3: Joint Optimization (Albedo + Shade) ===")
    joint_vectorization(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Albedo+Shade+Joint Optimization Pipeline")
    parser.add_argument("-c", "--config", type=str, default="./config/base_config.yaml", help="YAML config file")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess input image (SDS + masks)")
    parser.add_argument("--run_albedo", action="store_true", help="Run albedo optimization")
    parser.add_argument("--run_shade", action="store_true", help="Run shade optimization")
    parser.add_argument("--run_joint", action="store_true", help="Run joint optimization")
    parser.add_argument("--postprocess", action="store_true", help="Postprocess the results")
    parser.add_argument("--run_all", action="store_true", help="Run all steps")
    parser.add_argument("--image_name", type=str, default="2-thing-2.png", help="Input image name")
    parser.add_argument(
        "--is_simple",
        action="store_true",
        help="Use simple mode for flat images (icons); skips shade/joint/postprocess",
    )
    parser.add_argument("--path_num", type=int, default=64)
    parser.add_argument(
        "--generate_albedo",
        action="store_true",
        help="Generate albedo from init image using Intrinsic (complex mode only)",
    )
    parser.add_argument(
        "--force_generate_albedo",
        action="store_true",
        help="Regenerate albedo even if target_imgs/albedo/<image> already exists",
    )

    args = parser.parse_args()
    args = load_config(args.config, args, parser=parser)
    args.max_path_num_limit = args.path_num
    args.file_save_name = f"{args.image_name}/{args.path_num}_paths"

    sam_checkpoint = resolve_path(args.sam["sam_checkpoint"], env_var="SAM_CHECKPOINT")
    if not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {sam_checkpoint}\n"
            "Download sam_vit_h_4b8939.pth or set SAM_CHECKPOINT to its path."
        )
    args.sam["sam_checkpoint"] = sam_checkpoint

    image_root = "./target_imgs"
    if args.is_simple:
        args.albedo_image = os.path.join(image_root, "init", args.image_name)
    else:
        args.albedo_image = os.path.join(image_root, "albedo", args.image_name)
    args.init_image = os.path.join(image_root, "init", args.image_name)
    args.layered_albedo_masks = f"./workdir/{args.file_save_name}/layerd_albedo_masks.pkl"
    args.layered_shade_masks = f"./workdir/{args.file_save_name}/layerd_shade_masks.pkl"
    args.layered_combined_masks = f"./workdir/{args.file_save_name}/layerd_combined_masks.pkl"
    args.albedo_svg = f"./workdir/{args.file_save_name}/albedo/final_albedo.svg"
    args.shade_svg = f"./workdir/{args.file_save_name}/shade/final_shade.svg"
    args.joint_albedo_svg = f"./workdir/{args.file_save_name}/joint/joint_albedo.svg"
    args.joint_shade_svg = f"./workdir/{args.file_save_name}/joint/joint_shade.svg"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    init_diffvg(device=device)

    maybe_generate_albedo(args, device)

    if args.preprocess or args.run_all:
        preprocess_image(args, device)

    if args.run_albedo or args.run_all:
        run_albedo_optimize(args, device)

    if not args.is_simple:
        if args.run_shade or args.run_all:
            run_shade_optimize(args, device)

        if args.run_joint or args.run_all:
            run_joint_optimization(args, device)

        if args.postprocess or args.run_all:
            from svg_post_process import post_process as svg_post_process

            print("=== Step 4: Post-process the results ===")
            svg_post_process(args, device)

    print("=== Pipeline Completed ===")
