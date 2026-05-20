"""Generate albedo maps from RGB images using the Intrinsic decomposition pipeline."""

import argparse
import os
import time
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    from chrislib.data_util import load_image
    from chrislib.general import view
    from intrinsic.pipeline import load_models, run_pipeline
except ImportError as exc:
    raise ImportError(
        "Intrinsic is not installed. Install it with:\n"
        "  pip install git+https://github.com/compphoto/Intrinsic.git"
    ) from exc


def _resolve_checkpoints(intrinsic_conf: dict) -> Union[str, List[str]]:
    explicit = intrinsic_conf.get("checkpoints")
    if explicit:
        return explicit

    checkpoint_dir = os.path.expanduser(
        intrinsic_conf.get("checkpoint_dir", "~/.cache/torch/hub/checkpoints")
    )
    stage_files = [os.path.join(checkpoint_dir, f"stage_{i}.pt") for i in range(5)]
    if all(os.path.isfile(path) for path in stage_files):
        return stage_files

    model_version = intrinsic_conf.get("model_version", "v2")
    return model_version


class AlbedoGenerator:
    """Load Intrinsic once and reuse it for single or batch albedo generation."""

    def __init__(self, intrinsic_conf: dict, device: Union[str, torch.device] = "cuda"):
        self.device = str(device)
        model_spec = _resolve_checkpoints(intrinsic_conf)
        print("Loading Intrinsic model...")
        self.intrinsic_model = load_models(model_spec)
        print("Intrinsic model loaded.")

    def generate_single_albedo(self, input_path: str, output_path: str) -> bool:
        try:
            image = load_image(input_path)
            result = run_pipeline(self.intrinsic_model, image, device=self.device)
            albedo = view(result["hr_alb"])

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            Image.fromarray(np.uint8(np.clip(albedo, 0.0, 1.0) * 255)).save(output_path)
            return True
        except Exception as exc:
            print(f"Failed to generate albedo for {input_path}: {exc}")
            return False

    def batch_generate_albedo(self, input_folder: str, output_folder: str) -> int:
        os.makedirs(output_folder, exist_ok=True)
        supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
        image_files = sorted(
            name
            for name in os.listdir(input_folder)
            if name.lower().endswith(supported_formats)
        )

        print(f"Found {len(image_files)} images in {input_folder}")
        success_count = 0
        for index, filename in enumerate(image_files, start=1):
            print(f"Processing {index}/{len(image_files)}: {filename}")
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            if self.generate_single_albedo(input_path, output_path):
                success_count += 1
                print(f"Saved albedo to {output_path}")

        print(f"Batch complete: {success_count}/{len(image_files)} succeeded.")
        return success_count

    def release(self):
        self.intrinsic_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_generator: Optional[AlbedoGenerator] = None


def get_albedo_generator(intrinsic_conf: dict, device: Union[str, torch.device]) -> AlbedoGenerator:
    global _generator
    if _generator is None:
        _generator = AlbedoGenerator(intrinsic_conf, device=device)
    return _generator


def release_albedo_generator():
    global _generator
    if _generator is not None:
        _generator.release()
        _generator = None


def ensure_albedo_image(args, device) -> str:
    """Generate albedo from init image when needed."""
    if args.is_simple:
        return args.albedo_image

    albedo_path = args.albedo_image
    if os.path.isfile(albedo_path) and not args.force_generate_albedo:
        print(f"Using existing albedo: {albedo_path}")
        return albedo_path

    if not args.generate_albedo:
        raise FileNotFoundError(
            f"Albedo image not found at {albedo_path}. "
            "Place it under target_imgs/albedo/ or run with --generate_albedo."
        )

    if not os.path.isfile(args.init_image):
        raise FileNotFoundError(f"Init image not found at {args.init_image}")

    print("=== [Albedo] Generating albedo from init image via Intrinsic ===")
    print(f"  Input (init):  {os.path.abspath(args.init_image)}")
    print(f"  Output (albedo): {os.path.abspath(albedo_path)}")
    generator = get_albedo_generator(args.intrinsic, device)
    if not generator.generate_single_albedo(args.init_image, albedo_path):
        raise RuntimeError(f"Intrinsic albedo generation failed for {args.init_image}")

    if args.release_intrinsic_after_albedo:
        release_albedo_generator()

    print(f"[Done] Albedo saved to {albedo_path}")
    return albedo_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate albedo maps with Intrinsic")
    parser.add_argument("--input", type=str, help="Single RGB input image")
    parser.add_argument("--output", type=str, help="Output albedo PNG for --input")
    parser.add_argument("--input_dir", type=str, help="Folder of RGB input images (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Folder to save albedo PNGs (batch mode)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="~/.cache/torch/hub/checkpoints",
        help="Directory containing stage_0.pt ... stage_4.pt",
    )
    parser.add_argument("--model_version", type=str, default="v2", help="Fallback if stage checkpoints are missing")
    parser.add_argument("--device", type=str, default="cuda")
    cli_args = parser.parse_args()

    single_mode = bool(cli_args.input or cli_args.output)
    batch_mode = bool(cli_args.input_dir or cli_args.output_dir)
    if single_mode == batch_mode:
        parser.error("Use either --input/--output (single) or --input_dir/--output_dir (batch).")

    if single_mode and not (cli_args.input and cli_args.output):
        parser.error("--input and --output must be provided together.")

    intrinsic_conf = {
        "checkpoint_dir": cli_args.checkpoint_dir,
        "model_version": cli_args.model_version,
    }

    load_start = time.perf_counter()
    generator = AlbedoGenerator(intrinsic_conf, device=cli_args.device)
    load_seconds = time.perf_counter() - load_start
    print(f"[Timing] Model load: {load_seconds:.1f}s")

    if single_mode:
        infer_start = time.perf_counter()
        ok = generator.generate_single_albedo(cli_args.input, cli_args.output)
        infer_seconds = time.perf_counter() - infer_start
        print(f"[Timing] Inference: {infer_seconds:.1f}s")
        if ok:
            print(f"[Done] Albedo saved to {os.path.abspath(cli_args.output)}")
        raise SystemExit(0 if ok else 1)

    generator.batch_generate_albedo(cli_args.input_dir, cli_args.output_dir)
