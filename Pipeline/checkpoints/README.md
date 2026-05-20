# Checkpoints

Download model weights here before running the pipeline.

## SAM ViT-H (required)

```bash
wget -O sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Then set the path in `config/base_config.yaml`:

```yaml
sam:
  sam_checkpoint: "checkpoints/sam_vit_h_4b8939.pth"
```

Or export an absolute path:

```bash
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
```

## Intrinsic albedo weights (optional)

Only needed for `--generate_albedo` or `utils/albedo_generator.py`.

Place `stage_0.pt` … `stage_4.pt` under `~/.cache/torch/hub/checkpoints/`, or let Intrinsic download them automatically when using `model_version: v2` in config.
