import numpy as np
import cv2
from scipy import ndimage
from skimage import measure

from utils.img_process import layer_segmented_masks


def create_visualization_mask_fast(original_img, mask, save_path):
    """Overlay a semi-transparent red mask on the original image for debugging."""
    from PIL import Image

    if mask.max() > 1:
        mask_bool = (mask > 128).astype(bool)
    else:
        mask_bool = (mask > 0.5).astype(bool)

    h, w = original_img.shape[:2]
    if original_img.shape[2] == 3:
        rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_img[:, :, :3] = original_img
        rgba_img[:, :, 3] = 255
    else:
        rgba_img = original_img.copy()

    red_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    red_overlay[mask_bool] = [255, 0, 0, 128]

    dilated_mask = ndimage.binary_dilation(mask_bool, structure=np.ones((3, 3)))
    border_mask = dilated_mask & ~mask_bool
    red_overlay[border_mask] = [255, 255, 255, 255]

    alpha_red = red_overlay[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_red
    for channel in range(3):
        rgba_img[:, :, channel] = alpha_red * red_overlay[:, :, channel] + alpha_bg * rgba_img[:, :, channel]

    if original_img.shape[2] == 4:
        rgba_img[:, :, 3] = np.maximum(red_overlay[:, :, 3], original_img[:, :, 3])

    Image.fromarray(rgba_img).save(save_path, "PNG")


def mask_to_binary_images(init_img, layered_masks, binary_method="mean"):
    """Binarize each semantic mask region in the init image."""
    binary_images = []

    for group_idx, mask_group in enumerate(layered_masks):
        group_binary_images = []

        for mask_idx, mask in enumerate(mask_group):
            if mask.max() > 1:
                mask_bool = (mask > 128).astype(bool)
            else:
                mask_bool = (mask > 0.5).astype(bool)

            masked_region = np.zeros_like(init_img)
            masked_region[mask_bool] = init_img[mask_bool]
            gray = cv2.cvtColor(masked_region, cv2.COLOR_RGB2GRAY)
            gray_values_in_mask = gray[mask_bool]

            if len(gray_values_in_mask) > 0:
                if binary_method == "otsu":
                    _, binary_img = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                elif binary_method == "adaptive":
                    binary_img = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 2
                    )
                elif binary_method == "global":
                    _, binary_img = cv2.threshold(gray, 101, 255, cv2.THRESH_BINARY_INV)
                elif binary_method == "mean":
                    if len(gray_values_in_mask) > 10:
                        mean_value = np.mean(gray_values_in_mask)
                        std_value = np.std(gray_values_in_mask)
                        if std_value < 5:
                            continue
                        _, binary_img = cv2.threshold(
                            gray, mean_value, 255, cv2.THRESH_BINARY_INV
                        )
                    else:
                        _, binary_img = cv2.threshold(
                            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                        )
                else:
                    raise ValueError(f"Unknown binarization method: {binary_method}")

                binary_img[~mask_bool] = 0
            else:
                binary_img = np.zeros_like(gray)

            group_binary_images.append(binary_img)
            pixel_count = np.sum((binary_img > 128).astype(bool))
            print(f"Processed mask group {group_idx}, mask {mask_idx}: {pixel_count} pixels")

        binary_images.append(group_binary_images)

    print("Binarization complete.")
    return binary_images


def separate_mask_by_region(layered_masks, min_area=110, max_area=100000):
    """Split binarized masks into connected components and filter by area."""
    separated_masks = []
    for mask_group in layered_masks:
        for mask in mask_group:
            labeled_mask = measure.label(mask)
            regions = measure.regionprops(labeled_mask)

            for region in regions:
                single_region_mask = np.zeros_like(mask, dtype=np.uint8)
                coords = region.coords
                single_region_mask[coords[:, 0], coords[:, 1]] = 255

                if min_area < region.area < max_area:
                    separated_masks.append(single_region_mask)
                    pixel_count = np.sum((single_region_mask > 128).astype(bool))
                    print(f"Separated region with {pixel_count} pixels")

    return separated_masks


def create_semantic_binarization_mask(
    init_img, layered_masks, binary_method="mean", min_area=110, max_area=100000
):
    """Build layered shade masks via semantic binarization and connected-component splitting."""
    binary_images = mask_to_binary_images(init_img, layered_masks, binary_method=binary_method)
    separated_masks = separate_mask_by_region(binary_images, min_area=min_area, max_area=max_area)

    print(f"Mask count after separation: {len(separated_masks)}")
    if len(separated_masks) > 1:
        layered_binary_masks = layer_segmented_masks([[separated_masks[0]]], separated_masks[1:])
    elif len(separated_masks) == 1:
        layered_binary_masks = [[separated_masks[0]]]
    else:
        layered_binary_masks = [[]]

    return layered_binary_masks
