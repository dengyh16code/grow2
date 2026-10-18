import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import copy
import json
import re

# Object parsing: category vocabulary, image views, and response validation.
def categories_from_prompt(prompt):
    """Read the standalone JSON category list in the simulation prompt."""
    match = re.search(r"^\[\s*\n.*?^\]", prompt, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError("Simulation object prompt must contain a standalone JSON category list")
    categories = json.loads(match.group())
    if not categories or any(not isinstance(name, str) or not name.strip() for name in categories):
        raise ValueError("Object categories must be nonempty strings")
    return tuple(dict.fromkeys(categories))


def prepare_object_views(image_list, use_crops=True):
    """Keep each full image and add four overlapping crops at native resolution.

    Crops cover 65% of each dimension, so objects near the center or a tile
    boundary retain context. No scene labels, poses, or masks are used.
    """
    if not image_list:
        raise ValueError("Object parsing requires at least one image")
    images, views = [], []
    for source, image in enumerate(image_list):
        image = image.convert("RGB")
        width, height = image.size
        boxes = [("full", (0, 0, width, height))]
        if use_crops and min(width, height) > 1:
            crop_w = min(width, round(width * .65))
            crop_h = min(height, round(height * .65))
            for label, x, y in (
                ("top left", 0, 0),
                ("top right", width - crop_w, 0),
                ("bottom left", 0, height - crop_h),
                ("bottom right", width - crop_w, height - crop_h),
            ):
                boxes.append((label, (x, y, x + crop_w, y + crop_h)))
        for label, box in boxes:
            images.append(image if label == "full" else image.crop(box))
            views.append({"source": source, "view": label, "box": list(box)})
    context = (
        "Images are ordered as listed below. Each source's full image and crops "
        "show the SAME scene, not additional objects. Combine evidence across "
        "views and list each object once. Crop boxes use original image pixels.\n"
        + json.dumps(views)
    )
    return images, views, context


def _decode_json(response):
    if not isinstance(response, str):
        raise ValueError("Expected a JSON object response from the VLM")
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    data = json.loads(fenced.group(1) if fenced else text)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object response from the VLM")
    return data


def decode_object_list(response, allowed_categories=None):
    """Normalize and deduplicate names; restrict categories only when supplied."""
    objects = _decode_json(response).get("objects")
    if not isinstance(objects, list) or any(
        not isinstance(obj, str) or not obj.strip() for obj in objects
    ):
        raise ValueError("objects must be a list of nonempty category strings")
    canonical = (
        {name.casefold(): name for name in allowed_categories}
        if allowed_categories is not None else None
    )
    result, seen = [], set()
    for obj in objects:
        normalized = " ".join(obj.split()).casefold()
        if canonical is not None and normalized not in canonical:
            raise ValueError(f"Unsupported object category: {obj!r}")
        if normalized not in seen:
            seen.add(normalized)
            result.append(canonical[normalized] if canonical is not None else normalized)
    return result


###############################################
# AFFORDANCE DETECTION
###############################################
AFFORDANCE_COLORS = {
    "tool_grasp_part": (255, 0, 0),      # red
    "tool_function_part": (0, 255, 0),       # green
    "target_function_part": (0, 0, 255)        # blue
}

def crop_image(image, bbox, offset_ratio=0.1):
    x1, y1, x2, y2 = bbox

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    w_extend = int(w * (1 + offset_ratio))
    h_extend = int(h * (1 + offset_ratio))
    side = max(1, max(w_extend, h_extend))  # prevent side = 0

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_x1 = int(cx - side / 2)
    new_y1 = int(cy - side / 2)

    new_x1 = max(0, min(new_x1, image.width - 1))
    new_y1 = max(0, min(new_y1, image.height - 1))

    new_x2 = min(image.width,  new_x1 + side)
    new_y2 = min(image.height, new_y1 + side)

    if new_x2 <= new_x1:
        new_x2 = min(image.width, new_x1 + 1)
    if new_y2 <= new_y1:
        new_y2 = min(image.height, new_y1 + 1)

    crop_box = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
    cropped = image.crop(crop_box)

    return cropped, crop_box




def crop_image_and_apply_mask(image, bbox, mask, offset_ratio=0.1):
    """
    image: PIL.Image (RGB)
    mask:  numpy.ndarray, uint8, values {0,1}, shape (H,W)
    bbox:  (x1, y1, x2, y2)
    """

    # ---------- 1. Apply mask ----------
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Convert {0,1} -> {0,255}
    mask_255 = (mask * 255)

    # Convert to PIL mask
    mask_pil = Image.fromarray(mask_255, mode="L")

    # White background
    white_bg = Image.new("RGB", image.size, (255, 255, 255))

    # Composite image: mask==1 keeps color, else white
    masked_image = Image.composite(image, white_bg, mask_pil)

    if bbox is None:
        return masked_image, None

    x1, y1, x2, y2 = bbox
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    w_extend = int(w * (1 + offset_ratio))
    h_extend = int(h * (1 + offset_ratio))
    side = max(1, max(w_extend, h_extend))  # prevent side = 0

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_x1 = int(cx - side / 2)
    new_y1 = int(cy - side / 2)

    new_x1 = max(0, min(new_x1, image.width - 1))
    new_y1 = max(0, min(new_y1, image.height - 1))

    new_x2 = min(image.width,  new_x1 + side)
    new_y2 = min(image.height, new_y1 + side)

    if new_x2 <= new_x1:
        new_x2 = min(image.width, new_x1 + 1)
    if new_y2 <= new_y1:
        new_y2 = min(image.height, new_y1 + 1)

    crop_box = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
    cropped = masked_image.crop(crop_box)

    return cropped, crop_box


def project_mask_to_original(aff_mask, crop_bbox, original_size):
    W, H = original_size
    full_mask = np.zeros((H, W), dtype=np.uint8)

    x1, y1, x2, y2 = crop_bbox
    crop_h = y2 - y1
    crop_w = x2 - x1

    # Resize mask to match crop size
    if aff_mask.shape != (crop_h, crop_w):
        aff_mask = Image.fromarray(aff_mask).resize((crop_w, crop_h), Image.NEAREST)
        aff_mask = np.array(aff_mask)

    full_mask[y1:y2, x1:x2] = aff_mask
    return full_mask

def apply_affordance_color(original_img, mask, aff_type):
    img = copy.deepcopy(original_img)
    r, g, b = AFFORDANCE_COLORS[aff_type]
    img[mask > 0] = img[mask > 0] + [r, g, b]
    return img


###############################################
# OBJECT detection UTILS
###############################################
def xyxy_to_xywh(box):
    """
    Convert [x1, y1, x2, y2] to [x, y, w, h].

    Returns None for a missing box or Qwen's [0, 0, 0, 0] no-result sentinel.
    """
    if box is None:
        return None
    if len(box) == 4 and all(value == 0 for value in box):
        return None
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2 - x1, y2 - y1]


###############################################
# visualization
###############################################

def plot_mask(mask, color="r", ax=None):
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)

def plot_bbox(
    img_height,
    img_width,
    box,
    box_format="XYXY",
    relative_coords=True,
    color="r",
    linestyle="solid",
    text=None,
    ax=None,
):
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        facecolor = "w"
        ax.text(
            x,
            y - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},
        )

def plot_best(image, mask, bbox, score, save=False, save_path="seg.png", show=True):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    if mask is not None:
        plot_mask(mask, color="#FF4500")

    if bbox is not None:
        if score is None:
            plot_bbox(image.size[1], image.size[0], bbox, text=None,
                      box_format="XYXY", color="b", relative_coords=False)
        else:
            plot_bbox(image.size[1], image.size[0], bbox, text=f"Score: {score:.2f}",
                      box_format="XYXY", color="b", relative_coords=False)

    plt.axis("off")  # hide axis

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    plt.close()
