from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

MAX_DISPLAY_SAMPLES = 4
DEFAULT_OVERLAY_ALPHA = 0.55
DIFFERENCE_OFFSET = 0.5
DEFAULT_MASK_OVERLAY_COLOR = (1.0, 0.25, 0.25)

GOLDEN_RATIO_CONJUGATE = 0.61803398875
PALETTE_SATURATION = 0.75
PALETTE_VALUE = 0.95


def _extract_middle_slice(im: Tensor) -> Tensor:
    """Middle slice along dim 3 of a (N, C, D, H, W) volume."""
    if len(im.shape) == 5:
        return im[:, :, :, im.shape[3] // 2]
    return im


def render_for_tensorboard(im: Tensor, max_samples: int = MAX_DISPLAY_SAMPLES) -> Tensor:
    """Render the middle slice of an already normalized [0, 1] image as RGB."""
    im = _extract_middle_slice(im).float().clamp(0, 1)
    return im[:max_samples, [0, 0, 0]].detach().cpu()


def segmentation_labels_for_tensorboard(im: Tensor, max_samples: int = MAX_DISPLAY_SAMPLES) -> Tensor:
    """Convert single-channel segmentations to integer label images."""
    return torch.round(_extract_middle_slice(im)[:max_samples, 0]).long()


def _hsv_to_rgb(h: Tensor, s: Tensor, v: Tensor) -> Tensor:
    i = torch.floor(h * 6.0).long()
    f = h * 6.0 - i.float()
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    mask = i == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = i == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = i == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = i == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = i == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = i == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    return torch.stack([r, g, b], dim=1)


def _segmentation_palette(num_colors: int, device: torch.device) -> Tensor:
    palette = torch.zeros((max(num_colors, 1), 3), dtype=torch.float32, device=device)
    if num_colors <= 1:
        return palette

    class_ids = torch.arange(1, num_colors, dtype=torch.float32, device=device)
    hues = torch.remainder(class_ids * GOLDEN_RATIO_CONJUGATE, 1.0)
    saturation = torch.full_like(hues, PALETTE_SATURATION)
    value = torch.full_like(hues, PALETTE_VALUE)
    palette[1:] = _hsv_to_rgb(hues, saturation, value)
    return palette


def labels_to_color_image(labels: Tensor) -> Tensor:
    num_colors = int(labels.max().item()) + 1 if labels.numel() > 0 else 1
    palette = _segmentation_palette(num_colors, labels.device)
    color_idx = torch.remainder(labels, palette.shape[0])
    rgb = palette[color_idx]
    return rgb.permute(0, 3, 1, 2)


def render_segmentation_overlay_for_tensorboard(
    image: Tensor,
    segmentation: Tensor,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> Tensor:
    """Blend a rendered segmentation over its image for TensorBoard."""
    image_rgb = render_for_tensorboard(image).to(segmentation.device)
    labels = segmentation_labels_for_tensorboard(segmentation)
    seg_rgb = labels_to_color_image(labels)
    seg_mask = (labels > 0).unsqueeze(1)
    overlay = torch.where(
        seg_mask,
        (1.0 - alpha) * image_rgb + alpha * seg_rgb,
        image_rgb,
    )
    return overlay.detach().cpu()


def render_mask_overlay_for_tensorboard(
    image: Tensor,
    mask: Tensor,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    color: Tuple[float, float, float] = DEFAULT_MASK_OVERLAY_COLOR,
) -> Tensor:
    """Blend a binary ROI mask over its image as a single solid color."""
    image_rgb = render_for_tensorboard(image).to(mask.device)
    mask_bool = _extract_middle_slice(mask)[:MAX_DISPLAY_SAMPLES] > 0
    color_tensor = torch.tensor(
        color, dtype=image_rgb.dtype, device=image_rgb.device
    ).view(1, 3, 1, 1)
    overlay = torch.where(
        mask_bool,
        (1.0 - alpha) * image_rgb + alpha * color_tensor,
        image_rgb,
    )
    return overlay.detach().cpu()


def add_eval_images(
    writer: SummaryWriter,
    step: int,
    tag: str,
    moving: Tensor,
    fixed: Tensor,
    warped: Tensor,
    moving_seg: Optional[Tensor] = None,
    fixed_seg: Optional[Tensor] = None,
    warped_seg: Optional[Tensor] = None,
    moving_mask: Optional[Tensor] = None,
    fixed_mask: Optional[Tensor] = None,
) -> None:
    """Log each image type under its own tag, as ICON's training loop does."""
    difference = torch.clamp(warped[:, :1] - fixed[:, :1] + DIFFERENCE_OFFSET, 0, 1)
    images = {
        "moving": render_for_tensorboard(moving),
        "fixed": render_for_tensorboard(fixed),
        "warped": render_for_tensorboard(warped),
        "difference": render_for_tensorboard(difference),
    }
    if moving_seg is not None and fixed_seg is not None:
        images["moving_segmentation"] = render_segmentation_overlay_for_tensorboard(moving, moving_seg)
        images["fixed_segmentation"] = render_segmentation_overlay_for_tensorboard(fixed, fixed_seg)
        if warped_seg is not None:
            images["warped_segmentation"] = render_segmentation_overlay_for_tensorboard(warped, warped_seg)
    if moving_mask is not None and fixed_mask is not None:
        images["moving_mask"] = render_mask_overlay_for_tensorboard(moving, moving_mask)
        images["fixed_mask"] = render_mask_overlay_for_tensorboard(fixed, fixed_mask)
    for name, image in images.items():
        writer.add_images(f"{tag}/{name}", image, step, dataformats="NCHW")
