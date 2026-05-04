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

TENSORBOARD_FORMAT = "NCHW"


def _extract_middle_slice(im: Tensor) -> Tensor:
    """Take the middle slice along dim 3.

    For a 5D ``(N, C, D, H, W)`` tensor produced from an ITK volume that
    has been reoriented to RAS, ``D=Z, H=Y, W=X``. Slicing dim 3 (Y) yields
    the middle coronal slice, which is the convention used by the eval
    panels in TensorBoard.
    """
    if len(im.shape) == 5:
        return im[:, :, :, im.shape[3] // 2]
    return im


def _normalize_to_unit(im: Tensor) -> Tensor:
    """Normalize tensor values to the [0, 1] display range."""
    if not torch.is_floating_point(im):
        im = im.float()
    min_value = torch.min(im)
    max_value = torch.max(im)
    span = (max_value - min_value).clamp_min(torch.finfo(im.dtype).eps)
    return (im - min_value) / span


def render_for_tensorboard(
    im: Tensor,
    max_samples: int = MAX_DISPLAY_SAMPLES,
    normalize: bool = True,
) -> Tensor:
    """Prepare image tensor for TensorBoard as an RGB image batch.

    Set ``normalize=False`` for tensors whose absolute values carry meaning
    (e.g. the difference panel, where 0.5 = no residual). Min-max stretching
    those would erase the magnitude signal.
    """
    im = _extract_middle_slice(im)
    if normalize:
        im = _normalize_to_unit(im)
    return im[:max_samples, [0, 0, 0]].detach().cpu()


def segmentation_labels_for_tensorboard(im: Tensor, max_samples: int = MAX_DISPLAY_SAMPLES) -> Tensor:
    """Convert one-hot or label-map segmentations to integer label images."""
    im = _extract_middle_slice(im)
    if im.shape[1] == 1:
        return torch.round(im[:max_samples, 0]).long()
    return torch.argmax(im[:max_samples], dim=1).long()


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
    mask_slice = _extract_middle_slice(mask)[:MAX_DISPLAY_SAMPLES]
    if mask_slice.shape[1] != 1:
        mask_bool = (mask_slice > 0).any(dim=1, keepdim=True)
    else:
        mask_bool = mask_slice > 0
    color_tensor = torch.tensor(
        color, dtype=image_rgb.dtype, device=image_rgb.device
    ).view(1, 3, 1, 1)
    overlay = torch.where(
        mask_bool,
        (1.0 - alpha) * image_rgb + alpha * color_tensor,
        image_rgb,
    )
    return overlay.detach().cpu()


def _add_images(writer: SummaryWriter, tag: str, images: Tensor, step: int) -> None:
    writer.add_images(tag, images, step, dataformats=TENSORBOARD_FORMAT)


def add_eval_composite_panel(
    writer: SummaryWriter,
    step: int,
    moving: Tensor,
    fixed: Tensor,
    warped: Tensor,
    moving_seg: Optional[Tensor] = None,
    fixed_seg: Optional[Tensor] = None,
    warped_seg: Optional[Tensor] = None,
    moving_mask: Optional[Tensor] = None,
    fixed_mask: Optional[Tensor] = None,
    tag: str = "eval",
) -> None:
    """Write a single composite eval panel under ``{tag}`` (default ``"eval"``).

    Each row is a 4-column strip ``[moving | fixed | warped | difference]``.
    A segmentation row (overlays on the image) is appended when seg tensors
    are passed; a mask row (overlay) is appended when mask tensors are passed.
    Empty cells are filled with a black panel so columns stay aligned.
    """
    moving_rgb = render_for_tensorboard(moving)
    fixed_rgb = render_for_tensorboard(fixed)
    warped_rgb = render_for_tensorboard(warped)
    difference = torch.clamp(
        (warped[:MAX_DISPLAY_SAMPLES, :1] - fixed[:MAX_DISPLAY_SAMPLES, :1]) + DIFFERENCE_OFFSET,
        0,
        1,
    )
    diff_rgb = render_for_tensorboard(difference, normalize=False)
    rows = [torch.cat([moving_rgb, fixed_rgb, warped_rgb, diff_rgb], dim=3)]

    if moving_seg is not None and fixed_seg is not None:
        moving_seg_rgb = render_segmentation_overlay_for_tensorboard(moving, moving_seg)
        fixed_seg_rgb = render_segmentation_overlay_for_tensorboard(fixed, fixed_seg)
        if warped_seg is not None:
            warped_seg_rgb = render_segmentation_overlay_for_tensorboard(warped, warped_seg)
        else:
            warped_seg_rgb = torch.zeros_like(moving_seg_rgb)
        blank = torch.zeros_like(moving_seg_rgb)
        rows.append(torch.cat([moving_seg_rgb, fixed_seg_rgb, warped_seg_rgb, blank], dim=3))

    if moving_mask is not None and fixed_mask is not None:
        moving_mask_rgb = render_mask_overlay_for_tensorboard(moving, moving_mask)
        fixed_mask_rgb = render_mask_overlay_for_tensorboard(fixed, fixed_mask)
        # No warped_mask: the network does not propagate masks through the transform.
        blank = torch.zeros_like(moving_mask_rgb)
        rows.append(torch.cat([moving_mask_rgb, fixed_mask_rgb, blank, blank], dim=3))

    composite = torch.cat(rows, dim=2)
    _add_images(writer, tag, composite, step)
