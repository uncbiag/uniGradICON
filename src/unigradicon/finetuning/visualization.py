import torch


def render_for_tensorboard(im):
    if len(im.shape) == 5:
        im = im[:, :, :, im.shape[3] // 2]
    if torch.min(im) < 0:
        im = im - torch.min(im)
    if torch.max(im) > 1:
        im = im / torch.max(im)
    return im[:4, [0, 0, 0]].detach().cpu()


def segmentation_labels_for_tensorboard(im):
    if len(im.shape) == 5:
        im = im[:, :, :, im.shape[3] // 2]
    if im.shape[1] == 1:
        return torch.round(im[:4, 0]).long()
    return torch.argmax(im[:4], dim=1).long()


def _hsv_to_rgb(h, s, v):
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


def _segmentation_palette(num_colors, device):
    palette = torch.zeros((max(num_colors, 1), 3), dtype=torch.float32, device=device)
    if num_colors <= 1:
        return palette

    class_ids = torch.arange(1, num_colors, dtype=torch.float32, device=device)
    hues = torch.remainder(class_ids * 0.61803398875, 1.0)
    saturation = torch.full_like(hues, 0.75)
    value = torch.full_like(hues, 0.95)
    palette[1:] = _hsv_to_rgb(hues, saturation, value)
    return palette


def labels_to_color_image(labels):
    num_colors = int(labels.max().item()) + 1 if labels.numel() > 0 else 1
    palette = _segmentation_palette(num_colors, labels.device)
    color_idx = torch.remainder(labels, palette.shape[0])
    rgb = palette[color_idx]
    return rgb.permute(0, 3, 1, 2)


def render_segmentation_for_tensorboard(im):
    labels = segmentation_labels_for_tensorboard(im)
    return labels_to_color_image(labels).detach().cpu()


def render_segmentation_overlay_for_tensorboard(image, segmentation, alpha=0.55):
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


def add_eval_image_panels(writer, prefix, epoch, moving, fixed, warped):
    writer.add_images(
        f"{prefix}/moving_image", render_for_tensorboard(moving[:4]), epoch, dataformats="NCHW"
    )
    writer.add_images(
        f"{prefix}/fixed_image", render_for_tensorboard(fixed[:4]), epoch, dataformats="NCHW"
    )
    writer.add_images(
        f"{prefix}/warped_moving_image",
        render_for_tensorboard(warped),
        epoch,
        dataformats="NCHW",
    )
    writer.add_images(
        f"{prefix}/difference",
        render_for_tensorboard(torch.clip((warped[:4, :1] - fixed[:4, :1]) + 0.5, 0, 1)),
        epoch,
        dataformats="NCHW",
    )


def add_eval_segmentation_panels(
    writer,
    prefix,
    epoch,
    moving_seg,
    fixed_seg,
    warped_seg=None,
    moving_image=None,
    fixed_image=None,
    warped_image=None,
):
    writer.add_images(
        f"{prefix}/moving_segmentation",
        render_segmentation_for_tensorboard(moving_seg),
        epoch,
        dataformats="NCHW",
    )
    writer.add_images(
        f"{prefix}/fixed_segmentation",
        render_segmentation_for_tensorboard(fixed_seg),
        epoch,
        dataformats="NCHW",
    )
    if moving_image is not None:
        writer.add_images(
            f"{prefix}/moving_overlay",
            render_segmentation_overlay_for_tensorboard(moving_image, moving_seg),
            epoch,
            dataformats="NCHW",
        )
    if fixed_image is not None:
        writer.add_images(
            f"{prefix}/fixed_overlay",
            render_segmentation_overlay_for_tensorboard(fixed_image, fixed_seg),
            epoch,
            dataformats="NCHW",
        )
    if warped_seg is not None:
        writer.add_images(
            f"{prefix}/warped_moving_segmentation",
            render_segmentation_for_tensorboard(warped_seg),
            epoch,
            dataformats="NCHW",
        )
    if warped_seg is not None and warped_image is not None:
        writer.add_images(
            f"{prefix}/warped_moving_overlay",
            render_segmentation_overlay_for_tensorboard(warped_image, warped_seg),
            epoch,
            dataformats="NCHW",
        )
