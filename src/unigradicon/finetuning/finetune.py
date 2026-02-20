import os
import sys
import random
import footsteps
from tqdm import tqdm
import torch
import torch.nn.functional as F
import icon_registration as icon
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from icon_registration.losses import to_floats
import unigradicon


def loss_to_dict(loss_object):
    """Convert loss object (ICONLoss or ICONDiceLoss) to dictionary of floats."""
    def tensor_to_float(tensor):
        """Convert tensor to float, handling multi-GPU tensors."""
        if torch.is_tensor(tensor):
            return torch.mean(tensor).item()
        return tensor
    
    if hasattr(loss_object, 'dice_loss'):
        return {
            'all_loss': tensor_to_float(loss_object.all_loss),
            'inverse_consistency_loss': tensor_to_float(loss_object.inverse_consistency_loss),
            'similarity_loss': tensor_to_float(loss_object.similarity_loss),
            'transform_magnitude': tensor_to_float(loss_object.transform_magnitude),
            'flips': tensor_to_float(loss_object.flips),
            'dice_loss': tensor_to_float(loss_object.dice_loss),
        }
    else:
        return to_floats(loss_object)._asdict()


def _random_affine_params(batch_size, device):
    """Create random near-identity affine parameters."""
    identity_list = []
    for _ in range(batch_size):
        identity = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], device=device)
        idxs = set((0, 1, 2))
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1
        identity = identity * (torch.randint_like(identity, 0, 2, device=device) * 2 - 1)
        identity_list.append(identity)

    identity = torch.cat(identity_list)
    noise = torch.randn((batch_size, 3, 4), device=device)
    return identity + 0.05 * noise


def _affine_warp(image, forward, mode='bilinear'):
    grid_shape = list(image.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward, grid_shape, align_corners=True)
    return F.grid_sample(
        image,
        forward_grid,
        mode=mode,
        padding_mode='border',
        align_corners=True,
    )


def augment(image_A, image_B):
    """Apply random affine augmentation to image pairs."""
    device = image_A.device
    forward = _random_affine_params(image_A.shape[0], device)
    if image_A.shape[1] > 1:
        warped_A = _affine_warp(image_A[:, :1], forward)
        warped_A_seg = _affine_warp(image_A[:, 1:], forward, mode='nearest')
        warped_A = torch.cat([warped_A, warped_A_seg], axis=1)
    else:
        warped_A = _affine_warp(image_A, forward)

    forward = _random_affine_params(image_B.shape[0], device)

    if image_B.shape[1] > 1:
        warped_B = _affine_warp(image_B[:, :1], forward)
        warped_B_seg = _affine_warp(image_B[:, 1:], forward, mode='nearest')
        warped_B = torch.cat([warped_B, warped_B_seg], axis=1)
    else:
        warped_B = _affine_warp(image_B, forward)

    return warped_A, warped_B


def augment_with_segmentations(image_A, image_B, seg_A, seg_B):
    """Apply random affine augmentation to image/segmentation pairs."""
    forward_A = _random_affine_params(image_A.shape[0], image_A.device)
    warped_A = _affine_warp(image_A, forward_A)
    warped_seg_A = _affine_warp(seg_A, forward_A, mode='nearest')

    forward_B = _random_affine_params(image_B.shape[0], image_B.device)
    warped_B = _affine_warp(image_B, forward_B)
    warped_seg_B = _affine_warp(seg_B, forward_B, mode='nearest')
    return warped_A, warped_B, warped_seg_A, warped_seg_B


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


def render_warped_segmentation_overlay_for_tensorboard(image, warped_segmentation, alpha=0.55):
    image_rgb = render_for_tensorboard(image).to(warped_segmentation.device)
    labels = segmentation_labels_for_tensorboard(warped_segmentation)
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
            render_warped_segmentation_overlay_for_tensorboard(warped_image, warped_seg),
            epoch,
            dataformats="NCHW",
        )


def get_loss_function(similarity_type, sigma=5, mind_radius=2, mind_dilation=2):
    """Convert similarity type string to loss function object."""
    similarity_type = similarity_type.lower()
    
    if similarity_type == 'lncc':
        return icon.LNCC(sigma=sigma)
    elif similarity_type == 'lncc2':
        return icon.losses.SquaredLNCC(sigma=sigma)
    elif similarity_type == 'mind':
        return icon.losses.MINDSSC(radius=mind_radius, dilation=mind_dilation)
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}. Must be one of: lncc, lncc2, mind")


def finetune_multi_standard(input_shape, data_loader, val_data_loaders_dict, GPUS, device_ids, 
                           epochs, eval_period, save_period, learning_rate, weights_path,
                           lmbda=1.5, loss_fn=None, dice_loss_weight=0.0):
    """
    Finetuning with multiple datasets (standard mode: no segmentations).
    Handles datasets that return 2 outputs: (moving_image, fixed_image).
    
    Args:
        input_shape: Shape of input images
        data_loader: Training DataLoader with weighted sampling
        val_data_loaders_dict: Dict mapping dataset name to validation DataLoader
        GPUS: Number of GPUs
        device_ids: List of GPU device IDs
        epochs: Number of training epochs
        eval_period: Evaluate every N epochs
        save_period: Save checkpoint every N epochs
        learning_rate: Learning rate for optimizer
        weights_path: Path to model weights (automatically detects if resuming based on optimizer file existence)
        lmbda: Regularization weight for deformation smoothness
        loss_fn: Loss function object (e.g., icon.LNCC(sigma=5))
        dice_loss_weight: Weight for dice loss (typically 0.0 for standard mode)
    """
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter
    from icon_registration.losses import to_floats
    
    if loss_fn is None:
        loss_fn = icon.LNCC(sigma=5)
    net = unigradicon.make_network(
        input_shape, 
        include_last_step=True,
        lmbda=lmbda,
        loss_fn=loss_fn,
        use_label=False,
        dice_loss_weight=dice_loss_weight
    )
    
    torch.cuda.set_device(device_ids[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Handle weights path: auto-download if model name is specified
    if weights_path.lower() in ["unigradicon", "multigradicon"]:
        model_name = weights_path.lower()
        weights_path = f"../network_weights/{model_name}1.0/Step_2_final.trch"
        
        if not os.path.exists(weights_path):
            print(f"Downloading pretrained {model_name} model...")
            import urllib.request
            download_url = f"https://github.com/uncbiag/uniGradICON/releases/download/{model_name}_weights/Step_2_final.trch"
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            urllib.request.urlretrieve(download_url, weights_path)
            print(f"Downloaded to: {weights_path}")
    
    print(f"Loading weights from: {weights_path}")
    net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    
    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0]).cuda()
    
    optimizer = torch.optim.Adam(net_par.parameters(), lr=learning_rate)
    
    # Try to load optimizer state if available (for resuming training)
    # The save pattern is: network_weights_{epoch} -> optimizer_weights_{epoch}
    # Or for full paths: .../Step_2_final.trch -> .../optimizer_weights_Step_2_final.trch
    if "network_weights" in weights_path:
        optimizer_path = weights_path.replace("network_weights", "optimizer_weights", 1)
    else:
        weights_dir = os.path.dirname(weights_path)
        weights_filename = os.path.basename(weights_path)
        optimizer_path = os.path.join(weights_dir, "optimizer_weights_" + weights_filename)
    
    if os.path.exists(optimizer_path):
        print(f"Resuming optimizer from: {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
    else:
        print(f"No optimizer state found at {optimizer_path}, starting fresh")
    
    net_par.train()
    
    # Setup tensorboard
    writer = SummaryWriter(
        footsteps.output_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )
    iteration = 0
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for moving_image, fixed_image in data_loader:
            moving_image, fixed_image = moving_image.cuda(), fixed_image.cuda()
            
            with torch.no_grad():
                moving_image, fixed_image = augment(moving_image, fixed_image)
            
            optimizer.zero_grad()
            loss_object = net_par(moving_image, fixed_image)
            loss = torch.mean(loss_object.all_loss)
            loss.backward()
            optimizer.step()
            
            loss_dict = loss_to_dict(loss_object)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v, iteration)
            
            if iteration % 10 == 0:
                loss_str = f"[Epoch {epoch}, Iter {iteration}] "
                loss_str += " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                print(f"\n{loss_str}")
            
            iteration += 1
        
        if epoch % save_period == 0 and epoch > 0:
            torch.save(
                optimizer.state_dict(),
                footsteps.output_dir + f"checkpoints/optimizer_weights_{epoch}",
            )
            torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + f"checkpoints/network_weights_{epoch}",
            )
            print(f"\nCheckpoint saved at epoch {epoch}")
        
        if epoch % eval_period == 0:
            net_par.eval()
            net.eval()
            with torch.no_grad():
                for dataset_name, val_loader in val_data_loaders_dict.items():
                    try:
                        val_moving, val_fixed = next(iter(val_loader))
                        val_moving, val_fixed = val_moving.cuda(), val_fixed.cuda()
                        
                        val_loss = net(val_moving, val_fixed)
                        
                        for k, v in loss_to_dict(val_loss).items():
                            writer.add_scalar(f"{dataset_name}/val_{k}", v, epoch)
                        add_eval_image_panels(
                            writer,
                            dataset_name,
                            epoch,
                            val_moving,
                            val_fixed,
                            net.warped_image_A,
                        )
                        net.clean()
                    except Exception as e:
                        print(f"Warning: Validation failed for {dataset_name}: {e}")
            
            net_par.train()
    
    torch.save(
        net.regis_net.state_dict(),
        footsteps.output_dir + "checkpoints/Finetune_multi_final.trch",
    )
    print("\nTraining completed!")
    writer.close()


def finetune_multi_segmentation(input_shape, data_loader, val_data_loaders_dict, GPUS, device_ids,
                                epochs, eval_period, save_period, learning_rate, weights_path,
                                lmbda=1.5, loss_fn=None, dice_loss_weight=0.0,
                                loss_function_masking=False):
    """
    Finetuning with multiple datasets (segmentation mode).
    Handles datasets that return 4 outputs: (moving_image, fixed_image, moving_seg, fixed_seg).
    
    Args:
        input_shape: Shape of input images
        data_loader: Training DataLoader with weighted sampling
        val_data_loaders_dict: Dict mapping dataset name to validation DataLoader
        GPUS: Number of GPUs
        device_ids: List of GPU device IDs
        epochs: Number of training epochs
        eval_period: Evaluate every N epochs
        save_period: Save checkpoint every N epochs
        learning_rate: Learning rate for optimizer
        weights_path: Path to model weights (automatically detects if resuming based on optimizer file existence)
        lmbda: Regularization weight for deformation smoothness
        loss_fn: Loss function object (e.g., icon.LNCC(sigma=5))
        dice_loss_weight: Weight for dice loss (recommended: 0.3-0.5 for segmentation mode)
        loss_function_masking: Apply segmentation masking to similarity loss.
    """
    if loss_fn is None:
        loss_fn = icon.LNCC(sigma=5)
    
    # Create network with segmentation support
    net = unigradicon.make_network(
        input_shape, 
        include_last_step=True,
        lmbda=lmbda,
        loss_fn=loss_fn,
        use_label=True,
        dice_loss_weight=dice_loss_weight,
        loss_function_masking=loss_function_masking,
    )
    torch.cuda.set_device(device_ids[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Handle weights path: auto-download if model name is specified
    if weights_path.lower() in ["unigradicon", "multigradicon"]:
        model_name = weights_path.lower()
        weights_path = f"../network_weights/{model_name}1.0/Step_2_final.trch"
        
        if not os.path.exists(weights_path):
            print(f"Downloading pretrained {model_name} model...")
            import urllib.request
            download_url = f"https://github.com/uncbiag/uniGradICON/releases/download/{model_name}_weights/Step_2_final.trch"
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            urllib.request.urlretrieve(download_url, weights_path)
            print(f"Downloaded to: {weights_path}")
    
    print(f"Loading weights from: {weights_path}")
    net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    
    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0]).cuda()
    
    optimizer = torch.optim.Adam(net_par.parameters(), lr=learning_rate)
    
    # Try to load optimizer state if available (for resuming training)
    # The save pattern is: network_weights_{epoch} -> optimizer_weights_{epoch}
    # Or for full paths: .../Step_2_final.trch -> .../optimizer_weights_Step_2_final.trch
    if "network_weights" in weights_path:
        optimizer_path = weights_path.replace("network_weights", "optimizer_weights", 1)
    else:
        weights_dir = os.path.dirname(weights_path)
        weights_filename = os.path.basename(weights_path)
        optimizer_path = os.path.join(weights_dir, "optimizer_weights_" + weights_filename)
    
    if os.path.exists(optimizer_path):
        print(f"Resuming optimizer from: {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
    else:
        print(f"No optimizer state found at {optimizer_path}, starting fresh")
    
    net_par.train()
    
    writer = SummaryWriter(
        footsteps.output_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )
    
    print("Starting multi-dataset training (segmentation mode)...")
    iteration = 0
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for moving_image, fixed_image, moving_seg, fixed_seg in data_loader:
            moving_image = moving_image.cuda()
            fixed_image = fixed_image.cuda()
            moving_seg = moving_seg.cuda()
            fixed_seg = fixed_seg.cuda()

            with torch.no_grad():
                moving_image, fixed_image, moving_seg, fixed_seg = augment_with_segmentations(
                    moving_image,
                    fixed_image,
                    moving_seg,
                    fixed_seg,
                )
            
            optimizer.zero_grad()
            loss_object = net_par(moving_image, fixed_image, mask_A=moving_seg, mask_B=fixed_seg)
            loss = torch.mean(loss_object.all_loss)
            loss.backward()
            optimizer.step()
            
            loss_dict = loss_to_dict(loss_object)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v, iteration)
            
            if iteration % 10 == 0:
                loss_str = f"[Epoch {epoch}, Iter {iteration}] "
                loss_str += " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                print(f"\n{loss_str}")
            
            iteration += 1
        
        if epoch % save_period == 0 and epoch > 0:
            torch.save(
                optimizer.state_dict(),
                footsteps.output_dir + f"checkpoints/optimizer_weights_{epoch}",
            )
            torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + f"checkpoints/network_weights_{epoch}",
            )
            print(f"\nCheckpoint saved at epoch {epoch}")
        
        if epoch % eval_period == 0:
            net_par.eval()
            net.eval()
            with torch.no_grad():
                for dataset_name, val_loader in val_data_loaders_dict.items():
                    try:
                        val_moving, val_fixed, val_moving_seg, val_fixed_seg = next(iter(val_loader))
                        val_moving = val_moving.cuda()
                        val_fixed = val_fixed.cuda()
                        val_moving_seg = val_moving_seg.cuda()
                        val_fixed_seg = val_fixed_seg.cuda()
                        
                        val_loss = net(
                            val_moving,
                            val_fixed,
                            mask_A=val_moving_seg,
                            mask_B=val_fixed_seg,
                        )
                        
                        for k, v in loss_to_dict(val_loss).items():
                            writer.add_scalar(f"{dataset_name}/val_{k}", v, epoch)
                        add_eval_image_panels(
                            writer,
                            dataset_name,
                            epoch,
                            val_moving,
                            val_fixed,
                            net.warped_image_A,
                        )
                        warped_seg_for_viz = None
                        if dice_loss_weight > 0.0 and hasattr(net, "warped_seg_A"):
                            warped_seg_for_viz = net.warped_seg_A
                        add_eval_segmentation_panels(
                            writer,
                            dataset_name,
                            epoch,
                            val_moving_seg,
                            val_fixed_seg,
                            warped_seg_for_viz,
                            moving_image=val_moving,
                            fixed_image=val_fixed,
                            warped_image=net.warped_image_A,
                        )
                        net.clean()
                    except Exception as e:
                        print(f"Warning: Validation failed for {dataset_name}: {e}")
            
            net_par.train()
    
    torch.save(
        net.regis_net.state_dict(),
        footsteps.output_dir + "checkpoints/Finetune_multi_final.trch",
    )
    print("\nTraining completed!")
    writer.close()

def main(argv=None):
    import argparse
    from . import multi_dataset_loader

    parser = argparse.ArgumentParser(description="Finetuning for uniGradICON")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args(argv)
    
    train_loader, val_loaders, config, mode = multi_dataset_loader.create_multi_dataset_loaders(args.config)
    
    exp_config = config['experiment']
    train_config = config['training']
    
    footsteps.initialize(run_name=exp_config['name'])
    os.makedirs(footsteps.output_dir + "checkpoints", exist_ok=True)
    
    print(f"\nExperiment: {exp_config['name']}")
    print(f"Mode: {mode}")
    print(f"Training on {len(config['datasets'])} dataset(s)")
    
    input_shape_multi = [1, 1] + train_config['input_shape']
    device_ids_multi = train_config['gpus']
    gpus_multi = len(device_ids_multi)
    
    weights_path = exp_config['weights_path']
    
    epochs = train_config['epochs']
    eval_period = train_config['eval_period']
    save_period = train_config['save_period']
    learning_rate = train_config.get('learning_rate', 0.00005)
    
    similarity_type = train_config.get('similarity', 'lncc')
    lmbda = train_config.get('lambda', 1.5)
    dice_loss_weight = train_config.get('dice_loss_weight', 0.0)
    loss_function_masking = train_config.get('loss_function_masking', False)
    lncc_sigma = train_config.get('lncc_sigma', 5)
    mind_radius = train_config.get('mind_radius', 2)
    mind_dilation = train_config.get('mind_dilation', 2)
    
    loss_fn = get_loss_function(similarity_type, sigma=lncc_sigma, 
                                mind_radius=mind_radius, mind_dilation=mind_dilation)
    
    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  GPUs: {device_ids_multi}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weights: {weights_path}")
    
    print(f"\nLoss function configuration:")
    print(f"  Similarity: {similarity_type}")
    print(f"  Lambda (regularization): {lmbda}")
    print(f"  Dice loss weight: {dice_loss_weight}")
    print(f"  Loss function masking: {loss_function_masking}")
    if similarity_type in ['lncc', 'lncc2']:
        print(f"  LNCC sigma: {lncc_sigma}")
    elif similarity_type == 'mind':
        print(f"  MIND radius: {mind_radius}")
        print(f"  MIND dilation: {mind_dilation}")

    if mode == 'standard' and loss_function_masking:
        raise ValueError(
            "loss_function_masking requires segmentation datasets. "
            "Use dataset type 'unpaired_with_seg' or 'paired_with_seg'."
        )
    if mode == 'standard' and dice_loss_weight > 0.0:
        raise ValueError(
            "dice_loss_weight requires segmentation datasets. "
            "Set dice_loss_weight to 0.0 for standard datasets."
        )
    if loss_function_masking and dice_loss_weight > 0.0:
        raise ValueError(
            "loss_function_masking and dice_loss_weight are mutually exclusive in finetuning. "
            "Set dice_loss_weight to 0.0 when using loss_function_masking."
        )
    
    if mode == 'standard':
        print("\nStarting standard mode training (no segmentations)...")
        finetune_multi_standard(
            input_shape=input_shape_multi,
            data_loader=train_loader,
            val_data_loaders_dict=val_loaders,
            GPUS=gpus_multi,
            device_ids=device_ids_multi,
            epochs=epochs,
            eval_period=eval_period,
            save_period=save_period,
            learning_rate=learning_rate,
            weights_path=weights_path,
            lmbda=lmbda,
            loss_fn=loss_fn,
            dice_loss_weight=dice_loss_weight,
        )
    elif mode == 'segmentation':
        print("\nStarting segmentation mode training (with segmentations)...")
        finetune_multi_segmentation(
            input_shape=input_shape_multi,
            data_loader=train_loader,
            val_data_loaders_dict=val_loaders,
            GPUS=gpus_multi,
            device_ids=device_ids_multi,
            epochs=epochs,
            eval_period=eval_period,
            save_period=save_period,
            learning_rate=learning_rate,
            weights_path=weights_path,
            lmbda=lmbda,
            loss_fn=loss_fn,
            dice_loss_weight=dice_loss_weight,
            loss_function_masking=loss_function_masking,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    print("\n" + "=" * 60)
    print("FINETUNING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
