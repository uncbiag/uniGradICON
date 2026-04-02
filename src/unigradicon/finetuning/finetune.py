import logging
import os
import random
import traceback
import footsteps
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import icon_registration as icon
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from icon_registration.losses import to_floats
import unigradicon
from icon_registration import config as icon_config
from .visualization import add_eval_image_panels, add_eval_segmentation_panels

logger = logging.getLogger(__name__)


def loss_to_dict(loss_object):
    """Convert loss object (ICONLoss or ICONDiceLoss) to dictionary of floats."""
    def tensor_to_float(tensor):
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


def augment(batch):
    """Apply random affine augmentation to all spatial data in a batch dict.

    Images are warped with bilinear interpolation; segmentations and masks
    use nearest interpolation to preserve label values.

    Both images share the same random flip/permutation but have slightly
    different affine noise, so they share orientation but differ in detail.
    """
    device = batch["image_A"].device
    batch_size = batch["image_A"].shape[0]

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

    noise_A = torch.randn((batch_size, 3, 4), device=device)
    forward_A = identity + 0.05 * noise_A
    noise_B = torch.randn((batch_size, 3, 4), device=device)
    forward_B = identity + 0.05 * noise_B

    result = {}
    for key, tensor in batch.items():
        if not torch.is_tensor(tensor):
            result[key] = tensor
            continue
        forward = forward_A if key.endswith("_A") else forward_B
        if key.startswith("image") or key.startswith("label"):
            result[key] = _affine_warp(tensor, forward, mode='bilinear')
        else:
            result[key] = _affine_warp(tensor, forward, mode='nearest')

    return result


def get_loss_function(similarity_type, sigma=5, mind_radius=2, mind_dilation=2):
    """Convert similarity type string to loss function object.

    Delegates to ``unigradicon.make_sim()`` with configurable parameters.
    """
    return unigradicon.make_sim(similarity_type.lower(), sigma=sigma,
                                mind_radius=mind_radius, mind_dilation=mind_dilation)


def _save_checkpoint(net, optimizer, output_dir, epoch):
    """Save network and optimizer checkpoint for a given epoch."""
    torch.save(
        net.regis_net.state_dict(),
        os.path.join(output_dir, "checkpoints", f"network_weights_{epoch}.trch"),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(output_dir, "checkpoints", f"optimizer_weights_{epoch}.trch"),
    )


def _resolve_model_weights(model_weights):
    """Resolve model weights to an absolute path, downloading pretrained weights if needed.

    Accepts ``"unigradicon"``, ``"multigradicon"`` (auto-downloads), or a file path.
    Downloads use atomic rename to avoid corrupted partial files on failure.
    """
    if model_weights.lower() in ("unigradicon", "multigradicon"):
        model_name = model_weights.lower()
        weights_path = os.path.abspath(
            os.path.join("network_weights", f"{model_name}1.0", "Step_2_final.trch")
        )

        if not os.path.exists(weights_path):
            logger.info(f"Downloading pretrained {model_name} model...")
            import urllib.request
            download_url = f"https://github.com/uncbiag/uniGradICON/releases/download/{model_name}_weights/Step_2_final.trch"
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            tmp_path = weights_path + ".download"
            try:
                urllib.request.urlretrieve(download_url, tmp_path)
                os.rename(tmp_path, weights_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            logger.info(f"Downloaded to: {weights_path}")
        return weights_path

    weights_path = os.path.abspath(model_weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {model_weights} (resolved to {weights_path})")
    return weights_path


def finetune_multi(config, data_loader, val_data_loaders_dict, data_fields):
    """
    Unified finetuning loop.

    Args:
        config: Full configuration dict with 'experiment' and 'training' sections.
        data_loader: Training DataLoader with weighted sampling.
        val_data_loaders_dict: Dict mapping dataset name to validation DataLoader.
        data_fields: Frozenset of optional data fields (e.g. {"segmentation", "mask"}).
    """
    train_config = config['training']
    exp_config = config['experiment']

    input_shape = [1, 1] + train_config['input_shape']
    device_ids = train_config['gpus']
    num_gpus = len(device_ids)
    epochs = train_config['epochs']
    eval_period = train_config['eval_period']
    save_period = train_config['save_period']
    learning_rate = train_config.get('learning_rate', 0.00005)
    lmbda = train_config.get('lambda', 1.5)
    dice_loss_weight = train_config.get('dice_loss_weight', 0.0)
    loss_function_masking = train_config.get('loss_function_masking', False)
    roi_masking = train_config.get('roi_masking', False)
    use_label = train_config.get('use_label', False)

    similarity_type = train_config.get('similarity', 'lncc')
    loss_fn = get_loss_function(
        similarity_type,
        sigma=train_config.get('lncc_sigma', 5),
        mind_radius=train_config.get('mind_radius', 2),
        mind_dilation=train_config.get('mind_dilation', 2),
    )

    has_segmentation = "segmentation" in data_fields
    has_mask = "mask" in data_fields

    net = unigradicon.make_network(
        input_shape,
        include_last_step=True,
        lmbda=lmbda,
        loss_fn=loss_fn,
        use_label=use_label,
        dice_loss_weight=dice_loss_weight,
        loss_function_masking=loss_function_masking,
    )

    device = icon_config.device
    if device.type == "cuda":
        torch.cuda.set_device(device_ids[0])
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model_weights = _resolve_model_weights(exp_config['model_weights'])

    logger.info(f"Loading weights from: {model_weights}")
    net.regis_net.load_state_dict(torch.load(model_weights, map_location="cpu", weights_only=True))

    if device.type == "cuda" and num_gpus > 1:
        net_par = torch.nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0]).to(device)
    else:
        net_par = net.to(device)

    optimizer = torch.optim.Adam(net_par.parameters(), lr=learning_rate)

    weights_filename = os.path.basename(model_weights)
    weights_dir = os.path.dirname(model_weights)
    if weights_filename.startswith("network_weights"):
        optimizer_filename = weights_filename.replace("network_weights", "optimizer_weights", 1)
    else:
        optimizer_filename = "optimizer_weights_" + weights_filename
    optimizer_path = os.path.join(weights_dir, optimizer_filename)

    if os.path.exists(optimizer_path):
        logger.info(f"Resuming optimizer from: {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
    else:
        logger.info(f"No optimizer state found at {optimizer_path}, starting fresh")

    net_par.train()

    os.makedirs(os.path.join(footsteps.output_dir, "checkpoints"), exist_ok=True)

    writer = SummaryWriter(
        os.path.join(footsteps.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")),
        flush_secs=30,
    )

    logger.info(f"Starting training...")
    logger.info(f"Data fields: {data_fields or 'images only'}")
    logger.info(f"Training: epochs={epochs}, lr={learning_rate}, gpus={device_ids}")
    logger.info(f"Loss: similarity={similarity_type}, lambda={lmbda}, "
                f"dice_weight={dice_loss_weight}, masking={loss_function_masking}, "
                f"roi_masking={roi_masking}, use_label={use_label}")

    iteration = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                batch = augment(batch)

            if roi_masking:
                batch["image_A"] = batch["image_A"] * (batch["mask_A"] > 0).float()
                batch["image_B"] = batch["image_B"] * (batch["mask_B"] > 0).float()

            optimizer.zero_grad()

            forward_kwargs = {}
            if use_label and 'label_A' in batch:
                forward_kwargs['label_A'] = batch['label_A']
                forward_kwargs['label_B'] = batch['label_B']
            if dice_loss_weight > 0.0 and has_segmentation:
                forward_kwargs['segmentation_A'] = batch['segmentation_A']
                forward_kwargs['segmentation_B'] = batch['segmentation_B']
            if loss_function_masking and has_mask:
                forward_kwargs['mask_A'] = batch['mask_A']
                forward_kwargs['mask_B'] = batch['mask_B']

            loss_object = net_par(batch['image_A'], batch['image_B'], **forward_kwargs)
            loss = torch.mean(loss_object.all_loss)
            loss.backward()
            optimizer.step()

            net.clean()

            loss_dict = loss_to_dict(loss_object)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v, iteration)

            if iteration % 10 == 0:
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                logger.info(f"[Epoch {epoch}, Iter {iteration}] {loss_str}")

            iteration += 1

        is_last_epoch = (epoch == epochs - 1)
        if epoch > 0 and epoch % save_period == 0 and not is_last_epoch:
            _save_checkpoint(net, optimizer, footsteps.output_dir, epoch)
            logger.info(f"Checkpoint saved at epoch {epoch}")

        if epoch % eval_period == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            net_par.eval()
            net.eval()
            with torch.no_grad():
                for dataset_name, val_loader in val_data_loaders_dict.items():
                    try:
                        val_batch = next(iter(val_loader))
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}

                        forward_kwargs = {}
                        if use_label and 'label_A' in val_batch:
                            forward_kwargs['label_A'] = val_batch['label_A']
                            forward_kwargs['label_B'] = val_batch['label_B']
                        if dice_loss_weight > 0.0 and has_segmentation:
                            forward_kwargs['segmentation_A'] = val_batch['segmentation_A']
                            forward_kwargs['segmentation_B'] = val_batch['segmentation_B']
                        if loss_function_masking and has_mask:
                            forward_kwargs['mask_A'] = val_batch['mask_A']
                            forward_kwargs['mask_B'] = val_batch['mask_B']

                        val_loss = net(val_batch['image_A'], val_batch['image_B'], **forward_kwargs)

                        for k, v in loss_to_dict(val_loss).items():
                            writer.add_scalar(f"val/{dataset_name}/{k}", v, iteration)
                        add_eval_image_panels(
                            writer,
                            dataset_name,
                            iteration,
                            val_batch['image_A'],
                            val_batch['image_B'],
                            net.warped_image_A,
                        )

                        if has_segmentation:
                            warped_seg_for_viz = None
                            if dice_loss_weight > 0.0 and hasattr(net, "warped_seg_A"):
                                warped_seg_for_viz = net.warped_seg_A
                            add_eval_segmentation_panels(
                                writer,
                                dataset_name,
                                iteration,
                                val_batch['segmentation_A'],
                                val_batch['segmentation_B'],
                                warped_seg_for_viz,
                                moving_image=val_batch['image_A'],
                                fixed_image=val_batch['image_B'],
                                warped_image=net.warped_image_A,
                            )

                        net.clean()
                    except Exception:
                        logger.warning(f"Validation failed for {dataset_name}:\n{traceback.format_exc()}")

            net_par.train()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _save_checkpoint(net, optimizer, footsteps.output_dir, "final")
    logger.info("Training completed!")
    writer.close()


def main(argv=None):
    import argparse
    from .config import load_config, create_data_loaders, validate_config

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Finetuning for uniGradICON")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_config(config)
    exp_config = config['experiment']

    seed = config.get('training', {}).get('seed')
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")

    os.makedirs("results", exist_ok=True)
    footsteps.initialize(run_name=exp_config['name'])

    train_loader, val_loaders, config, data_fields = create_data_loaders(args.config, config=config)

    logger.info(f"Experiment: {exp_config['name']} | Data fields: {data_fields or 'images only'} | Datasets: {len(config['datasets'])}")

    finetune_multi(
        config=config,
        data_loader=train_loader,
        val_data_loaders_dict=val_loaders,
        data_fields=data_fields,
    )

    logger.info("=" * 40 + " FINETUNING COMPLETED " + "=" * 40)


if __name__ == "__main__":
    main()
