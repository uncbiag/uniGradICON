import logging
import os
import random
import footsteps
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional
from torch.utils.tensorboard import SummaryWriter
from icon_registration.losses import to_floats
import unigradicon
from icon_registration import config as icon_config
from .config import (
    ConfigSections,
    ExperimentKeys,
    FinetuningConfigSchema,
    TrainingConfig,
    TrainingKeys,
    set_reproducibility_seed,
)
from .dataset import Fields, PairKeys
from .visualization import add_eval_composite_panel

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints"
NETWORK_WEIGHTS_PREFIX = "network_weights"
OPTIMIZER_WEIGHTS_PREFIX = "optimizer_weights"
DEFAULT_LOG_PERIOD = 10


def loss_to_dict(loss_object: Any) -> Dict[str, float]:
    return to_floats(loss_object)._asdict()


def _affine_warp(image: torch.Tensor, forward: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    grid_shape = list(image.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward, grid_shape, align_corners=True)
    return F.grid_sample(
        image,
        forward_grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )


def augment(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply random affine augmentation to all spatial data in a batch dict.

    Images are warped with bilinear interpolation; segmentations and masks
    use nearest interpolation to preserve label values.

    Both images share the same random flip/permutation but have slightly
    different affine noise, so they share orientation but differ in detail.
    """
    device = batch[PairKeys.IMAGE_A].device
    batch_size = batch[PairKeys.IMAGE_A].shape[0]

    identity_list = []
    for _ in range(batch_size):
        identity = torch.zeros((1, 3, 4), dtype=torch.float32, device=device)
        idxs = {0, 1, 2}
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1
        identity = identity * (torch.randint_like(identity, 0, 2) * 2 - 1)
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
        if key.startswith(("image", "label")):
            result[key] = _affine_warp(tensor, forward, mode="bilinear")
        else:
            result[key] = _affine_warp(tensor, forward, mode="nearest")

    return result


def _save_checkpoint(net: Any, optimizer: torch.optim.Optimizer, output_dir: str, epoch: Any) -> None:
    torch.save(
        net.regis_net.state_dict(),
        os.path.join(output_dir, CHECKPOINT_DIR, f"{NETWORK_WEIGHTS_PREFIX}_{epoch}.trch"),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(output_dir, CHECKPOINT_DIR, f"{OPTIMIZER_WEIGHTS_PREFIX}_{epoch}.trch"),
    )


def _load_network_weights(net: Any, model_weights: str, loss_fn: Any, settings: TrainingConfig) -> Optional[str]:
    """Load pretrained model-zoo weights or a custom checkpoint path.

    Returns the resolved weights path so ``_build_optimizer`` can locate the
    matching optimizer state file and resume from it; returns ``None`` when
    model-zoo weights were used (no companion optimizer state to look for).
    """
    # A local file on disk wins over the model-zoo name match so a literal
    # file named ``unigradicon`` (or any case variant) is treated as a path.
    is_zoo_name = (
        not os.path.exists(model_weights)
        and model_weights.lower() in ("unigradicon", "multigradicon")
    )
    if is_zoo_name:
        pretrained_net = unigradicon.get_model_from_model_zoo(
            model_name=model_weights.lower(),
            loss_fn=loss_fn,
            dice_loss_weight=settings.dice_loss_weight,
            loss_function_masking=settings.loss_function_masking,
        )
        # Explicit ``strict=True`` catches architecture drift between
        # ``make_network`` and ``get_model_from_model_zoo``.
        net.regis_net.load_state_dict(pretrained_net.regis_net.state_dict(), strict=True)
        return None

    weights_path = os.path.abspath(model_weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {model_weights} (resolved to {weights_path})")
    logger.info(f"Loading network weights from {weights_path}.")
    net.regis_net.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    return weights_path


def _optimizer_state_path(model_weights: str) -> str:
    weights_filename = os.path.basename(model_weights)
    weights_dir = os.path.dirname(model_weights)
    if weights_filename.startswith(NETWORK_WEIGHTS_PREFIX):
        optimizer_filename = weights_filename.replace(NETWORK_WEIGHTS_PREFIX, OPTIMIZER_WEIGHTS_PREFIX, 1)
    else:
        optimizer_filename = f"{OPTIMIZER_WEIGHTS_PREFIX}_{weights_filename}"
    return os.path.join(weights_dir, optimizer_filename)


def _build_optimizer(
    net: Any,
    learning_rate: float,
    model_weights: Optional[str],
) -> torch.optim.Optimizer:
    """Build the Adam optimizer over the registration network parameters.

    Uses ``net.regis_net.parameters()`` so that the optimizer is bound to the
    same module whose ``state_dict`` is saved/loaded for checkpoints. This keeps
    parameter ordering stable regardless of whether the network is wrapped in
    DataParallel, which makes single-GPU ↔ multi-GPU resumes safe.
    """
    optimizer = torch.optim.Adam(net.regis_net.parameters(), lr=learning_rate)
    if model_weights is None:
        logger.info("Initializing optimizer from scratch (model-zoo weights have no companion optimizer state).")
        return optimizer

    optimizer_path = _optimizer_state_path(model_weights)
    if os.path.exists(optimizer_path):
        logger.info(f"Resuming optimizer state from {optimizer_path}.")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
    else:
        logger.info(f"Optimizer state not found at {optimizer_path}; initializing optimizer from scratch.")
    return optimizer


def _build_forward_kwargs(
    batch: Dict[str, torch.Tensor],
    settings: TrainingConfig,
    data_fields: FrozenSet[str],
) -> Dict[str, torch.Tensor]:
    forward_kwargs = {}
    if settings.use_label and PairKeys.LABEL_A in batch:
        forward_kwargs[PairKeys.LABEL_A] = batch[PairKeys.LABEL_A]
        forward_kwargs[PairKeys.LABEL_B] = batch[PairKeys.LABEL_B]
    if settings.dice_loss_weight > 0.0 and Fields.SEGMENTATION in data_fields:
        forward_kwargs[PairKeys.SEGMENTATION_A] = batch[PairKeys.SEGMENTATION_A]
        forward_kwargs[PairKeys.SEGMENTATION_B] = batch[PairKeys.SEGMENTATION_B]
    if settings.loss_function_masking and Fields.MASK in data_fields:
        forward_kwargs[PairKeys.MASK_A] = batch[PairKeys.MASK_A]
        forward_kwargs[PairKeys.MASK_B] = batch[PairKeys.MASK_B]
    return forward_kwargs


def _log_loss_scalars(writer: SummaryWriter, prefix: str, loss_dict: Dict[str, float], iteration: int) -> None:
    for key, value in loss_dict.items():
        writer.add_scalar(f"{prefix}/{key}", value, iteration)


def _apply_roi_masking(batch: Dict[str, torch.Tensor]) -> None:
    """Zero out background under the ROI mask. Applied to both training and
    validation batches so the network sees the same input distribution in
    both phases."""
    batch[PairKeys.IMAGE_A] = batch[PairKeys.IMAGE_A] * (batch[PairKeys.MASK_A] > 0).float()
    batch[PairKeys.IMAGE_B] = batch[PairKeys.IMAGE_B] * (batch[PairKeys.MASK_B] > 0).float()


def _train_one_batch(
    net: Any,
    net_par: Any,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    settings: TrainingConfig,
    data_fields: FrozenSet[str],
    device: torch.device,
) -> Dict[str, float]:
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        batch = augment(batch)

    if settings.roi_masking:
        _apply_roi_masking(batch)

    optimizer.zero_grad()
    forward_kwargs = _build_forward_kwargs(batch, settings, data_fields)
    loss_object = net_par(batch[PairKeys.IMAGE_A], batch[PairKeys.IMAGE_B], **forward_kwargs)
    loss = torch.mean(loss_object.all_loss)
    loss.backward()
    optimizer.step()
    net.clean()
    return loss_to_dict(loss_object)


def _run_validation(
    net: Any,
    net_par: Any,
    val_data_loaders_dict: Dict[str, Any],
    writer: SummaryWriter,
    iteration: int,
    settings: TrainingConfig,
    data_fields: FrozenSet[str],
    device: torch.device,
) -> None:
    if not val_data_loaders_dict:
        return

    ds_name, val_loader = random.choice(list(val_data_loaders_dict.items()))

    net_par.eval()
    net.eval()
    try:
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_batch = {key: value.to(device) for key, value in val_batch.items()}
            if settings.roi_masking:
                _apply_roi_masking(val_batch)
            forward_kwargs = _build_forward_kwargs(val_batch, settings, data_fields)
            val_loss = net(val_batch[PairKeys.IMAGE_A], val_batch[PairKeys.IMAGE_B], **forward_kwargs)
            _log_loss_scalars(writer, f"val/{ds_name}", loss_to_dict(val_loss), iteration)
            _write_eval_visualizations(
                writer,
                iteration,
                val_batch,
                net,
                settings,
                data_fields,
                ds_name,
            )
            net.clean()
    finally:
        net_par.train()


def _write_eval_visualizations(
    writer: SummaryWriter,
    iteration: int,
    val_batch: Dict[str, torch.Tensor],
    net: Any,
    settings: TrainingConfig,
    data_fields: FrozenSet[str],
    ds_name: str,
) -> None:
    has_seg = Fields.SEGMENTATION in data_fields
    has_mask = Fields.MASK in data_fields

    warped_seg = None
    if has_seg and settings.dice_loss_weight > 0.0 and hasattr(net, "warped_seg_A"):
        warped_seg = net.warped_seg_A

    add_eval_composite_panel(
        writer,
        iteration,
        moving=val_batch[PairKeys.IMAGE_A],
        fixed=val_batch[PairKeys.IMAGE_B],
        warped=net.warped_image_A,
        moving_seg=val_batch[PairKeys.SEGMENTATION_A] if has_seg else None,
        fixed_seg=val_batch[PairKeys.SEGMENTATION_B] if has_seg else None,
        warped_seg=warped_seg,
        moving_mask=val_batch[PairKeys.MASK_A] if has_mask else None,
        fixed_mask=val_batch[PairKeys.MASK_B] if has_mask else None,
        tag=f"eval/{ds_name}",
    )


def finetune_multi(
    config: Dict[str, Any],
    data_loader: Any,
    val_data_loaders_dict: Dict[str, Any],
    data_fields: FrozenSet[str],
) -> None:
    """
    Unified finetuning loop.

    The training step uses ``net_par`` (the DataParallel-wrapped network) so
    forward/backward can scatter across GPUs. Validation, checkpoint saving,
    and the optimizer use the unwrapped ``net``/``net.regis_net`` so behavior
    is independent of the parallel wrapping.

    Args:
        config: Full configuration dict with 'experiment' and 'training' sections.
        data_loader: Training DataLoader with weighted sampling.
        val_data_loaders_dict: Dict mapping dataset name to validation DataLoader.
        data_fields: Frozenset of auxiliary data fields (e.g. {"segmentation", "mask"}).
    """
    schema = FinetuningConfigSchema.from_dict(config)
    settings = schema.training
    exp_config = config[ConfigSections.EXPERIMENT]
    device = icon_config.device

    if device.type == "cuda":
        available = torch.cuda.device_count()
        invalid = [g for g in settings.gpus if g < 0 or g >= available]
        if invalid:
            raise ValueError(
                f"Requested GPU(s) {invalid} not available; "
                f"torch.cuda.device_count() reports {available} GPU(s)."
            )
        torch.cuda.set_device(settings.gpus[0])
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    loss_fn = unigradicon.make_sim(
        settings.similarity.lower(),
        sigma=settings.lncc_sigma,
        mind_radius=settings.mind_radius,
        mind_dilation=settings.mind_dilation,
    )
    net = unigradicon.make_network(
        settings.network_input_shape,
        include_last_step=True,
        lmbda=settings.lmbda,
        loss_fn=loss_fn,
        use_label=settings.use_label,
        dice_loss_weight=settings.dice_loss_weight,
        loss_function_masking=settings.loss_function_masking,
    )
    model_weights = _load_network_weights(net, exp_config[ExperimentKeys.MODEL_WEIGHTS], loss_fn, settings)

    if device.type == "cuda" and len(settings.gpus) > 1:
        net_par = torch.nn.DataParallel(
            net, device_ids=settings.gpus, output_device=settings.gpus[0]
        ).to(device)
    else:
        net_par = net.to(device)

    optimizer = _build_optimizer(net, settings.learning_rate, model_weights)
    net_par.train()
    os.makedirs(os.path.join(footsteps.output_dir, CHECKPOINT_DIR), exist_ok=True)
    writer = SummaryWriter(
        os.path.join(footsteps.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")),
        flush_secs=30,
    )

    logger.info("Starting training loop.")
    logger.info(f"Auxiliary data fields: {sorted(data_fields) if data_fields else 'images only'}.")
    logger.info(
        f"Schedule: epochs={settings.epochs}, learning_rate={settings.learning_rate}, "
        f"gpus={settings.gpus}, eval_period={settings.eval_period}, save_period={settings.save_period}."
    )
    logger.info(
        f"Loss configuration: similarity={settings.similarity}, lambda={settings.lmbda}, "
        f"dice_loss_weight={settings.dice_loss_weight}, "
        f"loss_function_masking={settings.loss_function_masking}, "
        f"roi_masking={settings.roi_masking}, use_label={settings.use_label}."
    )

    iteration = 0

    try:
        for epoch in tqdm(range(settings.epochs), desc="Training epochs"):
            for batch in data_loader:
                loss_dict = _train_one_batch(
                    net,
                    net_par,
                    optimizer,
                    batch,
                    settings,
                    data_fields,
                    device,
                )
                _log_loss_scalars(writer, "train", loss_dict, iteration)

                if iteration % DEFAULT_LOG_PERIOD == 0:
                    loss_str = " | ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                    logger.info(f"[epoch {epoch}, iteration {iteration}] {loss_str}")

                iteration += 1

            is_last_epoch = (epoch == settings.epochs - 1)
            is_periodic_save = (epoch > 0 and epoch % settings.save_period == 0)
            if is_periodic_save and not is_last_epoch:
                _save_checkpoint(net, optimizer, footsteps.output_dir, epoch)
                logger.info(f"Wrote checkpoint for epoch {epoch}.")

            if epoch % settings.eval_period == 0:
                _run_validation(
                    net,
                    net_par,
                    val_data_loaders_dict,
                    writer,
                    iteration,
                    settings,
                    data_fields,
                    device,
                )

        _save_checkpoint(net, optimizer, footsteps.output_dir, "final")
        logger.info("Training loop completed; final checkpoint written.")
    finally:
        writer.close()


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    from .config import load_config, create_data_loaders, validate_config

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Finetuning for uniGradICON")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_config(config)
    exp_config = config[ConfigSections.EXPERIMENT]

    seed = config.get(ConfigSections.TRAINING, {}).get(TrainingKeys.SEED)
    set_reproducibility_seed(seed)
    if seed is not None:
        logger.info(f"Reproducibility seed set: {seed}.")

    os.makedirs("results", exist_ok=True)
    footsteps.initialize(run_name=exp_config[ExperimentKeys.NAME])

    loaders = create_data_loaders(args.config, config=config)
    config = loaders.config

    logger.info(
        f"Experiment '{exp_config[ExperimentKeys.NAME]}' ready: "
        f"{len(config[ConfigSections.DATASETS])} dataset(s), "
        f"auxiliary fields={sorted(loaders.data_fields) if loaders.data_fields else 'images only'}."
    )

    finetune_multi(
        config=config,
        data_loader=loaders.train_loader,
        val_data_loaders_dict=loaders.val_loaders,
        data_fields=loaders.data_fields,
    )

    logger.info("Finetuning run completed successfully.")


if __name__ == "__main__":
    main()
