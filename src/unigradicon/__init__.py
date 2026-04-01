import itk
import os

import footsteps
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple

import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration import config
from icon_registration.mermaidlite import compute_warped_image_multiNC
import icon_registration.itk_wrapper

# Extended loss object that includes dice_loss for segmentation-based training
ICONDiceLoss = namedtuple('ICONDiceLoss', ['all_loss', 'inverse_consistency_loss', 'similarity_loss', 'transform_magnitude', 'flips', 'dice_loss'])

input_shape = [1, 1, 175, 175, 175]

class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda, use_label=False, apply_intensity_conservation_loss=False, dice_loss_weight=0.0, loss_function_masking=False):
        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.dice_loss_weight = dice_loss_weight
        self.similarity = similarity
        self.use_label = use_label
        self.apply_intensity_conservation_loss = apply_intensity_conservation_loss
        self.loss_function_masking = loss_function_masking

    def forward(self, image_A, image_B, label_A=None, label_B=None, mask_A=None, mask_B=None, segmentation_A=None, segmentation_B=None):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        if self.use_label:
            label_A = image_A if label_A is None else label_A
            label_B = image_B if label_B is None else label_B
            assert self.identity_map.shape[2:] == label_A.shape[2:]
            assert self.identity_map.shape[2:] == label_B.shape[2:]

        if self.loss_function_masking:
            assert mask_A is not None and mask_B is not None, \
                "mask_A and mask_B must be provided when loss_function_masking=True"
        if self.dice_loss_weight > 0.0:
            assert segmentation_A is not None and segmentation_B is not None, \
                "segmentation_A and segmentation_B must be provided when dice_loss_weight>0"
            unique_A = torch.unique(segmentation_A.long())
            unique_B = torch.unique(segmentation_B.long())
            common_classes = unique_A[torch.isin(unique_A, unique_B)]
            common_classes = common_classes[common_classes != 0]  # exclude background
            num_classes = len(common_classes)

            if num_classes == 0:
                seg_A_one_hot = seg_B_one_hot = None
            else:
                max_class_id = int(torch.max(unique_A.max(), unique_B.max()).item())
                remap = torch.zeros(max_class_id + 1, dtype=torch.long, device=segmentation_A.device)
                remap[common_classes] = torch.arange(1, num_classes + 1, device=segmentation_A.device)

                seg_A_one_hot = F.one_hot(remap[segmentation_A.long()], num_classes=num_classes + 1)[:,0].permute(0, 4, 1, 2, 3)[:, 1:].float()
                seg_B_one_hot = F.one_hot(remap[segmentation_B.long()], num_classes=num_classes + 1)[:,0].permute(0, 4, 1, 2, 3)[:, 1:].float()

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        if self.apply_intensity_conservation_loss:
            if len(self.input_shape) - 2 == 3:
                jacobian_slice = np.index_exp[:, :, :-1, :-1, :-1]
            elif len(self.input_shape) - 2 == 2:
                jacobian_slice = np.index_exp[:, :, :-1, :-1]
            else:
                jacobian_slice = np.index_exp[:, :, :-1]
                
            jacobian_AB = self.compute_jacobian_determinant(self.phi_AB_vectorfield)
            jacobian_BA = self.compute_jacobian_determinant(self.phi_BA_vectorfield)
            
        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], dim=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], dim=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )

        if self.dice_loss_weight > 0.0 and seg_A_one_hot is not None:
            self.warped_seg_A = compute_warped_image_multiNC(
                torch.cat([seg_A_one_hot, inbounds_tag], dim=1) if inbounds_tag is not None else seg_A_one_hot.float(),
                self.phi_AB_vectorfield,
                self.spacing,
                1,
            )

            self.warped_seg_B = compute_warped_image_multiNC(
                torch.cat([seg_B_one_hot, inbounds_tag], dim=1) if inbounds_tag is not None else seg_B_one_hot.float(),
                self.phi_BA_vectorfield,
                self.spacing,
                1,
            )
            dice_loss = self.dice_loss(self.warped_seg_A, seg_B_one_hot) + self.dice_loss(self.warped_seg_B, seg_A_one_hot)
        else:
            dice_loss = 0.0
            
        if self.use_label:
            self.warped_label_A = compute_warped_image_multiNC(
                torch.cat([label_A, inbounds_tag], dim=1) if inbounds_tag is not None else label_A,
                self.phi_AB_vectorfield,
                self.spacing,
                1,
            )
            
            self.warped_label_B = compute_warped_image_multiNC(
                torch.cat([label_B, inbounds_tag], dim=1) if inbounds_tag is not None else label_B,
                self.phi_BA_vectorfield,
                self.spacing,
                1,
            )
            
            self.warped_loss_input_A = self.warped_label_A
            self.warped_loss_input_B = self.warped_label_B
        else:
            self.warped_loss_input_A = self.warped_image_A
            self.warped_loss_input_B = self.warped_image_B
            
              
        if self.apply_intensity_conservation_loss:
            self.warped_loss_input_A_jacob = self.warped_loss_input_A[jacobian_slice] * jacobian_AB
            self.warped_loss_input_B_jacob = self.warped_loss_input_B[jacobian_slice] * jacobian_BA
            similarity_loss = self.similarity(
                self.warped_loss_input_B_jacob, 
                image_A[jacobian_slice], 
                mask_A[jacobian_slice] if mask_A is not None else None
            ) + self.similarity(
                self.warped_loss_input_A_jacob, 
                image_B[jacobian_slice], 
                mask_B[jacobian_slice] if mask_B is not None else None
            )
        else:
            if self.loss_function_masking:
                similarity_loss = self.similarity(self.warped_loss_input_A, image_B, mask_B) + self.similarity(self.warped_loss_input_B, image_A, mask_A)
            else:
                similarity_loss = self.similarity(self.warped_loss_input_A, image_B) + self.similarity(self.warped_loss_input_B, image_A)

        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.tensor([[[[[0.0]]], [[[0.0]]], [[[delta]]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss + dice_loss * self.dice_loss_weight

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        
        if self.dice_loss_weight > 0.0:
            return ICONDiceLoss(
                all_loss,
                inverse_consistency_loss,
                similarity_loss,
                transform_magnitude,
                icon.losses.flips(self.phi_BA_vectorfield),
                dice_loss if isinstance(dice_loss, torch.Tensor) else torch.tensor(dice_loss),
            )
        else:
            return icon.losses.ICONLoss(
                all_loss,
                inverse_consistency_loss,
                similarity_loss,
                transform_magnitude,
                icon.losses.flips(self.phi_BA_vectorfield),
            )

    def compute_jacobian_determinant(self, phi):
        if len(phi.size()) == 4:
            du = phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]
            dv = phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]
            dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
            dA = dA[:, None, :]
            return (
                dA * (self.identity_map.shape[2] - 1) * (self.identity_map.shape[3] - 1)
            )
        if len(phi.size()) == 5:
            a = phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]
            b = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]
            c = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]

            dV = torch.sum(torch.cross(a, b, 1) * c, dim=1, keepdim=True)
            dV = (
                dV
                * (self.identity_map.shape[2] - 1)
                * (self.identity_map.shape[3] - 1)
                * (self.identity_map.shape[4] - 1)
            )
            return dV
        
    def dice_loss(self, pred, target, epsilon=1e-6):
        """
        Compute Dice loss between one-hot encoded prediction and target.
        Args:
            pred (Tensor): One-hot encoded prediction of shape (N, C, H, W, D)
            target (Tensor): One-hot encoded ground truth of same shape
            epsilon (float): Smoothing factor to avoid division by zero
        Returns:
            loss (float): Dice loss
        """
        assert pred.shape == target.shape, "Pred and target must be the same shape"

        intersection = torch.sum(pred * target, dim=(2,3,4))
        pred_sum = torch.sum(pred, dim=(2,3,4))
        target_sum = torch.sum(target, dim=(2,3,4))

        dice_score = (2. * intersection + epsilon) / (pred_sum + target_sum + epsilon)

        dice_loss = 1 - dice_score.mean()
        return dice_loss
    
    def clean(self):
        del self.phi_AB, self.phi_BA, self.phi_AB_vectorfield, self.phi_BA_vectorfield, self.warped_image_A, self.warped_image_B
        if self.use_label:
            del self.warped_label_A, self.warped_label_B
        if hasattr(self, 'warped_seg_A'):
            del self.warped_seg_A
        if hasattr(self, 'warped_seg_B'):
            del self.warped_seg_B
        if hasattr(self, 'warped_loss_input_A'):
            del self.warped_loss_input_A
        if hasattr(self, 'warped_loss_input_B'):
            del self.warped_loss_input_B
        if hasattr(self, 'warped_loss_input_A_jacob'):
            del self.warped_loss_input_A_jacob
        if hasattr(self, 'warped_loss_input_B_jacob'):
            del self.warped_loss_input_B_jacob

def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5), use_label=False, apply_intensity_conservation_loss=False, dice_loss_weight=0.0, loss_function_masking=False):
    dimension = len(input_shape) - 2
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=dimension),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension)))

    net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda, use_label=use_label, apply_intensity_conservation_loss=apply_intensity_conservation_loss, dice_loss_weight=dice_loss_weight, loss_function_masking=loss_function_masking)
    net.assign_identity_map(input_shape)
    return net

def make_sim(similarity, sigma=5, mind_radius=2, mind_dilation=2):
    if similarity == "lncc":
        return icon.LNCC(sigma=sigma)
    elif similarity == "lncc2":
        return icon.losses.SquaredLNCC(sigma=sigma)
    elif similarity == "mind":
        return icon.losses.MINDSSC(radius=mind_radius, dilation=mind_dilation)
    else:
        raise ValueError(f"Similarity measure {similarity} not recognized. Choose from [lncc, lncc2, mind].")

def get_multigradicon(loss_fn=icon.LNCC(sigma=5), apply_intensity_conservation_loss=False, weights_location=None, dice_loss_weight=0.0, loss_function_masking=False):
    net = make_network(input_shape, include_last_step=True, loss_fn=loss_fn, apply_intensity_conservation_loss=apply_intensity_conservation_loss, dice_loss_weight=dice_loss_weight, loss_function_masking=loss_function_masking)
    from os.path import exists
    if weights_location is None:
        weights_location = "network_weights/multigradicon1.0/Step_2_final.trch"
    if not exists(weights_location):
        print("Downloading pretrained multigradicon model")
        import urllib.request
        import os
        download_path = "https://github.com/uncbiag/uniGradICON/releases/download/multigradicon_weights/Step_2_final.trch"
        os.makedirs("network_weights/multigradicon1.0/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    print(f"Loading weights from {weights_location}")
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"), weights_only=True)
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net

def get_unigradicon(loss_fn=icon.LNCC(sigma=5), apply_intensity_conservation_loss=False, weights_location=None, dice_loss_weight=0.0, loss_function_masking=False):
    net = make_network(input_shape, include_last_step=True, loss_fn=loss_fn, apply_intensity_conservation_loss=apply_intensity_conservation_loss, dice_loss_weight=dice_loss_weight, loss_function_masking=loss_function_masking)
    from os.path import exists
    if weights_location is None:
        weights_location = "network_weights/unigradicon1.0/Step_2_final.trch"
    if not exists(weights_location):
        print("Downloading pretrained unigradicon model")
        import urllib.request
        import os
        download_path = "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch"
        os.makedirs("network_weights/unigradicon1.0/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"), weights_only=True)
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net

def get_model_from_model_zoo(model_name="unigradicon", loss_fn=icon.LNCC(sigma=5), apply_intensity_conservation_loss=False, dice_loss_weight=0.0, loss_function_masking=False, weights_location=None):
    if model_name == "unigradicon":
        return get_unigradicon(
            loss_fn,
            apply_intensity_conservation_loss,
            weights_location=weights_location,
            dice_loss_weight=dice_loss_weight,
            loss_function_masking=loss_function_masking,
        )
    elif model_name == "multigradicon":
        return get_multigradicon(
            loss_fn,
            apply_intensity_conservation_loss,
            weights_location=weights_location,
            dice_loss_weight=dice_loss_weight,
            loss_function_masking=loss_function_masking,
        )
    else:
        raise ValueError(f"Model {model_name} not recognized. Choose from [unigradicon, multigradicon].")

def quantile(arr: torch.Tensor, q):
    arr = arr.flatten()
    l = len(arr)
    return torch.kthvalue(arr, max(1, min(int(q * l), l))).values

def apply_mask(image, segmentation):
    segmentation_cast_filter = itk.CastImageFilter[type(segmentation),
                                            itk.Image.F3].New()
    segmentation_cast_filter.SetInput(segmentation)
    segmentation_cast_filter.Update()
    segmentation = segmentation_cast_filter.GetOutput()
    mask_filter = itk.MultiplyImageFilter[itk.Image.F3, itk.Image.F3,
                                    itk.Image.F3].New()

    mask_filter.SetInput1(image)
    mask_filter.SetInput2(segmentation)
    mask_filter.Update()

    return mask_filter.GetOutput()

def preprocess(image, modality="ct", segmentation=None, ct_window=None, quantile_range=None):
    """Preprocess a medical image for registration.

    Args:
        image: ITK image to preprocess.
        modality: 'ct' or 'mri'.
        segmentation: Optional ITK segmentation to mask the image (ROI masking).
        ct_window: Optional (min, max) HU window for CT. Default: (-1000, 1000).
        quantile_range: Optional (lower, upper) quantile range for MRI normalization.
            Default (None): uses actual image min and 99th percentile, matching
            the finetuning default of (0.0, 0.99).
    """
    if modality == "ct":
        if ct_window is None:
            ct_window = (-1000, 1000)
        min_ = ct_window[0]
        max_ = ct_window[1]
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
        image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    elif modality == "mri":
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
        if quantile_range is not None:
            arr = torch.tensor(np.array(image))
            min_ = quantile(arr, quantile_range[0]).item()
            max_ = quantile(arr, quantile_range[1]).item()
        else:
            min_, _ = itk.image_intensity_min_max(image)
            max_ = quantile(torch.tensor(np.array(image)), .99).item()
        image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    else:
        raise ValueError(f"{modality} not recognized. Use 'ct' or 'mri'.")

    image = itk.shift_scale_image_filter(image, shift=-min_, scale = 1/(max_-min_))

    if segmentation is not None:
        image = apply_mask(image, segmentation)
    return image

def main():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Register two images using unigradicon.")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the moving image.")
    parser.add_argument("--fixed_modality", required=True,
                         type=str, help="The modality of the fixed image. Should be 'ct' or 'mri'.")
    parser.add_argument("--moving_modality", required=True,
                         type=str, help="The modality of the moving image. Should be 'ct' or 'mri'.")
    parser.add_argument("--fixed_segmentation", required=False,
                         type=str, help="The path of the segmentation map of the fixed image.")
    parser.add_argument("--moving_segmentation", required=False,
                         type=str, help="The path of the segmentation map of the moving image.")
    parser.add_argument(
        "--masking_mode",
        choices=["roi", "loss", "none"],
        default="roi",
        help="How to use provided segmentations: 'roi' masks the input images (default), "
             "'loss' uses masks only for loss/Dice, 'none' ignores masks for both."
    )
    parser.add_argument("--transform_out", required=True,
                         type=str, help="The path to save the transform.")
    parser.add_argument("--warped_moving_out", required=False,
                        default=None, type=str, help="The path to save the warped image.")
    parser.add_argument("--io_iterations", required=False,
                         default="50", help="The number of IO iterations. Default is 50. Set to 'None' to disable IO.")
    parser.add_argument("--io_sim", required=False,
                         default="lncc", help="The similarity measure used in IO. Default is LNCC. Choose from [lncc, lncc2, mind].")
    parser.add_argument("--model", required=False,
                         default="unigradicon", help="The model to load. Default is unigradicon. Choose from [unigradicon, multigradicon].")
    parser.add_argument("--loss_function_masking", required=False,
                         action="store_true", help="Apply loss/similarity masking using the provided segmentations "
                                                    "(can be combined with masking_mode).")
    parser.add_argument("--intensity_conservation_loss", required=False,
                            action="store_true", help="Enable determinant-based intensity correction in the loss \
                            function for mass-conserving registration. Applicable only for CT modality where -1000 HU represents air.")
    parser.add_argument("--dice_loss_weight", required=False, type=float, default=0.0,
                            help="The weight of the Dice loss if segmentations are provided. Default is 0.0 (no Dice loss).")
    parser.add_argument("--network_weights", required=False, type=str, default=None,
                            help="Path to a custom network weights checkpoint (e.g., results/.../network_weights_100). "
                                 "If omitted, packaged weights are downloaded/used.")
    parser.add_argument("--ct_window", required=False, nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                            help="Custom CT HU window [min max]. Default: -1000 1000. "
                                 "Use to match finetuning preprocessing if you used a custom ct_window.")
    parser.add_argument("--quantile_range", required=False, nargs=2, type=float, default=None, metavar=("LOWER", "UPPER"),
                            help="Custom quantile range [lower upper] for MRI intensity normalization. "
                                 "Default: actual image min and 99th percentile.")

    args = parser.parse_args()
    
    if args.intensity_conservation_loss:
        if args.fixed_modality != "ct" or args.moving_modality != "ct":
            raise ValueError("Intensity conservation loss is only supported for CT images.")

    if args.network_weights is not None and not os.path.exists(args.network_weights):
        raise FileNotFoundError(f"Network weights file not found: {args.network_weights}")

    net = get_model_from_model_zoo(
        args.model,
        make_sim(args.io_sim),
        args.intensity_conservation_loss,
        args.dice_loss_weight,
        args.loss_function_masking,
        weights_location=args.network_weights,
    )

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)
    
    if args.fixed_segmentation is not None:
        fixed_segmentation = itk.imread(args.fixed_segmentation)
        fixed_segmentation = itk.CastImageFilter[type(fixed_segmentation), itk.Image[itk.SS, 3]].New()(fixed_segmentation)
    else:
        fixed_segmentation = None
    
    if args.moving_segmentation is not None:
        moving_segmentation = itk.imread(args.moving_segmentation)
        moving_segmentation = itk.CastImageFilter[type(moving_segmentation), itk.Image[itk.SS, 3]].New()(moving_segmentation)
    else:
        moving_segmentation = None
        
    needs_masks = args.loss_function_masking
    needs_segmentations = args.dice_loss_weight > 0.0
    if (needs_masks or needs_segmentations) and (fixed_segmentation is None or moving_segmentation is None):
        raise ValueError("Loss masking/Dice requires both fixed and moving segmentations.")

    if args.io_iterations == "None":
        io_iterations = None
    else:
        io_iterations = int(args.io_iterations)

    # Apply ROI masking to inputs if requested
    ct_win = tuple(args.ct_window) if args.ct_window else None
    q_range = tuple(args.quantile_range) if args.quantile_range else None

    masked_moving = preprocess(
        moving,
        args.moving_modality,
        moving_segmentation if args.masking_mode == "roi" else None,
        ct_window=ct_win,
        quantile_range=q_range,
    )
    masked_fixed = preprocess(
        fixed,
        args.fixed_modality,
        fixed_segmentation if args.masking_mode == "roi" else None,
        ct_window=ct_win,
        quantile_range=q_range,
    )

    if needs_masks or needs_segmentations:
        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair_with_mask(
            net,
            masked_moving,
            masked_fixed,
            mask_A=moving_segmentation if needs_masks else None,
            mask_B=fixed_segmentation if needs_masks else None,
            finetune_steps=io_iterations,
            segmentation_A=moving_segmentation if needs_segmentations else None,
            segmentation_B=fixed_segmentation if needs_segmentations else None,
        )
    else:
        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
            net,
            masked_moving,
            masked_fixed,
            finetune_steps=io_iterations)

    itk.transformwrite([phi_AB], args.transform_out)

    if args.warped_moving_out:
        moving, maybe_cast_back = maybe_cast(moving)
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
        warped_moving_image = itk.resample_image_filter(
                moving,
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed
                )
        warped_moving_image = maybe_cast_back(warped_moving_image)
        itk.imwrite(warped_moving_image, args.warped_moving_out)

def warp_command():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Warp an image with given transformation.")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the moving image.")
    parser.add_argument("--transform")
    parser.add_argument("--warped_moving_out", required=True)
    parser.add_argument('--nearest_neighbor', action='store_true')
    parser.add_argument('--linear', action='store_true')

    args = parser.parse_args()

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)
    if not args.transform:
        phi_AB = itk.IdentityTransform[itk.D, 3].New()
    else:
        phi_AB = itk.transformread(args.transform)[0]

    moving, maybe_cast_back = maybe_cast(moving)

    if args.linear:
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
    elif args.nearest_neighbor:
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(moving)
    else:
        raise Exception("Specify --nearest_neighbor or --linear")
    warped_moving_image = itk.resample_image_filter(
            moving,
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=fixed
            )
    
    warped_moving_image = maybe_cast_back(warped_moving_image)

    itk.imwrite(warped_moving_image, args.warped_moving_out)

def maybe_cast(img: itk.Image):
    """
    If an itk image is of a type that can't be used with InterpolateImageFunctions, cast it 
    and be able to cast it back
    """
    maybe_cast_back = lambda x: x

    if str((type(img), itk.D)) not in itk.NearestNeighborInterpolateImageFunction.GetTypesAsList():

        if type(img) in (itk.Image[itk.ULL, 3], itk.Image[itk.UL, 3]):
            raise Exception("Label maps of type unsigned long may have values that cannot be represented in a double")
 
        maybe_cast_back = itk.CastImageFilter[itk.Image[itk.D, 3], type(img)].New()

        img = itk.CastImageFilter[type(img), itk.Image[itk.D, 3]].New()(img)

    return img, maybe_cast_back

def compute_jacobian_map_command():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Compute the Jacobian map of a given transform.")
    parser.add_argument("--transform", required=True, type=str,
                            help="The path to the transform file.")
    parser.add_argument("--fixed", required=True, type=str,
                            help="The path to the fixed image that has been used in the registration.")
    parser.add_argument("--jacob", required=True, default=None, help="The path to the output Jacobian map, \
                        e.g. /path/to/output_jacobian.nii.gz")
    parser.add_argument("--log_jacob", required=False, default=None, help="The path to the output log Jacobian map. \
                            If not specified, the log Jacobian map will not be saved.")
    args = parser.parse_args()

    transform_file = args.transform
    fixed_img_file = args.fixed
    transform = itk.transformread(transform_file)[0]
    jacob = itk.displacement_field_jacobian_determinant_filter(
        itk.transform_to_displacement_field_filter(
            transform,
            reference_image=itk.imread(fixed_img_file),
            use_reference_image=True
        )
    )
    itk.imwrite(jacob, args.jacob)

    if args.log_jacob is not None:
        log_jacob = itk.LogImageFilter.New(jacob)
        log_jacob.Update()
        log_jacob = log_jacob.GetOutput()
        itk.imwrite(log_jacob, args.log_jacob)
