
import os
import footsteps
import itk
import footsteps
import tqdm
import icon_registration.itk_wrapper
import numpy as np
import matplotlib.pyplot as plt
import torch

from . import get_unigradicon, preprocess


def make_atlas(dataset: torch.Tensor, model):
    atlas = torch.mean(dataset, axis=0, keepdims=True).cuda()
    with torch.no_grad():
        disps = []
        warpeds = []
        for img in tqdm.tqdm(dataset[:]):
            img = img[None].cuda()
            model(img, atlas)
            warpeds.append(model.warped_image_A.cpu())
            model(atlas, img)
            disps.append(model.phi_AB_vectorfield.cpu())
        warpeds = torch.cat(warpeds)
        disps = torch.cat(disps)
    atlas = torch.mean(warpeds, axis=0, keepdims=True)
    fix = torch.mean(disps, axis=0, keepdims=True)
    atlas = model.as_function(atlas)(fix)
    atlas = atlas.cuda()

    disps = []
    warpeds = []
    for img in tqdm.tqdm(dataset[:]):
        model = cli.get_model()
        img = img[None].cuda()
        icon_registration.itk_wrapper.finetune_execute(model, img, atlas, 10)
        warpeds.append(model.warped_image_A.cpu().detach())
    warpeds = torch.cat(warpeds)
    disps = torch.cat(disps)
    atlas = torch.mean(warpeds, axis=0, keepdims=True)
    atlas = torch.median(warpeds, axis=0, keepdims=True)[0]
    
    return atlas


def main():
    import itk
    import argparse
    parser = argparse.argumentparser(description="register two images using unigradicon.")
    parser.add_argument("--images", required=True, type=str,
                         help="a glob for the images")
    parser.add_argument("--modality", required=True,
                         type=str, help="the modality of the images")
    parser.add_argument("--atlas_out", required=False,
                        default=none, type=str, help="the path to save the atlas")

    args = parser.parse_args()

    net = get_unigradicon()

    images = []

    image_paths = glob.glob(args.images)

    for path in image_paths:
        image = itk.imread(path)
        image = preprocess(image, args.modality)
        image = torch.Tensor(np.array(image))[None, None]
        image = F.interpolate(
            image, size=[175, 175, 175], mode="trilinear", align_corners=False
        )
        images.append(image)

    images = torch.cat(images)
    atlas_torch = make_atlas(images, net)

    metadata_image = itk.imread(image_paths[0])
    memory_view = itk.GetImageViewFromArray(metadata_image)
    old_shape = memory_view.shape
    atlas_resized = F.interpolate(atlas_torch, size=old_shape, mode="trilinear", align_corners=False)

    atlas_resized = np.array(atlas_resized)
    atlas_resized /= np.max(atlas_resized)
    atlas_resized *= np.max(memory_view) - np.min(memory_view)
    atlas_resized += np.min(memory_view)
    
    memory_view[:] = atlas_resized

    itk.imwrite(metadata_image, args.atlas_out)

if __name__ == "__main__":
    main()
