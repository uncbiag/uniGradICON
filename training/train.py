import os
import random
from datetime import datetime

from tqdm import tqdm
import torch
import torch.nn.functional as F
from dataset import COPDDataset, HCPDataset, OAIDataset, L2rAbdomenDataset
from torch.utils.data import ConcatDataset, DataLoader

import icon_registration as icon
import icon_registration.networks as networks
from icon_registration.losses import ICONLoss, to_floats

from unigradicon import make_network

def write_stats(writer, stats: ICONLoss, ite, prefix=""):
    for k, v in to_floats(stats)._asdict().items():
        writer.add_scalar(f"{prefix}{k}", v, ite)

input_shape = [1, 1, 175, 175, 175]

BATCH_SIZE= 4
device_ids = [1, 0, 2, 3]
GPUS = len(device_ids)
EXP_DIR = "./results/unigradicon/"

def get_dataset():
    data_num = 1000
    return ConcatDataset(
        (
        COPDDataset(desired_shape=input_shape[2:], device=device_ids[0], ROI_only=True, data_num=data_num),
        OAIDataset(desired_shape=input_shape[2:], device=device_ids[0], data_num=data_num),
        HCPDataset(desired_shape=input_shape[2:], device=device_ids[0], data_num=data_num),
        L2rAbdomenDataset(desired_shape=input_shape[2:], device=device_ids[0], data_num=data_num)
        )
    )

def augment(image_A, image_B):
    device = image_A.device
    identity_list = []
    for i in range(image_A.shape[0]):
        identity = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], device=device)
        idxs = set((0, 1, 2))
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1 
        identity = identity * (torch.randint_like(identity, 0, 2, device=device) * 2  - 1)
        identity_list.append(identity)

    identity = torch.cat(identity_list)
    
    noise = torch.randn((image_A.shape[0], 3, 4), device=device)

    forward = identity + .05 * noise  

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward, grid_shape)
   
    if image_A.shape[1] > 1:
        # Then we have segmentations
        warped_A = F.grid_sample(image_A[:, :1], forward_grid, padding_mode='border')
        warped_A_seg = F.grid_sample(image_A[:, 1:], forward_grid, mode='nearest', padding_mode='border')
        warped_A = torch.cat([warped_A, warped_A_seg], axis=1)
    else:
        warped_A = F.grid_sample(image_A, forward_grid, padding_mode='border')

    noise = torch.randn((image_A.shape[0], 3, 4), device=device)
    forward = identity + .05 * noise  

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward, grid_shape)

    if image_B.shape[1] > 1:
        # Then we have segmentations
        warped_B = F.grid_sample(image_B[:, :1], forward_grid, padding_mode='border')
        warped_B_seg = F.grid_sample(image_B[:, 1:], forward_grid, mode='nearest', padding_mode='border')
        warped_B = torch.cat([warped_B, warped_B_seg], axis=1)
    else:
        warped_B = F.grid_sample(image_B, forward_grid, padding_mode='border')

    return warped_A, warped_B

def train_kernel(optimizer, net, moving_image, fixed_image, writer, ite):
    optimizer.zero_grad()
    loss_object = net(moving_image, fixed_image)
    loss = torch.mean(loss_object.all_loss)
    loss.backward()
    optimizer.step()
    # print(to_floats(loss_object))
    write_stats(writer, loss_object, ite, prefix="train/")

def train(
    net,
    optimizer,
    data_loader,
    val_data_loader,
    epochs=200,
    eval_period=-1,
    save_period=-1,
    step_callback=(lambda net: None),
    unwrapped_net=None,
    data_augmenter=None,
):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """
    import footsteps
    from torch.utils.tensorboard import SummaryWriter

    if unwrapped_net is None:
        unwrapped_net = net

    loss_curve = []
    writer = SummaryWriter(
        footsteps.output_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )

    iteration = 0
    for epoch in tqdm(range(epochs)):
        for moving_image, fixed_image in data_loader:
            moving_image, fixed_image = moving_image.cuda(), fixed_image.cuda()
            if data_augmenter is not None:
                with torch.no_grad():
                    moving_image, fixed_image = data_augmenter(moving_image, fixed_image)
            train_kernel(optimizer, net, moving_image, fixed_image,
                         writer, iteration)
            iteration += 1

            step_callback(unwrapped_net)
        
        
        if epoch % save_period == 0:
            torch.save(
                optimizer.state_dict(),
                footsteps.output_dir + "checkpoints/optimizer_weights_" + str(epoch),
            )
            torch.save(
                unwrapped_net.regis_net.state_dict(),
                footsteps.output_dir + "checkpoints/network_weights_" + str(epoch),
            )

        if epoch % eval_period == 0:
            visualization_moving, visualization_fixed = next(iter(val_data_loader))
            visualization_moving, visualization_fixed = visualization_moving[:, :1].cuda(), visualization_fixed[:, :1].cuda()
            unwrapped_net.eval()
            warped = []
            with torch.no_grad():
                eval_loss = unwrapped_net(visualization_moving, visualization_fixed)
                write_stats(writer, eval_loss, epoch, prefix="val/")
                warped = unwrapped_net.warped_image_A.cpu()
                del eval_loss
                unwrapped_net.clean()
            unwrapped_net.train()

            def render(im):
                if len(im.shape) == 5:
                    im = im[:, :, :, im.shape[3] // 2]
                if torch.min(im) < 0:
                    im = im - torch.min(im)
                if torch.max(im) > 1:
                    im = im / torch.max(im)
                return im[:4, [0, 0, 0]].detach().cpu()

            writer.add_images(
                "moving_image", render(visualization_moving[:4]), epoch, dataformats="NCHW"
            )
            writer.add_images(
                "fixed_image", render(visualization_fixed[:4]), epoch, dataformats="NCHW"
            )
            writer.add_images(
                "warped_moving_image",
                render(warped),
                epoch,
                dataformats="NCHW",
            )
            writer.add_images(
                "difference",
                render(torch.clip((warped[:4, :1] - visualization_fixed[:4, :1].cpu()) + 0.5, 0, 1)),
                epoch,
                dataformats="NCHW",
            )
    
    torch.save(
        optimizer.state_dict(),
        footsteps.output_dir + "checkpoints/optimizer_weights_" + str(epoch),
    )
    torch.save(
        unwrapped_net.regis_net.state_dict(),
        footsteps.output_dir + "checkpoints/network_weights_" + str(epoch),
    )

def train_two_stage(input_shape, data_loader, val_data_loader, GPUS, epochs, eval_period, save_period, resume_from):

    net = make_network(input_shape, include_last_step=False)

    torch.cuda.set_device(device_ids[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # device = f"cuda:{device_ids[0]}"

    # Continue train 
    if resume_from != "":
        print("Resume from: ", resume_from)
        net.regis_net.load_state_dict(torch.load(resume_from, map_location="cpu"))
    
    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0]).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    if resume_from != "":
        optimizer.load_state_dict(torch.load(resume_from.replace("network_weights_", "optimizer_weights_"), map_location="cpu"))

    net_par.train()

    print("start train.")
    train(net_par, optimizer, data_loader, val_data_loader, unwrapped_net=net, 
          epochs=epochs[0], eval_period=eval_period, save_period=save_period, data_augmenter=augment)
    
    torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + "checkpoints/Step_1_final.trch",
            )

    net_2 = make_network(input_shape, include_last_step=True)

    net_2.regis_net.netPhi.load_state_dict(net.regis_net.state_dict())

    # Continue train 
    # if resume_from != "":
    #     print("Resume from: ", resume_from)
    #     net_2.regis_net.load_state_dict(torch.load(resume_from, map_location="cpu"))

    del net
    del net_par
    del optimizer

    if GPUS == 1:
        net_2_par = net_2.cuda()
    else:
        net_2_par = torch.nn.DataParallel(net_2, device_ids=device_ids, output_device=device_ids[0]).cuda()
    optimizer = torch.optim.Adam(net_2_par.parameters(), lr=0.00005)

    # if resume_from != "":
    #     optimizer.load_state_dict(torch.load(resume_from.replace("network_weights_", "optimizer_weights_"), map_location="cpu"))

    net_2_par.train()
    
    # We're being weird by training two networks in one script. This hack keeps
    # the second training from overwriting the outputs of the first.
    footsteps.output_dir_impl = footsteps.output_dir + "2nd_step/"
    os.makedirs(footsteps.output_dir + "checkpoints", exist_ok=True)

    train(net_2_par, optimizer, data_loader, val_data_loader, unwrapped_net=net_2, epochs=epochs[1], eval_period=eval_period, save_period=save_period, data_augmenter=augment)
    
    torch.save(
                net_2.regis_net.state_dict(),
                footsteps.output_dir + "checkpoints/Step_2_final.trch",
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", required=False, default="")
    args = parser.parse_args()
    resume_from = args.resume_from

    import footsteps
    footsteps.initialize(output_root=EXP_DIR)

    dataset = get_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE*GPUS,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    print("Finish data loading...")

    os.makedirs(footsteps.output_dir + "checkpoints", exist_ok=True)

    train_two_stage(input_shape, dataloader, val_dataloader, GPUS, [801,201], 20, 20, resume_from)