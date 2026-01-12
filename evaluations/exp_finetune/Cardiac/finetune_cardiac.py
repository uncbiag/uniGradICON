import os
import torch
from dataset import ACDCMMDataset
from torch.utils.data import DataLoader

from unigradicon import make_network

import sys
# Dynamically add the training folder to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(current_dir, "..", "..", "..", "training")
sys.path.append(os.path.abspath(training_dir))
from train import augment, train


input_shape = [1, 1, 175, 175, 175]

BATCH_SIZE= 4
device_ids = [0,1,2,3]
GPUS = len(device_ids)


# Dataset on GPU2 or GPU3
def get_dataset(data_num=640):
    # TODO: Set the path to the preprocessed cardiac data
    return ACDCMMDataset(
        data_path="./processed_data/cardiac_mri/imgs",
        data_num=data_num,
        desired_shape=input_shape[2:], device=device_ids[0])


def finetune(input_shape, data_loader, val_data_loader, GPUS, epochs, eval_period, save_period, init_from, resume_from=None):

    net = make_network(input_shape, include_last_step=True)

    torch.cuda.set_device(device_ids[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # device = f"cuda:{device_ids[0]}"

    # Continue train 
    if init_from != "":
        print("Init from: ", init_from)
        net.regis_net.load_state_dict(torch.load(init_from, map_location="cpu"))
    else:
        print("Train from scratch.")

    if resume_from:
        print("Resume from: ", resume_from)
        net.regis_net.load_state_dict(torch.load(resume_from, map_location="cpu"))
        start_epochs = int(resume_from.split("_")[-1]) + 1
    else:
        start_epochs = 0
    
    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0]).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    if resume_from:
        print("Resume Optimizer from: ", resume_from.replace("network_weights_", "optimizer_weights_"))
        optimizer.load_state_dict(torch.load(resume_from.replace("network_weights_", "optimizer_weights_"), map_location="cpu"))

    net_par.train()

    print("start train.")
    train(net_par, optimizer, data_loader, val_data_loader, unwrapped_net=net, start_epochs=start_epochs,
          epochs=epochs, eval_period=eval_period, save_period=save_period, data_augmenter=augment)
    
    torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + "checkpoints/Finetune_final.trch",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_from", required=False, default="")
    parser.add_argument("--resume_from", required=False, default="")
    parser.add_argument("--exp", required=False, default="finetune_ACDCMM")
    parser.add_argument("--epochs", required=False, type=int, default="1000")
    parser.add_argument("--data_num", required=False, type=int, default="640")
    args = parser.parse_args()
    init_from = args.init_from

    import footsteps
    footsteps.initialize(run_name=args.exp, output_root=os.path.join(current_dir, "training_results"))

    if args.data_num <= 4:
        GPUS = 1
        device_ids = [0]
        BATCH_SIZE = args.data_num
    elif args.data_num < 16:
        GPUS = args.data_num // 4
        device_ids = [0,1,2,3][:GPUS]
        BATCH_SIZE = 4

    dataset = get_dataset(data_num=args.data_num)
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
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )
    print("Finish data loading...")

    os.makedirs(footsteps.output_dir + "checkpoints", exist_ok=True)

    # Check dataloader
    print(f"Loading data {next(iter(dataloader))[0].shape} per epoch.")

    finetune(input_shape, dataloader, val_dataloader, GPUS, args.epochs, 100, 100, init_from, args.resume_from)