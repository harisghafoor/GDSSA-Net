import os
import time
from glob import glob
import random
import json
import torch
from torch.utils.data import DataLoader
import argparse  # Import argparse for command-line arguments

# from dataloader import T_Data
from dataloader import DataLungNodulesLoader

# from model import VNet
# from attention_unet import AttentionUNet
# from deep_sup_attention_unet_type2 import AttentionUNetpp
from models import AttentionUNetppGradual
from loss import (
    DiceLoss,
    DiceBCELoss,
    dice_loss,
    HausdorffDTLoss,
    DiceLoss_V1,
    DiceTestingScore,
)
from utils import seeding, create_dir, epoch_time
import warnings

warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from config import (
    KERNEL_DILATION_1,
    KERNEL_DILATION_2,
    BATCH_SIZE,
    HEIGHT,
    WIDTH,
    WEIGHTED_COEFF_MAIN_LOSS,
    WEIGHTED_COEFF_AUX_LOSS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_LR,
)


# DataLoader(..., collate_fn=collate_fn)
def train(model, loader, optimizer, loss_fn, device, loss_fn_boundary):
    """
    Trains the model using the provided data loader and optimizer.

    Args:
        model (torch.nn.Module): The model to be trained.
        loader (torch.utils.data.DataLoader): The data loader containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_fn (torch.nn.Module): The loss function used to calculate the main loss.
        device (torch.device): The device on which the training will be performed.
        loss_fn_boundary (torch.nn.Module): The loss function used to calculate the boundary loss.

    Returns:
        tuple: A tuple containing the epoch loss, primary loss, and auxiliary loss.
    """
    epoch_loss = 0.0
    epoch_primary_loss = 0.0
    epoch_auxiliary_loss = 0.0

    model.train()
    for x, y, contour, dilated_1, dilated_2 in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        contour = contour.to(device, dtype=torch.float32)
        dilated_1 = dilated_1.to(device, dtype=torch.float32)
        dilated_2 = dilated_2.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred, ds_out = model(x)

        deepest_loss = loss_fn(y_pred, y)
        weighted_deepest_loss = WEIGHTED_COEFF_MAIN_LOSS * deepest_loss
        ds_corr_weight = WEIGHTED_COEFF_AUX_LOSS / len(ds_out)

        if type(ds_out) is list:
            list_of_loss = []
            for ground_truth, deep_output in zip([dilated_2, dilated_1], ds_out[:-1]):
                new_height = deep_output.size(-2)
                new_width = deep_output.size(-1)
                downsampled_mask = F.interpolate(
                    ground_truth,
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=True,
                )
                aux_loss = loss_fn(deep_output, downsampled_mask)
                list_of_loss.append(ds_corr_weight * aux_loss)

            contour_downsampled = F.interpolate(
                contour,
                size=(ds_out[-1].size(-2), ds_out[-1].size(-1)),
                mode="bilinear",
                align_corners=True,
            )
            contour_downsampled = contour_downsampled.to(device, dtype=torch.float32)
            list_of_loss.append(
                ds_corr_weight
                * loss_fn_boundary(pred=ds_out[-1], target=contour_downsampled)
            )
            list_of_loss.append(weighted_deepest_loss)
            stacked_tensors_of_weighted_aux_loss = torch.stack(list_of_loss)
            loss = torch.sum(stacked_tensors_of_weighted_aux_loss, dim=0)
        else:
            loss = deepest_loss

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_primary_loss += deepest_loss.item()
        epoch_auxiliary_loss += aux_loss.item()

    epoch_loss = epoch_loss / len(loader)
    epoch_primary_loss = epoch_primary_loss / len(loader)
    epoch_auxiliary_loss = epoch_auxiliary_loss / len(loader)

    return epoch_loss, epoch_primary_loss, epoch_auxiliary_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y, y1, y2, y3 in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred, _ = model(
                x
            )  # Does not matter what we get from other stages -> focus on main output while evaluating
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":
    # Define default values for arguments
    DEFAULT_DATA_DIR = "/path/to/dataset"
    DEFAULT_OUTPUT_DIR = "/output/path"
    DEFAULT_EXPERIMENT_NAME = "MyExperiment"
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_NUM_EPOCHS = 100
    DEFAULT_DEVICE = "cuda:0"
    DEFAULT_PARENT_DIR = "BestFoldAttentionUnetDDTI"
    DEFAULT_AUGMENTED_DATA = "augmented_data_ddti"
    DEFAULT_FOLD = "fold_2"

    parser = argparse.ArgumentParser(
        description="Lung Nodule Segmentation Training Script"
    )

    parser.add_argument(
        "--data_dir", default=DEFAULT_DATA_DIR, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Path to the output directory"
    )
    parser.add_argument(
        "--experiment_name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument("--lr",
                        type=float,
                        default=DEFAULT_LR,
                        help="Learning rate")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Device for training (e.g., 'cuda:0' or 'cpu')",
    )
    parser.add_argument(
        "--PARENT_DIR", default=DEFAULT_PARENT_DIR, help="Root directory of checkpoints"
    )
    parser.add_argument(
        "--augmented_data",
        default=DEFAULT_AUGMENTED_DATA,
        help="Directory of Augmented Images",
    )
    parser.add_argument("--fold", default=DEFAULT_FOLD, help="Name of the fold")

    args = parser.parse_args()

    # Override configuration parameters with command-line arguments
    PARENT_DIR = args.PARENT_DIR
    AUGMENTED_DATA = args.augmented_data

    # Seeding
    seeding(42)

    # Directories
    create_dir(args.output_dir)

    # Dataset and loader
    checkpoint_path = os.path.join(args.output_dir, f"{args.experiment_name}.pth")
    writer = SummaryWriter(comment=f"_{args.experiment_name}")

    aug_x = os.listdir(os.path.join(args.augmented_data, "images"))
    aug_y = os.listdir(os.path.join(args.augmented_data, "masks"))
    new_aug_x = [os.path.join(args.augmented_data, "images", x) for x in aug_x]
    new_aug_y = [os.path.join(args.augmented_data, "masks", y) for y in aug_y]

    with open(os.path.join(args.data_dir + ".json"), "r") as f:
        folds = json.load(f)
    data = folds[args.fold]
    train_paths = data["train"]
    images, masks = zip(*train_paths)
    new_train_x = list(images) + new_aug_x
    new_train_y = list(masks) + new_aug_y
    new_train_paths = [*zip(new_train_x, new_train_y)]
    valid_paths = data["val"]
    train_x, train_y = zip(*new_train_paths)
    valid_x, valid_y = zip(*valid_paths)

    # Hyperparameters
    H = HEIGHT
    W = WIDTH
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr

    # Create datasets and dataloaders for this fold
    train_dataset = DataLungNodulesLoader(
        train_x,
        train_y,
        image_size=(H, W),
        kernel_size_1=KERNEL_DILATION_1,
        kernel_size_2=KERNEL_DILATION_2,
    )
    valid_dataset = DataLungNodulesLoader(
        valid_x,
        valid_y,
        image_size=(H, W),
        kernel_size_1=KERNEL_DILATION_1,
        kernel_size_2=KERNEL_DILATION_2,
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    device = torch.device(args.device)
    print("Device:", device)

    model = AttentionUNetppGradual()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, verbose=True
    )
    loss_fn_seg = DiceLoss_V1()
    testing_loss = DiceTestingScore()
    loss_fn_boundary = HausdorffDTLoss()
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid:{len(valid_x)}\n"
    print(data_str)

    # Training the model
    best_valid_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, primary_loss, auxiliary_loss = train(
            model, train_loader, optimizer, loss_fn_seg, device, loss_fn_boundary
        )
        valid_loss = evaluate(model, valid_loader, loss_fn_seg, device)

        # Saving the model
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
        else:
            scheduler.step(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.3f} | Primary Loss: {primary_loss:.3f} | Auxiliary Loss: {auxiliary_loss:.3f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.3f}\n"
        print(data_str)
        # Logging training and validation losses to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", valid_loss, epoch)
        writer.add_scalars(
            "Loss/Train", {"Primary": primary_loss, "Auxiliary": auxiliary_loss}, epoch
        )

    # Last Iteration model
    torch.save(
        model.state_dict(),
        os.path.join(args.output_dir, f"{args.experiment_name}_LastIteration.pth"),
    )
    # Close the TensorBoard writer when finished logging
    writer.close()
