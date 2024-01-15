import numpy as np
import matplotlib.pyplot as plt
import json
from operator import add
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.utils import seeding, create_dir, epoch_time
from src.dataloader import DataLungNodulesLoader
from src.models import AttentionUNetppGradual
from src.loss import DiceLoss

from src.config import (KERNEL_DILATION_1,
                        KERNEL_DILATION_2,
                        BATCH_SIZE,
                        HEIGHT,WIDTH
)
import argparse


def calculate_metrics(y_true, y_pred, threshold):
    # Ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # Prediction
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > threshold
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def save_summary_to_file(metrics_summary, experiment_name):
    summary_path = f"summaries/{experiment_name}"
    create_dir(summary_path)
    summary_filename = f"{summary_path}/summary.json"
    with open(summary_filename, "w") as json_file:
        json.dump(metrics_summary, json_file)
    print(f"Summary for threshold saved to {summary_filename}")


def save_decoder_outputs_to_disk(decoder_output, experiment_name):
    output_path = f"decoder_outputs/{experiment_name}"
    create_dir(output_path)

    batch_size, height, width = decoder_output.shape

    for i in range(batch_size):
        decoder_output_np = decoder_output[i, :, :].cpu().numpy()

        output_filename = f"{output_path}/decoder_output_sample_{i}.png"
        plt.imsave(output_filename, decoder_output_np, cmap="gray")


def evaluate(model, loader, loss_fn, device, thresh):
    epoch_loss = 0.0
    model.eval()

    metrics_score = {
        "main_prediction": [0.0, 0.0, 0.0, 0.0, 0.0],
        "branch_1": [0.0, 0.0, 0.0, 0.0, 0.0],
        "branch_2": [0.0, 0.0, 0.0, 0.0, 0.0],
        "branch_3": [0.0, 0.0, 0.0, 0.0, 0.0],
        "ensemble": [0.0, 0.0, 0.0, 0.0, 0.0],
    }

    with torch.no_grad():
        for x, y, y1, y2, y3 in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred, ls = model(x)
            x = x.squeeze(0)
            y_pred = y_pred.squeeze(0)
            y_pred = y_pred.squeeze(1)
            y = y.squeeze(1)

            loss = loss_fn(y_pred, y)
            main_score = calculate_metrics(y_true=y, y_pred=y_pred, threshold=thresh[0])
            metrics_score["main_prediction"] = list(
                map(add, metrics_score["main_prediction"], main_score)
            )

            ensemble_prediction = 0.95 * y_pred
            for i, out in enumerate(ls, start=1):
                resized_outs = F.interpolate(
                    out, size=(256, 256), mode="bilinear", align_corners=True
                )
                resized_outs = resized_outs.squeeze(1)
                ensemble_prediction += 0.025 * resized_outs

            ensemble_score = calculate_metrics(
                y_true=y, y_pred=ensemble_prediction, threshold=thresh[0]
            )
            metrics_score["ensemble"] = list(
                map(add, metrics_score["ensemble"], ensemble_score)
            )

            save_decoder_outputs_to_disk(
                ensemble_prediction, experiment_name + "_ensemble"
            )

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)

        metrics_summary = {}
        print("Length of Loader:", len(loader))
        for key, value in metrics_score.items():
            if key == "main_prediction":
                branch_name = "Main Prediction"
            elif key == "ensemble":
                branch_name = "Ensemble"
            else:
                branch_name = key.capitalize()

            jaccard = value[0] / len(loader)
            f1 = value[1] / len(loader)
            recall = value[2] / len(loader)
            precision = value[3] / len(loader)
            acc = value[4] / len(loader)
            dsc = 1 - f1

            metrics_summary[branch_name] = {
                "Jaccard": jaccard,
                "F1": f1,
                "Recall": recall,
                "Precision": precision,
                "Acc": acc,
                "DSC": dsc,
            }

            print(
                f"{branch_name} - Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - DSC: {dsc:1.4f}"
            )

        save_summary_to_file(metrics_summary, experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Testing")
    parser.add_argument(
        "--fold", type=str, default="fold_2", help="Specify the fold to use"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Thyroid_Segmentation_Experiment",
        help="Specify experiment name",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.1, 0.1, 0.5],
        help="Threshold values to evaluate",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="folds_combo.json",
        help="Path to the JSON file containing fold data",
    )
    # Add other command-line arguments as needed
    args = parser.parse_args()
    PARENT_DIR = "BestFoldAttentionUnetPP_Type2"

    seeding(42)
    # Load folds from the specified JSON file
    with open(args.json_file, "r") as f:
        folds = json.load(f)
    fold = args.fold
    data = folds[fold]
    experiment_name = args.experiment_name
    checkpoint_path = args.checkpoint_path
    train_paths, valid_paths = data["train"], data["val"]

    valid_x, valid_y = zip(*valid_paths)

    H = HEIGHT
    W = WIDTH
    size = (H, W)
    batch_size = BATCH_SIZE 
    valid_dataset = DataLungNodulesLoader(
        valid_x,
        valid_y,
        image_size=(H, W),
        kernel_size_1=KERNEL_DILATION_1,
        kernel_size_2=KERNEL_DILATION_2,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    device = args.device
    print("device: ", device)
    model = AttentionUNetppGradual()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device), strict=False
    )
    model = model.to(device)
    loss_fn = DiceLoss()

    thresholds = args.thresholds
    valid_loss = evaluate(model, valid_loader, loss_fn, device, thresh=thresholds)
