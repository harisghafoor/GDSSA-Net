import json
import os
import argparse
from glob import glob
from sklearn.model_selection import KFold


def create_folds(data_paths, save_path, num_folds):
    all_x = []
    all_y = []

    for data_path in data_paths:
        x = sorted(glob(os.path.join(data_path[0], "*")))
        y = sorted(glob(os.path.join(data_path[1], "*")))
        assert len(x) == len(y), "Number of images and masks must match."

        all_x.extend(x)
        all_y.extend(y)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds = {}

    for i, (train_index, val_index) in enumerate(kf.split(all_x)):
        folds[f"fold_{i+1}"] = {
            "train": [(all_x[idx], all_y[idx]) for idx in train_index],
            "val": [(all_x[idx], all_y[idx]) for idx in val_index],
        }

    with open(save_path, "w") as f:
        json.dump(folds, f)


if __name__ == "__main__":
    # Define default paths
    DEFAULT_DATA_PATHS = [
       ("/Users/harisghafoor/Documents/GitHub/DeepAttent-EdgeCraft-Gradual-Supervision-for-Superior-Segmentation/datasets/Thyroid Dataset/tn3k/trainval-image",
        "/Users/harisghafoor/Documents/GitHub/DeepAttent-EdgeCraft-Gradual-Supervision-for-Superior-Segmentation/datasets/Thyroid Dataset/tn3k/trainval-mask"),
        ("/Users/harisghafoor/Documents/GitHub/DeepAttent-EdgeCraft-Gradual-Supervision-for-Superior-Segmentation/datasets/Thyroid Dataset/tn3k/test-image",
        "/Users/harisghafoor/Documents/GitHub/DeepAttent-EdgeCraft-Gradual-Supervision-for-Superior-Segmentation/datasets/Thyroid Dataset/tn3k/test-mask")
    ]  # Add your default data paths here
    
    DEFAULT_SAVE_PATH = "tn3k_combo_folds.json"  # Specify your default save path here
    DEFAULT_NUM_FOLDS = 10  # Specify the default number of folds here
    
    parser = argparse.ArgumentParser(description="Create Folds Script")
    parser.add_argument(
        "--data_paths", nargs="+", default=DEFAULT_DATA_PATHS, help="List of data paths"
    )
    parser.add_argument(
        "--save_path", default=DEFAULT_SAVE_PATH, help="Path to save the folds JSON"
    )
    parser.add_argument("--num_folds", type=int, default=DEFAULT_NUM_FOLDS, help="Number of folds")
    args = parser.parse_args()

    data_paths = args.data_paths
    save_path = args.save_path
    num_folds = args.num_folds

    create_folds(data_paths, save_path, num_folds)