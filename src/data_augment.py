import os
import argparse
import cv2
import json
from tqdm import tqdm
from albumentations import RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip
from glob import glob


def load_data(path):
    # Get a list of image and mask file paths
    images = sorted(
        glob(os.path.join(path, "images/*"))
    )  # Corrected the path to images
    masks = sorted(glob(os.path.join(path, "masks/*")))  # Corrected the path to masks
    return images, masks


def create_dir(path):
    # Create a directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def augment_data(images, masks, save_path, augment=True):
    H = 128
    W = 128

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = os.path.basename(x).split(".")
        image_name = name[0]
        image_extn = name[1]

        name = os.path.basename(y).split(".")
        mask_name = name[0]
        mask_extn = name[1]

        # Read image and mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if augment == True:
            # Apply data augmentation transformations
            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            save_images = [x2, x3, x4, x5]
            save_masks = [y2, y3, y4, y5]
        else:
            save_images = [x]
            save_masks = [y]

        idx = 0
        for i, m in zip(save_images, save_masks):
            image_path = os.path.join(
                save_path, "images", f"{image_name}_{idx}.{image_extn}"
            )
            mask_path = os.path.join(
                save_path, "masks", f"{mask_name}_{idx}.{mask_extn}"
            )

            # Save augmented image and mask
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Data Augmentation Script")
    parser.add_argument("--data_path", required=True, help="Path to the data folder")
    parser.add_argument(
        "--output_path", required=True, help="Path to the output folder"
    )
    parser.add_argument(
        "--augment", action="store_true", help="Whether to augment the data"
    )
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path
    augment = args.augment

    images, masks = load_data(data_path)

    create_dir(os.path.join(output_path, "images"))
    create_dir(os.path.join(output_path, "masks"))

    augment_data(images, masks, output_path, augment)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation Script")
    parser.add_argument(
        "--json_path", required=True, help="Path to the JSON file containing folds"
    )
    parser.add_argument(
        "--fold_name", required=True, help="Name of the fold to process"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Name of the fold to process"
    )
    args = parser.parse_args()

    json_path = args.json_path
    fold_name = args.fold_name

    # Load folds from JSON
    with open(json_path, "r") as f:
        folds = json.load(f)

    if fold_name not in folds:
        print(f"Fold '{fold_name}' not found in the JSON file.")
    else:
        data = folds[fold_name]
        train_paths = data["train"]
        images, masks = zip(*train_paths)
        print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

        create_dir(f"{args.output_dir}/images")
        create_dir(f"{args.output_dir}/masks")

        augment_data(images, masks, "augmented_data", augment=True)
