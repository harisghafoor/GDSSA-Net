import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class DataLungNodulesLoader(Dataset):
    def __init__(
        self, images_path, masks_path, image_size, kernel_size_1, kernel_size_2
    ):
        """
        Initialize the Lung Nodules Dataset Loader.

        Args:
            images_path (list of str): List of paths to lung nodule images.
            masks_path (list of str): List of paths to lung nodule masks.
            image_size (tuple): Desired image size (height and width).
            kernel_size_1 (int): First dilated mask kernel size.
            kernel_size_2 (int): Second dilated mask kernel size.
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.to_return_size = image_size
        self.kernel_sizes = [kernel_size_1, kernel_size_2]

    def __getitem__(self, index):
        """Reading image"""
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.to_return_size)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """Reading mask"""
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.to_return_size)
        mask = mask / mask.max()
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.uint8)
        mask = torch.from_numpy(mask)

        """Generating contour mask"""
        contour_mask = self.contour_image_from_binary_mask(mask.unsqueeze(0))
        contour_mask = contour_mask / 255.0  # Normalize to [0, 1]

        """Generating dilated masks"""
        dilated_masks = []
        for size_tuple in self.kernel_sizes:
            dilated_mask = self.dilate_contour_mask(
                contour_mask.unsqueeze(0),
                mask.unsqueeze(0),
                kernel_size=(size_tuple, size_tuple),
            )
            dilated_mask = dilated_mask / 255.0
            dilated_masks.append(dilated_mask)

        return image, mask, contour_mask, dilated_masks[0], dilated_masks[1]

    def __len__(self):
        return self.n_samples

    def dilate_contour_mask(
        self, contour_mask, ground_truth_mask, kernel_size=(21, 21)
    ):
        """
        Dilates the contour mask and masks out the expansion outside the original ground truth boundaries.

        Args:
            contour_mask (Tensor): PyTorch Tensor representing the contour binary mask.
            ground_truth_mask (Tensor): PyTorch Tensor representing the original ground truth binary mask.
            kernel_size (tuple): Kernel size for dilation.

        Returns:
            dilated_mask (Tensor): PyTorch Tensor representing the dilated and masked contour mask.
        """
        # Convert PyTorch Tensors to NumPy arrays
        contour_mask = contour_mask.cpu().numpy().squeeze()
        ground_truth_mask = ground_truth_mask.cpu().numpy().squeeze()

        # Dilate the contour mask
        dilated_mask = cv2.dilate(
            contour_mask.astype(np.uint8),
            kernel=np.ones(kernel_size, np.uint8),
            iterations=1,
        )
        # Use the original ground truth mask to mask out expansion outside the original boundaries
        dilated_mask = np.minimum(dilated_mask, ground_truth_mask)
        # Convert the NumPy array to a PyTorch Tensor
        dilated_mask = torch.from_numpy(dilated_mask).unsqueeze(0).float()

        return dilated_mask

    def contour_image_from_binary_mask(self, binary_mask):
        """
        Generates a contour image from a binary mask and returns it as a PyTorch Tensor.

        Args:
            binary_mask (Tensor): PyTorch Tensor representing the binary mask.

        Returns:
            contour_tensor (Tensor): PyTorch Tensor representing the contour image.
        """
        # Ensure the input mask is in the right format (single channel binary image)
        binary_mask = binary_mask.cpu().numpy().squeeze()

        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contours on a blank image
        contour_image = np.zeros_like(binary_mask)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

        # Convert the NumPy array to a PyTorch Tensor
        contour_tensor = torch.from_numpy(contour_image).unsqueeze(0).float()

        return contour_tensor
