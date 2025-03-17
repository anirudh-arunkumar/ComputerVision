"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(inp_size: Tuple[int, int]) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fundamental_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    # raise NotImplementedError(
    #     "`get_fundamental_transforms` function in "
    #     + "`data_transforms.py` needs to be implemented"
    # )

    fundamental_transforms = transforms.Compose([transforms.Resize(inp_size), transforms.ToTensor(),])

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fundamental_transforms


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    fund_aug_transforms = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    # raise NotImplementedError(
    #     "`get_fundamental_augmentation_transforms` function in "
    #     + "`data_transforms.py` needs to be implemented"
    # )

    fund_aug_transforms = transforms.Compose([
        transforms.Resize((int(inp_size[0] * 1.1), int(inp_size[1] * 1.1))),
        transforms.RandomCrop(inp_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()])

    ###########################################################################
    # Student code end
    ###########################################################################
    return fund_aug_transforms


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model alomg with
    normalization.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fund_norm_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    fund_norm_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean, pixel_std)
    ])
    # raise NotImplementedError(
    #     "`get_fundamental_normalization_transforms` function in "
    #     + "`data_transforms.py` needs to be implemented"
    # )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fund_norm_transforms


def get_all_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set,
    along with normalization. This should just be your previous method + normalization.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    all_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    all_transforms = transforms.Compose([
        transforms.Resize((int(inp_size[0] * 1.1), int(inp_size[1] * 1.1))),
        transforms.RandomCrop(inp_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean, pixel_std)
    ])
    # raise NotImplementedError(
    #     "`get_all_transforms` function in "
    #     + "`data_transforms.py` needs to be implemented"
    # )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return all_transforms
