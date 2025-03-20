import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    # raise NotImplementedError(
    #         "`compute_mean_and_std` function in "
    #         + "`stats_helper.py` needs to be implemented"
    #     )

    pattern = os.path.join(dir_name, '**', '*.jpg')
    files = glob.glob(pattern, recursive=True)
    sq_sum = 0
    count = 0
    sum_total = 0
    for f in files:
        img = Image.open(f).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        sum_total += arr.sum()
        sq_sum += np.sum(arr ** 2)
        count += arr.size
    
    mean = sum_total / count
    var = (sq_sum - count * mean ** 2) / (count - 1)
    std = np.sqrt(var)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
