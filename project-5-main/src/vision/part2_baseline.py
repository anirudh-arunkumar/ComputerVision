from typing import Tuple

import torch
from torch import nn


class Baseline(nn.Module):
    '''
    A simple baseline that counts points per voxel in the point cloud
    and then uses a linear classifier to make a prediction
    '''
    def __init__(self,
        classes: int,
        voxel_resolution=4,
        mode="count"
    ) -> None:
        '''
        Constructor for Baseline to define layers.

        Args:
        -   classes: Number of output classes
        -   voxel_resolution: Number of positions per dimension to count
        -   mode: Whether to count the number of points per voxel ("count") or just check binary occupancy ("occupancy")
        '''
        assert mode in ["count", "occupancy"]

        super().__init__()

        self.classifier = None
        self.voxel_resolution = voxel_resolution
        self.mode = mode

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`part2_baseline.py` needs to be implemented"
        # )

        input_features = voxel_resolution ** 3
        self.classifier = nn.Linear(input_features, classes)

        ############################################################################
        # Student code end
        ############################################################################


    def count_points(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Create the feature as input to the linear classifier by counting the number of points per voxel.
        This is effectively taking a 3D histogram for every item in a batch.

        Hint: 
        1) torch.histogramdd will be useful here

        Args:
        -   x: tensor of shape (B, N, in_dim)

        Output:
        -   counts: tensor of shape (B, voxel_resolution**3), indicating the percentage of points that landed in each voxel
        '''

        counts = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`count_points` function in "
        #     + "`part2_baseline.py` needs to be implemented"
        # )

        B, N, dim = x.shape
        features = []
        res = self.voxel_resolution

        for b in range(B):
            pts = x[b]

            maxes = pts.max(dim=0).values
            mins = pts.min(dim=0).values

            ranges = []

            for i in range(dim):
                min_val = mins[i].item()
                max_val = maxes[i].item()
                ranges.append((min_val, max_val))
            
            h, i = torch.histrogramdd(sample=pts, bins=[res, res, res], range=ranges, density=False)

            histogram = h.flatten()
            histogram = histogram / N
            features.append(histogram)
        
        counts = torch.stack(features, dim=0)
        ############################################################################
        # Student code end
        ############################################################################

        return counts


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the Baseline model. Make sure you handle the case where the mode 
        is set to "occupancy" by thresholding the result of count_points on zero.

        Args:
            x: tensor of shape (B, N, 3), where B is the batch size and N is the number of points per
               point cloud
        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   None, just used to allow reuse of training infrastructure
        '''

        class_outputs = None

        ############################################################################
        # Student code begin
        ############################################################################

        raise NotImplementedError(
            "`forward` function in "
            + "`part2_baseline.py` needs to be implemented"
        )

        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, None
