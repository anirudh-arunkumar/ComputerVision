from typing import Tuple

import torch
from torch import nn


class PointNet(nn.Module):
    '''
    A simplified version of PointNet (https://arxiv.org/abs/1612.00593)
    Ignoring the transforms and segmentation head.
    '''
    def __init__(self,
        classes: int,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        classifier_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet to define layers.

        Hint: See the modified PointNet architecture diagram from the pdf.
        You will need to repeat the first hidden dim (see mlp(64, 64) in the diagram).
        Furthermore, you will want to include a BatchNorm1d after each layer in the encoder
        except for the final layer for easier training.

        Args:
        -   classes: Number of output classes
        -   in_dim: Input dimensionality for points. This parameter is 3 by default for
                    for the basic PointNet.
        -   hidden_dims: The dimensions of the encoding MLPs.
        -   classifier_dims: The dimensions of classifier MLPs.
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.encoder_head = None
        self.classifier_head = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`part3_pointnet.py` needs to be implemented"
        # )

        # self.encoder_head == nn.Sequential(
        #     nn.Conv2d(in_dim, hidden_dims[0], 1),
        #     nn.BatchNorm1d(hidden_dims[0]),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_dims[0], hidden_dims[1], 1),
        #     nn.BatchNorm1d(hidden_dims[1]),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dims[1], hidden_dims[2], 1),
        #     nn.BatchNorm1d(hidden_dims[2]),
        #     nn.ReLU(),
        # )

        dimensions = [
            in_dim,
            hidden_dims[0],
            hidden_dims[0],
            hidden_dims[1],
            hidden_dims[2]
        ]


        e_layers = []
        for i in range(4):
            e_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))

            if i < 3:
                e_layers.append(nn.BatchNorm1d(dimensions[i+1]))
                e_layers.append(nn.ReLU())
        self.encoder_head = nn.Sequential(*e_layers)

        prev = hidden_dims[2]
        class_layers = []
        for d in classifier_dims:
            class_layers.append(nn.Linear(prev, d))
            class_layers.append(nn.ReLU())
            class_layers.append(nn.Dropout(p=0.3))
            prev = d
        
        class_layers.append(nn.Linear(prev, classes))
        self.classifier_head = nn.Sequential(*class_layers)
        ############################################################################
        # Student code end
        ############################################################################


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Args:
            x: tensor of shape (B, N, in_dim), where B is the batch size, N is the number of points per
               point cloud, and in_dim is the input point dimension

        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point
                       before global maximization. This will be used later for analysis.
        '''

        class_outputs = None
        encodings = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`part3_pointnet.py` needs to be implemented"
        # )

        B, N, i = x.shape

        flatt = x.view(B * N, -1)
        henc = self.encoder_head(flatt)
        encodings = henc.view(B, N, -1)

        glob_feat, i = encodings.max(dim=1)

        class_outputs = self.classifier_head(glob_feat)
        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, encodings
