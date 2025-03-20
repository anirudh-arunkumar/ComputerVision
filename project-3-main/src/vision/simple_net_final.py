import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super(SimpleNetFinal, self).__init__()

        # self.conv_layers = nn.Sequential()
        # self.fc_layers = nn.Sequential()
        # self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`simple_net_final.py` needs to be implemented"
        # )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=15, kernel_size=5, padding=2),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc_layers = nn.Sequential(nn.Linear(in_features=32*3*3, out_features=128), nn.ReLU(), nn.Linear(in_features=128, out_features=15))
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        for m in self.conv_layers.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        
        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`simple_net_final.py` needs to be implemented"
        # )

        model_output = self.conv_layers(x)
        model_output = model_output.view(model_output.size(0), -1)
        model_output = self.fc_layers(model_output)
        
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
