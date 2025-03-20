import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MultilabelResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Consider which activation function to use
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.activation = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`multi_resnet.py` needs to be implemented"
        # )

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        for p in resnet.parameters():
            p.requires_grad = False

        counter = resnet.fc.in_features
        resnet.fc = nn.Linear(counter, 7)

        for p in resnet.fc.parameters():
            p.requires_grad = True

        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_layers = resnet.fc
        self.activation = nn.Sigmoid()
        self.loss_criterion = nn.BCELoss(reduction="mean")

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        ############################################################################
        # Student code begin
        ############################################################################
        
        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`multi_resnet.py` needs to be implemented"
        # )

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        model_output = self.fc_layers(x)
        model_output = self.activation(model_output)
        ############################################################################
        # Student code end
        ############################################################################
        return model_output
