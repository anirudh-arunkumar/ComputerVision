"""
Utilities to be used along with the deep model
"""
from typing import Union

import torch
from vision.my_resnet import MyResNet18
from vision.simple_net import SimpleNet
from vision.simple_net_final import SimpleNetFinal
from vision.multilabel_resnet import MultilabelResNet18
from torch import nn


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K classes logits[k] (where 0 <= k < K) corresponds to the
                log-odds of class `k` being the correct one.
                Shape: (batch_size, num_classes)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size)
    Returns:
        accuracy: The accuracy of the predicted logits
                   (number of correct predictions / total number of examples)
    """
    batch_accuracy = 0.0
    ############################################################################
    # Student code begin
    ############################################################################

    # raise NotImplementedError(
    #     "`compute_accuracy` function in "
    #     + "`dl_utils.py` needs to be implemented"
    # )
    batch_accuracy = ((torch.argmax(logits, dim=1) == labels).sum().item()) / labels.shape[0]

    ############################################################################
    # Student code end
    ############################################################################

    return batch_accuracy


def compute_loss(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, MultilabelResNet18],
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the loss between the model output and the target labels

    Args:
    -   model: a model (which inherits from nn.Module)
    -   model_output: the raw scores output by the net
    -   target_labels: the ground truth class labels
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the loss value
    """
    loss = None

    ############################################################################
    # Student code begin
    ############################################################################

    # raise NotImplementedError(
    #     "`compute_loss` function in "
    #     + "`dl_utils.py` needs to be implemented"
    # )

    if is_normalize and model.loss_criterion.reduction == 'sum':
        loss = model.loss_criterion(model_output, target_labels) / target_labels.size(0)
    else:
        loss = model.loss_criterion(model_output, target_labels)

    ############################################################################
    # Student code end
    ############################################################################

    return loss

def compute_multilabel_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K labels logits[k] (where 0 <= k < K) corresponds to the
                log-odds of label `k` being present in the image.
                Shape: (batch_size, num_labels)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size, num_labels)
    Returns:
        accuracy: The accuracy of the predicted logits
                  (number of correct predictions / total number of labels)
    """
    batch_accuracy = 0.0
    ############################################################################
    # Student code begin
    ############################################################################

    # raise NotImplementedError(
    #     "`compute_multilabel_accuracy` function in "
    #     + "`dl_utils.py` needs to be implemented"
    # )

    pred = (logits > 0.5).float()
    corr = (pred == labels).float().sum()
    batch_accuracy = corr / labels.numel()
    batch_accuracy = batch_accuracy.item()

    ############################################################################
    # Student code end
    ############################################################################

    return batch_accuracy


def save_trained_model_weights(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, MultilabelResNet18], out_dir: str
) -> None:
    """Saves the weights of a trained model along with class name

    Args:
    -   model: The model to be saved
    -   out_dir: The path to the folder to store the save file in
    """
    class_name = model.__class__.__name__
    state_dict = model.state_dict()

    assert class_name in set(
        ["SimpleNet", "SimpleNetFinal", "MyResNet18", "MultilabelResNet18"]
    ), "Please save only supported models"

    save_dict = {"class_name": class_name, "state_dict": state_dict}
    torch.save(save_dict, f"{out_dir}/trained_{class_name}_final.pt")
