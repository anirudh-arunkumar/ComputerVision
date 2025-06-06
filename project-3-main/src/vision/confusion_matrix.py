from typing import Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from vision.image_loader import ImageLoader
from torch import nn
from torch.utils.data import DataLoader, Subset


def generate_confusion_data(
    model: nn.Module,
    dataset: ImageLoader,
    use_cuda: bool = False,
) -> Tuple[Sequence[int], Sequence[int], Sequence[str]]:
    """
    Get the accuracy on the val/train dataset

    Args:
    -   model: Model to generate confusion matrix data for
    -   dataset: The ImageLoader dataset that corresponds to training or val data
    -   use_cuda: Whether to evaluate on CPU or GPU

    Returns:
    -   targets: a numpy array of shape (N) containing the targets indices
    -   preds: a numpy array of shape (N) containing the predicted indices
    -   class_labels: A list containing the class labels at the index of their label_number
                      e.g. if the labels are {"Cat": 0, "Monkey": 2, "Dog": 1},
                           the return value should be ["Cat", "Dog", "Monkey"]
    """

    batch_size = 32
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataloader_args = {"num_workers": 1, "pin_memory": True} if device.type != "cpu" else {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **dataloader_args)

    preds = torch.zeros(len(dataset), dtype=torch.int32, device=device)
    targets = torch.zeros(len(dataset), dtype=torch.int32, device=device)
    label_to_idx = dataset.get_classes()
    class_labels = [None] * len(label_to_idx)

    for label_name, i in label_to_idx.items():
        class_labels[i] = label_name
    model = model.to(device)
    model.eval()
    ##########################################################################
    # Student code begins here
    ##########################################################################

    # raise NotImplementedError(
    #     "`generate_confusion_data` function in "
    #     + "`confusion_matrix.py` needs to be implemented"
    # )
    
    counter = 0

    with torch.no_grad():
        for img, labels, in loader:

            img = img.to(device).float()
            labels = labels.to(device)
            outputs = model(img)
            pred = outputs.argmax(dim=1)
            actual_size = labels.size(0)
            targets[counter:counter+actual_size] = labels
            preds[counter:counter+actual_size] = pred
            counter += actual_size

    ##########################################################################
    # Student code ends here
    ##########################################################################
    model.train()

    return targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), class_labels


def generate_confusion_matrix(
    targets: np.ndarray, preds: np.ndarray, num_classes: int, normalize=True
) -> np.ndarray:
    """Generate the actual confusion matrix values

    The confusion matrix is a num_classes x num_classes matrix that shows the
    number of classifications made to a predicted class, given a ground truth class

    If the classifications are:
        ground_truths: [1, 0, 1, 2, 0, 1, 0, 2, 2]
        predicted:     [1, 1, 0, 2, 0, 1, 1, 2, 0]

    Then the confusion matrix is:
        [1 2 0],
        [1 2 0],
        [1 0 2],

    Each ground_truth value corresponds to a row,
    and each predicted value is a column

    A confusion matrix can be normalized by dividing all the entries of
    each ground_truth prior by the number of actual instances of the ground truth
    in the dataset.

    Args:
    -   targets: a numpy array of shape (N) containing the targets indices
    -   preds: a numpy array of shape (N) containing the predicted indices
    -   num_classes: Number of classes in the confusion matrix
    -   normalize: Whether to normalize the confusion matrix or not
    Returns:
    -   confusion_matrix: a (num_classes, num_classes) numpy array
                          representing the confusion matrix
    """

    confusion_matrix = np.zeros((num_classes, num_classes))

    for target, prediction in zip(targets, preds):
        ##########################################################################
        # Student code begins here
        ##########################################################################
    
        # raise NotImplementedError(
        #     "`generate_confusion_matrix` function in "
        #     + "`confusion_matrix.py` needs to be implemented"
        # )
        confusion_matrix[target, prediction] += 1
        
        ##########################################################################
        # Student code ends here
        ##########################################################################

    if normalize:
        ##########################################################################
        # Student code begins here
        ##########################################################################
    
        # raise NotImplementedError(
        #     "`generate_confusion_matrix` function in "
        #     + "`confusion_matrix.py` needs to be implemented"
        # )

        for i in range(num_classes):
            sum_of_rows = confusion_matrix[i].sum()
            if sum_of_rows > 0:
                confusion_matrix[i] = confusion_matrix[i] / sum_of_rows
    
        ##########################################################################
        # Student code ends here
        ##########################################################################
    return confusion_matrix


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_labels: Sequence[str]
) -> None:
    """Plots the confusion matrix

    Args:
    -   confusion_matrix: a (num_classes, num_classes) numpy array
                          representing the confusion matrix
    -   class_labels: A list containing the class labels at the index of their label_number
                      e.g. if the labels are {"Cat": 0, "Monkey": 2, "Dog": 1},
                           the return value should be ["Cat", "Dog", "Monkey"]
                      The length of class_labels should be num_classes
    """
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    num_classes = len(class_labels)

    ax.imshow(confusion_matrix, cmap="Blues")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-Truth label")
    ax.set_title("Confusion Matrix")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            _ = ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.show()


def generate_and_plot_confusion_matrix(
    model: nn.Module, dataset: ImageLoader, use_cuda: bool = False
) -> None:
    """Runs the entire confusion matrix pipeline for convenience

    Args:
    -   model: Model to generate confusion matrix data for
    -   dataset: The ImageLoader dataset that corresponds to training or validation data
    -   use_cuda: Whether to evaluate on CPU or GPU
    """

    targets, predictions, class_labels = generate_confusion_data(
        model, dataset, use_cuda=use_cuda
    )

    confusion_matrix = generate_confusion_matrix(
        np.array(targets, dtype=np.int32),
        np.array(predictions, np.int32),
        len(class_labels),
    )

    plot_confusion_matrix(confusion_matrix, class_labels)


def get_pred_images_for_target(
    model: nn.Module,
    dataset: ImageLoader,
    predicted_class: int,
    target_class: int,
    use_cuda: bool = False,
) -> Sequence[str]:
    """Returns a list of image paths that correspond to a particular prediction
    for a given target class

    Args:
    -   model: Model to generate confusion matrix data for
    -   dataset: The ImageLoader dataset that corresponds to training or validation data
    -   predicted_class: The class predicted by the model
    -   target_class: The actual class of the image
    -   use_cuda: Whether to evaluate on CPU or GPU

    Returns:
    -   valid_image_paths: Image paths that are classified as <predicted_class>
                           but actually belong to <target_class>
    """
    model.eval()
    device = next(model.parameters()).device
    dataset_list = dataset.dataset
    indices = []
    image_paths = []
    for i, (image_path, class_label) in enumerate(dataset_list):
        if class_label == target_class:
            indices.append(i)
            image_paths.append(image_path)
    subset = Subset(dataset, indices)
    dataloader_args = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    loader = DataLoader(subset, batch_size=32, shuffle=False, **dataloader_args)
    preds = []
    for i, (inp, _) in enumerate(loader):
        if use_cuda:
            inp = inp.cuda()
        inp = inp.to(device)
        logits = model(inp)
        p = torch.argmax(logits, dim=1)
        preds.append(p)
    predictions = torch.cat(preds, dim=0).cpu().tolist()
    valid_image_paths = [
        image_paths[i] for i, p in enumerate(predictions) if p == predicted_class
    ]
    model.train()
    return valid_image_paths


def generate_accuracy_data(
    model: nn.Module,
    dataset: ImageLoader,
    num_attributes: int,
    use_cuda: bool = False,
) -> Tuple[Sequence[int], Sequence[int], Sequence[str]]:
    """
    Get the accuracy on the val/train dataset

    Args:
    -   model: Model to generate accuracy table data for
    -   dataset: The ImageLoader dataset that corresponds to training or val data
    -   num_attributes: number of attributes to predict per image = k
    -   use_cuda: Whether to evaluate on CPU or GPU

    Returns:
    -   targets: a numpy array of shape (N, k) containing the target labels
    -   preds: a numpy array of shape (N, k) containing the predicted labels
    """

    batch_size = 32
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataloader_args = {"num_workers": 1, "pin_memory": True} if device.type != "cpu" else {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **dataloader_args)
    
    preds = torch.zeros((len(dataset), num_attributes), dtype=torch.int32, device=device)
    targets = torch.zeros((len(dataset), num_attributes), dtype=torch.int32, device=device)
    # label_to_idx = dataset.get_classes()
    # class_labels = [""] * len(label_to_idx)
    model = model.to(device)
    model.eval()
    ##########################################################################
    # Student code begins here
    ##########################################################################

    # raise NotImplementedError(
    #         "`generate_accuracy_data` function in "
    #         + "`confusion_matrix.py` needs to be implemented"
    #     )

    temp = 0

    with torch.no_grad():

        for img, labels in loader:
            img = img.to(device).float()
            labels = labels.to(device)
            logits = model(img)
            pred = (logits > 0).int()
            batch = labels.size(0)
            targets[temp:temp+batch] = labels
            preds[temp:temp+batch] = pred
            temp += batch
        
    ##########################################################################
    # Student code ends here
    ##########################################################################
    model.train()

    return targets.cpu().numpy(), preds.cpu().numpy()


def generate_accuracy_table(
    targets: np.ndarray, preds: np.ndarray, num_attributes: int
) -> np.ndarray:
    """Generate the actual accuracy table values

    The accuracy table is a (num_attributes, ) array that shows the
    number of classifications made to a predicted attribute, given a ground truth 
    label of attributes

    
    If the classifications are:
        ground_truths: [[1, 0, 1],
                        [0, 0, 1],
                        [1, 0, 0]]
        predicted:     [[0, 0, 1],
                        [0, 1, 1],
                        [1, 0, 1]]

    Then the accuracy table is:
        [0.67 0.67 0.67]

    Args:
    -   targets: a numpy array of shape (N, k) containing the targets attributes
    -   preds: a numpy array of shape (N, k) containing the predicted attributes
    -   num_attributes: Number of attributes in the accuracy table
    Returns:
    -   accuracy_table: a (num_attributes, ) numpy array
                          representing the accuracy table
    """

    accuracy_table = np.zeros(num_attributes)

    ##########################################################################
    # Student code begins here
    ##########################################################################

    # raise NotImplementedError(
    #         "`generate_accuracy_table` function in "
    #         + "`confusion_matrix.py` needs to be implemented"
    #     )

    accuracy_table = np.zeros(num_attributes, dtype=np.float32)
    for i in range(num_attributes):
        calc = np.sum(preds[:, i] == targets[:, i])
        accuracy_table[i] = calc / float(targets.shape[0])

    ##########################################################################
    # Student code ends here
    ##########################################################################

    return accuracy_table


def plot_accuracy_table(
    accuracy_table: np.ndarray, attribute_labels: Sequence[str]
) -> None:
    """Plots the accuracy table

    Args:
    -   accuracy table: a (num_attributes, ) numpy array
                          representing the accuracy table
    -   attribute_labels: A list containing the attribute labels
                        The length of attribute_labels should be num_attributes
    """
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    num_att = len(attribute_labels)

    ax.imshow(accuracy_table[np.newaxis, :], cmap="Blues")

    ax.set_xticks(np.arange(num_att))
    ax.set_xticklabels(attribute_labels)

    ax.set_xlabel("Attributes")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Table")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_att):
        _ = ax.text(
            i,
            0,
            f"{accuracy_table[i]:.2f}",
            ha="center",
            va="center",
            color="black",
        )

    plt.show()


def generate_and_plot_accuracy_table(
    model: nn.Module, 
    dataset: ImageLoader, 
    num_attributes = int, 
    attribute_labels = Sequence[str],
    use_cuda: bool = False
) -> None:
    """Runs the entire accuracy table pipeline for convenience

    Args:
    -   model: Model to generate confusion matrix data for
    -   dataset: The ImageLoader dataset that corresponds to training or validation data
    -   num_attributes: Number of attributes in the accuracy table
    -   attribute_labels: list of attribute names
    -   use_cuda: Whether to evaluate on CPU or GPU
    """

    targets, predictions = generate_accuracy_data(
        model, dataset, num_attributes, use_cuda=use_cuda
    )

    accuracy_table = generate_accuracy_table(
        np.array(targets, dtype=np.int32),
        np.array(predictions, np.int32),
        num_attributes
    )

    plot_accuracy_table(accuracy_table, attribute_labels)