import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def sample_input(
        num_classes: int,
        num_per_class: int,
        data_loader: DataLoader,
        device: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `num_per_class` input images for each category from the dataset
    :param num_classes:
    :param num_per_class:
    :param data_loader
    :return:
    """
    sampled_inputs = []
    sampled_labels = []

    if device is None:
        device = torch.device("cpu")

    slot = {i: num_per_class for i in range(num_classes)}
    # for input, label in tqdm(data_loader, desc="sampling", mininterval=1):
    for input, label in data_loader:
        input = input.to(device)
        label = label.to(device)

        if slot[label.item()] > 0:
            sampled_inputs.append(input)
            sampled_labels.append(label)
            slot[label.item()] -= 1
        if sum(slot.values()) == 0:
            break

    return torch.cat(sampled_inputs, dim=0), torch.cat(sampled_labels, dim=0)
