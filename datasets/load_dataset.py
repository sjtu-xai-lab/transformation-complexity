from .cifar10 import *


def load_dataset(
        dataset: str,
        data_root: str,
        batch_size: int,
        num_workers: int = 4,
        shuffle_train: bool = True
):
    if dataset == "cifar10":
        return load_cifar10(
            data_root=data_root, batch_size=batch_size,
            num_workers=num_workers, shuffle_train=shuffle_train
        )
    else:
        raise NotImplementedError