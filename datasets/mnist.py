import torch
import torch.utils.data
from torchvision import datasets, transforms


def load_mnist(
        data_root: str,
        batch_size: int,
        num_workers: int = 4,
        shuffle_train: bool = True,
        data_aug_train: bool = True,
):
    if data_aug_train:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train,
                                               num_workers=num_workers, drop_last=True)
    test_set = datasets.MNIST(root=data_root, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(
        data_root="/data2/limingjie/data/Vision",
        batch_size=64,
    )
    _train_loader, _test_loader = load_mnist(args)

    for image, label in _train_loader:
        print(image.shape, label.shape)
        exit()