import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_dataloaders(cfg):
    dataset_name = cfg["data"]["dataset"].lower()
    data_dir = cfg["data"].get("data_dir", "./data")
    batch_size = cfg["data"].get("batch_size", 128)
    num_workers = cfg["data"].get("num_workers", 0)
    val_split = cfg["data"].get("val_split", 0.2)
    seed = cfg["experiment"].get("seed", 0)

    if dataset_name == "cifar10":
        dataset_cls = datasets.CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "cifar100":
        dataset_cls = datasets.CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = dataset_cls(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = dataset_cls(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = int(len(full_train) * 0.8)   # 40000
    val_size   = len(full_train) - train_size # 10000
    train_set, val_set = random_split(full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))


    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")


    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader