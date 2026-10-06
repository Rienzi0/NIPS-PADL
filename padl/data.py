"""The result-producing protocol: shuffled training images and random crops."""

import numpy as np
import torch
from torchvision import datasets, transforms


def load_patches(data_dir, dataset_name, n_samples, patch_size, device,
                 download=False, patches_per_image=10):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=data_dir, train=True,
                                   download=download, transform=transform)
    elif dataset_name == "svhn":
        dataset = datasets.SVHN(root=data_dir, split="train",
                               download=download, transform=transform)
    else:
        raise ValueError("Dataset must be cifar100 or svhn")

    loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=True)
    images = next(iter(loader))[0].to(device)
    patches = []
    for image in images:
        _, height, width = image.shape
        if height <= patch_size or width <= patch_size:
            raise ValueError("Images must be larger than the patch size")
        for _ in range(patches_per_image):
            # Preserve the historical exclusive bounds and NumPy RNG draw order.
            top = np.random.randint(0, height - patch_size)
            left = np.random.randint(0, width - patch_size)
            patches.append(image[:, top:top + patch_size, left:left + patch_size].flatten())
    return torch.stack(patches, dim=1).to(device)
