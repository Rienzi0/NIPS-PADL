import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch
import numpy as np

def load_custom_dataset(data_dir="data", dataset_name='cifar100', n_samples=5000, patch_size=8, device="cpu", download=False):

    """
    加载自定义数据集并提取图像块
    dataset_name: 可选 'mnist', 'fashion-mnist', 'cifar100', 'svhn', 'stl10'
    """
    # 定义转换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 如果不是MNIST或Fashion-MNIST，且想用灰度，取消注释
        transforms.ToTensor()
    ])

    # 根据数据集名称加载数据
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    elif dataset_name == 'fashion-mnist':
        dataset = datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transform)
    elif dataset_name == 'svhn':
        dataset = datasets.SVHN(root=data_dir, split='train', download=download, transform=transform)
    elif dataset_name == 'stl10':
        # STL-10 的 'unlabeled' split 包含大量未标注数据，适合无监督重建
        dataset = datasets.STL10(root=data_dir, split='unlabeled', download=download, transform=transform)
    else:
        raise ValueError("Unknown dataset name")

    # 创建DataLoader并取部分样本
    loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=True)
    data_iter = iter(loader)
    
    # 使用Python内置的next()函数而不是.next()方法
    data = next(data_iter)
    
    # 处理不同数据集的返回格式
    if dataset_name == 'svhn':
        images = data[0]  # SVHN返回的是元组 (data, target)
    else:
        images = data[0] if isinstance(data, (list, tuple)) else data
    
    images = images.to(device)

    # 提取patch (随机裁剪小块)
    patches = []
    for img in images:
        # 确保图像有正确的形状 [C, H, W]
        if len(img.shape) == 2:  # 如果只有 [H, W]
            img = img.unsqueeze(0)  # 添加通道维度 [1, H, W]
        
        C, H, W = img.shape
        
        for _ in range(10):  # 每张图取几个 patch
            if H <= patch_size or W <= patch_size:
                continue  # 跳过太小的图像
                
            top = np.random.randint(0, H - patch_size)
            left = np.random.randint(0, W - patch_size)
            patch = img[:, top:top+patch_size, left:left+patch_size]
            patches.append(patch.flatten())

    patches = torch.stack(patches, dim=1)  # [d, n_samples]
    return patches.to(device)
