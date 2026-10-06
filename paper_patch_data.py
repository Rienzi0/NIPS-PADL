"""Paper-aligned CIFAR-100/SVHN patch loading.

The paper experiments use grayscale 8x8 non-overlapping patches sampled from
the official training and test splits.  This module implements that protocol
without materializing an entire image split in memory.

The high-level entry point is :func:`load_paper_patch_splits`.  Tests and
custom datasets can use :func:`load_patch_splits_from_datasets` directly.
Both return matrices in dictionary-learning convention, ``[d, N]``.

This file intentionally uses Python 3.8-compatible typing and APIs.
"""

import random
from collections import defaultdict
from typing import DefaultDict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


TensorOrTensors = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _extract_grayscale_tensor(item: object) -> torch.Tensor:
    """Extract a CPU ``[1, H, W]`` tensor from a dataset item.

    The torchvision datasets used by :func:`load_paper_patch_splits` apply a
    grayscale-plus-ToTensor transform.  Keeping this validation here prevents
    silent mixing of color and grayscale experiment protocols.
    """

    if isinstance(item, (tuple, list)):
        if not item:
            raise ValueError("Dataset returned an empty tuple/list.")
        image = item[0]
    else:
        image = item

    if not torch.is_tensor(image):
        raise TypeError(
            "Dataset images must be torch tensors. Apply Grayscale(1) and "
            "ToTensor() before sampling."
        )
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(
            "Expected an image with shape [H, W] or [1, H, W], got %r."
            % (tuple(image.shape),)
        )
    if image.shape[0] != 1:
        raise ValueError(
            "Expected a grayscale image with one channel, got %d channels."
            % int(image.shape[0])
        )
    return image.detach().cpu()


def image_to_nonoverlapping_patches(
    image: torch.Tensor,
    patch_size: int = 8,
) -> torch.Tensor:
    """Convert one grayscale image to non-overlapping patch columns.

    Args:
        image: Tensor shaped ``[H, W]`` or ``[1, H, W]``.
        patch_size: Square patch edge length.

    Returns:
        Tensor shaped ``[patch_size ** 2, patches_per_image]``.  Patch columns
        are ordered left-to-right and then top-to-bottom.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    image = _extract_grayscale_tensor(image)
    _, height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "Non-overlapping patches require image dimensions divisible by "
            "patch_size; got H=%d, W=%d, patch_size=%d."
            % (height, width, patch_size)
        )

    # unfold returns [batch, channels * patch_size**2, number_of_patches].
    return F.unfold(
        image.unsqueeze(0),
        kernel_size=patch_size,
        stride=patch_size,
    ).squeeze(0)


def sample_global_patch_ids(
    total_patches: int,
    n_patches: int,
    seed: int,
) -> Tuple[int, ...]:
    """Sample unique global patch IDs without allocating ``total_patches`` IDs."""

    if total_patches <= 0:
        raise ValueError("total_patches must be positive.")
    if n_patches <= 0:
        raise ValueError("n_patches must be positive.")
    if n_patches > total_patches:
        raise ValueError(
            "Requested %d patches, but the split contains only %d "
            "non-overlapping patches." % (n_patches, total_patches)
        )
    # random.sample(range(...)) is O(n_patches) memory for a range population.
    rng = random.Random(int(seed))
    return tuple(rng.sample(range(total_patches), n_patches))


def sample_nonoverlapping_patches(
    dataset: Sequence[object],
    n_patches: int,
    patch_size: int = 8,
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
    return_global_ids: bool = False,
) -> TensorOrTensors:
    """Uniformly sample patch columns from a fixed-size image dataset.

    Sampling is without replacement over all non-overlapping patch positions
    in the split.  Only images that contain selected patches are fetched.

    Args:
        dataset: Random-access dataset returning an image or ``(image, target)``.
        n_patches: Exact number of patch columns to return.
        patch_size: Square, non-overlapping patch size.
        seed: Reproducible sampling seed.
        device: Destination device for the returned matrix.
        return_global_ids: Also return sampled global patch IDs as an int64
            tensor.  This is useful for experiment manifests and tests.

    Returns:
        ``X`` with shape ``[patch_size**2, n_patches]``.  If
        ``return_global_ids=True``, returns ``(X, global_ids)``.
    """

    dataset_size = len(dataset)
    if dataset_size <= 0:
        raise ValueError("Dataset split is empty.")

    probe_image = _extract_grayscale_tensor(dataset[0])
    probe_patches = image_to_nonoverlapping_patches(
        probe_image,
        patch_size=patch_size,
    )
    patch_dimension, patches_per_image = probe_patches.shape
    total_patches = dataset_size * int(patches_per_image)
    global_ids = sample_global_patch_ids(total_patches, n_patches, seed)

    # Map each requested patch to its image.  The output position preserves the
    # pseudorandom sample order while each selected image is decoded only once.
    requests: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
    for output_position, global_patch_id in enumerate(global_ids):
        image_index, patch_index = divmod(
            int(global_patch_id),
            int(patches_per_image),
        )
        requests[image_index].append((output_position, patch_index))

    output = torch.empty(
        (int(patch_dimension), int(n_patches)),
        dtype=probe_patches.dtype,
        device="cpu",
    )
    for image_index in sorted(requests):
        if image_index == 0:
            patches = probe_patches
        else:
            patches = image_to_nonoverlapping_patches(
                _extract_grayscale_tensor(dataset[image_index]),
                patch_size=patch_size,
            )
        if patches.shape != probe_patches.shape:
            raise ValueError(
                "All images in a split must have the same patch layout; "
                "image %d produced %r instead of %r."
                % (
                    image_index,
                    tuple(patches.shape),
                    tuple(probe_patches.shape),
                )
            )
        for output_position, patch_index in requests[image_index]:
            output[:, output_position] = patches[:, patch_index]

    output = output.to(device)
    if return_global_ids:
        return output, torch.tensor(global_ids, dtype=torch.int64)
    return output


def load_patch_splits_from_datasets(
    train_dataset: Sequence[object],
    test_dataset: Sequence[object],
    train_patches: int = 20000,
    test_patches: int = 2000,
    patch_size: int = 8,
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample paper-protocol matrices from explicit train and test datasets.

    The train split uses ``seed`` and the test split uses ``seed + 1``.  This
    keeps the two selections deterministic but independent.
    """

    train_x = sample_nonoverlapping_patches(
        train_dataset,
        n_patches=train_patches,
        patch_size=patch_size,
        seed=seed,
        device=device,
    )
    test_x = sample_nonoverlapping_patches(
        test_dataset,
        n_patches=test_patches,
        patch_size=patch_size,
        seed=seed + 1,
        device=device,
    )
    return train_x, test_x


def _build_torchvision_split(
    data_dir: str,
    dataset_name: str,
    train: bool,
    download: bool,
) -> Sequence[object]:
    """Build one torchvision split using APIs available in torchvision 0.9.1."""

    # Import lazily so synthetic unit tests do not import or initialize
    # torchvision and can never trigger a dataset download.
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    normalized_name = dataset_name.lower().replace("_", "").replace("-", "")
    if normalized_name == "cifar100":
        return datasets.CIFAR100(
            root=data_dir,
            train=train,
            download=download,
            transform=transform,
        )
    if normalized_name == "svhn":
        return datasets.SVHN(
            root=data_dir,
            split="train" if train else "test",
            download=download,
            transform=transform,
        )
    raise ValueError(
        "Unsupported dataset %r. Expected 'cifar100' or 'svhn'."
        % dataset_name
    )


def load_paper_patch_splits(
    data_dir: str,
    dataset_name: str = "cifar100",
    train_patches: int = 20000,
    test_patches: int = 2000,
    patch_size: int = 8,
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
    download: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load official splits and return paper-protocol ``(train_X, test_X)``.

    Defaults reproduce the paper data geometry: grayscale 8x8
    non-overlapping patches, 20,000 training patches, and 2,000 test patches.
    Dataset files are not downloaded unless ``download=True`` is explicit.
    """

    train_dataset = _build_torchvision_split(
        data_dir=data_dir,
        dataset_name=dataset_name,
        train=True,
        download=download,
    )
    test_dataset = _build_torchvision_split(
        data_dir=data_dir,
        dataset_name=dataset_name,
        train=False,
        download=download,
    )
    return load_patch_splits_from_datasets(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_patches=train_patches,
        test_patches=test_patches,
        patch_size=patch_size,
        seed=seed,
        device=device,
    )


__all__ = [
    "image_to_nonoverlapping_patches",
    "load_paper_patch_splits",
    "load_patch_splits_from_datasets",
    "sample_global_patch_ids",
    "sample_nonoverlapping_patches",
]
