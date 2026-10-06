import unittest

import torch

try:
    from paper_patch_data import (
        image_to_nonoverlapping_patches,
        load_patch_splits_from_datasets,
        sample_global_patch_ids,
        sample_nonoverlapping_patches,
    )
except ImportError:
    from work.new_runner.paper_patch_data import (
        image_to_nonoverlapping_patches,
        load_patch_splits_from_datasets,
        sample_global_patch_ids,
        sample_nonoverlapping_patches,
    )


class SyntheticImageDataset(object):
    """Random-access fixed-size grayscale dataset with access accounting."""

    def __init__(self, size, offset=0.0):
        self.size = int(size)
        self.offset = float(offset)
        self.accessed = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = int(index)
        self.accessed.append(index)
        image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        image = image + self.offset + 16.0 * index
        return image, index


class PaperPatchDataTests(unittest.TestCase):
    def test_nonoverlapping_patch_order_and_shape(self):
        image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        patches = image_to_nonoverlapping_patches(image, patch_size=2)
        expected = torch.tensor(
            [
                [0.0, 2.0, 8.0, 10.0],
                [1.0, 3.0, 9.0, 11.0],
                [4.0, 6.0, 12.0, 14.0],
                [5.0, 7.0, 13.0, 15.0],
            ]
        )
        self.assertEqual(tuple(patches.shape), (4, 4))
        self.assertTrue(torch.equal(patches, expected))

    def test_sampling_is_exact_unique_and_reproducible(self):
        dataset_a = SyntheticImageDataset(1000)
        x_a, ids_a = sample_nonoverlapping_patches(
            dataset_a,
            n_patches=17,
            patch_size=2,
            seed=23,
            return_global_ids=True,
        )
        dataset_b = SyntheticImageDataset(1000)
        x_b, ids_b = sample_nonoverlapping_patches(
            dataset_b,
            n_patches=17,
            patch_size=2,
            seed=23,
            return_global_ids=True,
        )

        self.assertEqual(tuple(x_a.shape), (4, 17))
        self.assertEqual(ids_a.numel(), 17)
        self.assertEqual(torch.unique(ids_a).numel(), 17)
        self.assertTrue(torch.equal(ids_a, ids_b))
        self.assertTrue(torch.equal(x_a, x_b))

    def test_sampling_does_not_materialize_full_dataset(self):
        dataset = SyntheticImageDataset(1000)
        sample_nonoverlapping_patches(
            dataset,
            n_patches=10,
            patch_size=2,
            seed=7,
        )
        # At most one probe plus one fetch per selected image; in particular,
        # this is independent of the 1000-image split size.
        self.assertLessEqual(len(dataset.accessed), 11)
        self.assertLess(len(dataset.accessed), len(dataset))

    def test_different_seed_changes_sample(self):
        ids_a = sample_global_patch_ids(4000, 20, seed=1)
        ids_b = sample_global_patch_ids(4000, 20, seed=2)
        self.assertNotEqual(ids_a, ids_b)

    def test_train_and_test_splits_have_exact_requested_counts(self):
        train_dataset = SyntheticImageDataset(100, offset=0.0)
        test_dataset = SyntheticImageDataset(100, offset=100000.0)
        train_x, test_x = load_patch_splits_from_datasets(
            train_dataset,
            test_dataset,
            train_patches=13,
            test_patches=7,
            patch_size=2,
            seed=11,
        )
        self.assertEqual(tuple(train_x.shape), (4, 13))
        self.assertEqual(tuple(test_x.shape), (4, 7))
        self.assertLess(float(train_x.max()), 100000.0)
        self.assertGreaterEqual(float(test_x.min()), 100000.0)

    def test_request_larger_than_split_is_rejected(self):
        dataset = SyntheticImageDataset(2)
        with self.assertRaisesRegex(ValueError, "contains only"):
            sample_nonoverlapping_patches(
                dataset,
                n_patches=9,
                patch_size=2,
                seed=0,
            )

    def test_nondivisible_image_is_rejected(self):
        image = torch.zeros(1, 5, 4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            image_to_nonoverlapping_patches(image, patch_size=2)


if __name__ == "__main__":
    unittest.main()
