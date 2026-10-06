# NIPS-PADL

Fixed-configuration CIFAR-100 and SVHN reconstruction experiments, trained from
scratch. The two recipes preserve the previously verified direct-gated model,
random-patch sampling and numerical environment.

## Run

Use Python 3.12 and the pinned dependencies:

```bash
git clone https://github.com/Rienzi0/NIPS-PADL.git
cd NIPS-PADL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
python reproduce.py --download --device mps --output outputs/cifar100-seed57
python reproduce.py --dataset svhn --download --device mps --output outputs/svhn-seed57
```

The recorded results use Apple MPS, PyTorch 2.13.0 and torchmetrics 1.9.0.
`--device auto` selects CUDA, then MPS, then CPU. CPU/CUDA executions save their
actual results and compare them to the MPS reference; matching performance on
those devices has not been established. Changing library versions can change
optimization and SSIM results.

To use existing datasets, omit `--download` and add `--data-dir /path/to/data`.
Downloads are opt-in. SVHN uses `train_32x32.mat`; torchvision verifies the
official checksum. Every run requires a new output directory.

## Recorded results

| Dataset | Pixel RMSE | Mosaic PSNR | Mosaic SSIM | Mean coefficients per patch |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | 0.099168 | 20.188501 | 0.603126 | 13.475 |
| SVHN | 0.086381 | 21.078730 | 0.618446 | 10.450 |

Each row is one seed-57 training run. Both have 128 globally active rows;
per-patch coefficient counts do not establish the separate 19-atom cost claim.

The complete defaults and reference fingerprints are in
[configs/cifar100.json](configs/cifar100.json) and
[configs/svhn.json](configs/svhn.json). These files directly configure the shared
runner: K=128, Beta(1,3.5) initialization, 400 training iterations, 20 coefficient
and 10 dictionary updates per iteration, and 100 prediction steps. Delta is
0.44 for CIFAR-100 and 0.49 for SVHN.

## Protocol and scope

Each recipe samples 200 shuffled images from the dataset's **training partition**
and ten random grayscale 8×8 crops per image. The first 1,800 patches train the
dictionary; the final 200 evaluate it. Crop coordinates preserve the historical
exclusive upper bound `image_size - patch_size`. The RNG is seeded once, before
sampling, and continues through training and evaluation.

RMSE uses all 200 evaluation patches in the original pixel scale. PSNR and SSIM
arrange consecutive crops into three 64×64 mosaics, omit the final eight patches,
and use the joint ground-truth/reconstruction intensity range. These are mosaic
metrics; the random crops do not reconstruct complete source images.

The released model uses sigmoid coefficients, soft elementwise gates during
training, hard gates during evaluation, and column-max regularization. This is
the historical direct-gated variant, which differs from the paper's Algorithm 1.
The recipes reproduce the recorded cases above. They do not reproduce Table 2's
ten-run summaries under its stated official-split, non-overlapping-patch protocol.

## Files and verification

```text
reproduce.py           CLI, reference checks and result saving
padl/model.py          direct-gated dictionary learning
padl/data.py           dataset loading and random patches
padl/experiment.py     shared training and metric evaluation
configs/               the two fixed recipes
tests/                 numerical and output-integrity checks
```

Each run saves `metrics.csv`, `matched_evaluation.npz` and `manifest.json`,
including matched dictionary/coefficients/data, runtime versions, source hashes,
fingerprints and metric differences. Data or metric mismatches return a nonzero
exit code and retain the measurements. Absolute comparison tolerances are
0.0005 RMSE, 0.05 PSNR and 0.005 mosaic SSIM; data fingerprints must match exactly.

```bash
python -m unittest discover -s tests -v
```

Code is released under the [MIT License](LICENSE). Dataset images are downloaded
separately and are not bundled.
