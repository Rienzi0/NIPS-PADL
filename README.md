# NIPS-PADL

The default command runs the historical CIFAR-100 experiment confirmed on
3 August 2026. The SVHN command extends the same model and random-patch protocol
with the existing SVHN threshold delta=0.49. Both recipes have been run on real
data with the pinned dependencies.

## Reproduce CIFAR-100 and SVHN

Use Python 3.12 and the pinned dependencies:

```bash
git clone https://github.com/Rienzi0/NIPS-PADL.git
cd NIPS-PADL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
python reproduce.py --download --device mps --output outputs/cifar-historical-seed57
python reproduce.py --dataset svhn --download --device mps --output outputs/svhn-legacy-seed57
```

The verified executions use Apple MPS. `--device auto` is the CLI default;
CPU and CUDA are explicit alternatives whose results are recorded and compared
to the references. Downloads require `--download`. For existing files, omit
that flag and use `--data-dir /path/to/datasets`. SVHN needs only the official
`train_32x32.mat` file for this legacy-protocol recipe; torchvision checks its
original checksum.

| Fixed seed-57 experiment | Pixel RMSE | Patch-mosaic PSNR | Patch-mosaic SSIM | Mean retained coefficients per patch |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-100 historical recipe | 0.099168 | 20.188501 | 0.603126 | 13.475 |
| SVHN extension | 0.086381 | 21.078730 | 0.618446 | 10.450 |

The coefficient counts are per-patch means. Both audited cases have 128 global
active rows; these results do not demonstrate the separate 19-atom cost claim.

Both recipes fix seed 57, K=128, Beta(1,3.5) initialization, 400 training
iterations with 20 R / 10 D updates, and 100 prediction steps. Delta is 0.44
for CIFAR and 0.49 for SVHN. Each samples 200 images from the dataset's training
partition and crops ten grayscale 8x8 patches per image, using the first
1,800 patches for training and the final 200 for evaluation. SSIM uses the
historical 64x64 patch mosaic. Exact conditions and data fingerprints are in
`profiles/historical_cifar.json` and `profiles/historical_svhn.json`.

These are single-configuration legacy-protocol experiments. The paper's
Table 2 reports ten-run summaries under a different stated data protocol;
these recipes do not reconstruct those original ten runs. See
[PAPER_REPRODUCTION_STATUS.md](PAPER_REPRODUCTION_STATUS.md) for the measured
results and their relationship to the manuscript.

## Outputs and checks

Each invocation trains from scratch and saves `metrics.csv`,
`matched_evaluation.npz` and `manifest.json`. The manifest records the data and
dictionary fingerprints, configuration, runtime versions, source hashes and
metric differences. Existing output directories are rejected. A data or metric
mismatch returns a nonzero exit code and preserves the actual measurements.

```bash
python -m unittest discover -s tests -v
```

[VALIDATION.md](VALIDATION.md) records real-data and packaging checks.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) describes the metric and model
semantics, Algorithm 1 reference, and reduced supplemental presets.
The earlier CIFAR 0.1054 / SVHN 0.0730 presets remain available through
`reproduce_supplemental.py` and are separate from these recipes.

Code is released under the [MIT License](LICENSE).
Dataset images are downloaded separately and are not bundled.
