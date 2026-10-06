# Reproduction scope

## Recovered confirmed experiment

The prior reproduction work already resolved the historical CIFAR settings.
On 3 August 2026, `general_exp.run_historical_cifar_experiment(seed=57,
device="mps")` was implemented and run on real CIFAR-100 data, producing:

- RMSE: 0.09916787594556808
- Patch-mosaic PSNR: 20.188501358032227
- Patch-mosaic SSIM: 0.6031256318092346
- Mean retained coefficients per patch: 13.475000381469727

The public `reproduce.py` now uses that experiment, including its 1,800/200
random-patch split, beta=3.5, delta=0.44, Beta regularizer, and 100 prediction
steps. This recovery uses the existing code and prior execution record;
it does not require a new parameter search.

The historical environment used Python 3.11, PyTorch 2.13.0, torchvision
0.28.0, torchmetrics 1.9.0, NumPy 2.4.6, SciPy 1.17.1 and Pillow 12.3.0.
The package pins these library versions. Current execution checks and any
remaining differences are recorded in `VALIDATION.md`.

## Manuscript targets and coverage

The manuscript PDF page 8, Table 2, and `table2_rmse_summary.csv` report:

| Dataset | PADL RMSE | PSNR | SSIM | Repeated runs |
| --- | --- | --- | --- | ---: |
| CIFAR-100 | 0.099±0.009 | 20.56±0.55 | 0.568±0.030 | 10 |
| SVHN | 0.090±0.006 | 20.88±1.03 | 0.583±0.037 | 10 |

The recovered historical recipe demonstrates the previously confirmed CIFAR
RMSE case. Its random crops, sample counts and proxy SSIM differ from the
manuscript's stated official-split, non-overlapping 20,000/2,000-patch protocol.
A single seed cannot reproduce the ten-run means and standard deviations.
The mean retained coefficient count is a per-patch statistic, not the globally
19-atom dictionary reported in the separate Table 3 cost experiment.

The two reduced official-split presets initially packaged here (CIFAR RMSE
about 0.1054; SVHN about 0.0730) are supplemental experiments. Their entry point
is now `reproduce_supplemental.py`. Their existing artifacts and validation
records remain intact. They are not presented as Table 2 reproductions.

## SVHN extension verified on 6 October 2026

The public runner now also accepts `--dataset svhn`. It extends the confirmed
CIFAR legacy protocol without a new parameter sweep: seed 57, 200 train-partition
images, 1,800/200 random patches, beta=3.5, K=128, 400 outer iterations,
100 prediction updates, and the pre-existing SVHN delta=0.49.

The real-data result is RMSE 0.08638068288564682, patch-mosaic PSNR
21.0787296295166, and patch-mosaic SSIM 0.6184457540512085. Repeating the
same fixed recipe returned identical metrics and data/dictionary fingerprints.
Mean retained coefficients per patch were 10.45; global active rows were 128.
The full record is `validation/svhn_legacy_run.json`.

This establishes repeatability of the SVHN extension, alongside the historical
CIFAR case. It does not establish the original SVHN Table 2 ten-run summary,
the official-split 20,000/2,000-patch experiment, or a globally compact dictionary.
The two recipe outputs are documented as measured single-configuration results.
