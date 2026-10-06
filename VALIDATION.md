# Validation record — 2026-10-06

## Historical reproduction

The default recipe was recovered from the real CIFAR execution on 3 August
2026, rather than inferred from the reduced supplemental artifacts. The public
runner now packages that historical entry point. Its three packaging regression
tests verify the random-patch split, beta=3.5, delta=0.44, 100 prediction updates,
64px proxy metric, and refusal to overwrite existing runs.

A first packaging check with the earlier supplemental environment (PyTorch
2.5.1 and torchmetrics 0.11.4) yielded RMSE 0.099371 and proxy SSIM 0.647227.
The historical metric comparison correctly rejected that result. The
original execution record identifies PyTorch 2.13.0 and torchmetrics 1.9.0;
`requirements.txt` now pins the recovered versions. The final packaging run with the recovered libraries matched all three
historical metrics exactly: RMSE 0.09916787594556808, PSNR
20.188501358032227, proxy SSIM 0.6031256318092346. It used Python 3.12.14
on macOS arm64 MPS, demonstrating that the recovered library recipe also works
with that Python version. Mean retained coefficients per patch were 13.475;
global active atoms were 128. The single run completed in about nine seconds.
The data fingerprint is
`90b32e03e00137c2ec663c8d85712e4c9f048b17fe365b2a6ff2ccdabd781853`.
The matched metric row and runtime/source/configuration record are in
`validation/historical_cifar_metrics.csv` and
`validation/historical_cifar_run.json`.

A controlled re-evaluation of the *same* reconstruction from the first run
changed proxy SSIM from 0.647227 under torchmetrics 0.11.4 to 0.605725 under
1.9.0. This isolates the metric-version effect without retraining or adjusting
the experiment's parameters.

With the recovered dependencies, all 45 standalone tests and all 61 original
repository tests passed; `python -m pip check` reported no broken requirements.
The six copied metric/seed helper ASTs are identical to the historical source.
All 18 public Python files parse and source-manifest checks pass. The original
working-repository code was not changed.

The final real-data artifact was checked for one CSV row, finite arrays,
matched reconstruction and strict row-axis global activation. This remains
one historical CIFAR training run, not ten-run Table 2 reproduction.
CPU and CUDA execution have not been validated for this historical recipe.

## SVHN extension checks

The SVHN configuration was fixed before evaluation: delta=0.49, seed 57,
Beta(1,3.5), K=128, 400 outer iterations, 20 R / 10 D steps, and 100 prediction
steps on the legacy 1,800/200 random-patch split. It uses the same dependency
versions and metric functions as CIFAR.

| Metric | Measured value |
| --- | ---: |
| Pixel RMSE | 0.08638068288564682 |
| Patch-mosaic PSNR | 21.0787296295166 |
| Patch-mosaic SSIM | 0.6184457540512085 |
| Mean retained coefficients per patch | 10.449999809265137 |
| Global active rows | 128 |

The first execution and verification execution matched these metrics exactly.
Their shared data fingerprint is
`2f4db47c97a2d11b9809fb07c85486c30e66d38ba25e16c6044a7dbb262189d3`;
their dictionary fingerprints also match. This is one seed/split case checked
for repeatability, not independent repeated-run statistics. The first run
had no established numerical reference and was recorded accordingly; the second
run checked against its recorded reference with the predeclared tolerances.

Records are in `validation/svhn_initial_measurement.json`,
`validation/svhn_legacy_run.json`, and the single-row
`validation/svhn_legacy_metrics.csv`. The CLI records the SVHN dataset and
threshold independently from CIFAR and rejects wrong data fingerprints.

All 47 standalone tests and all 61 original repository tests passed after the
SVHN extension. Tests cover dataset routing, the SVHN delta=0.49, the shared
legacy split and metric settings, and strict row-axis activation at the
threshold boundary. Original working-repository code and experimental
artifacts were not modified.

The official SVHN training download endpoint returned HTTP 200 with the
expected 182,040,794-byte content length during the release check. Actual
training used the existing local official file, verified by torchvision;
a fresh full download was not needed for these checks.

## Earlier supplemental validation

The remaining record describes the reduced official-split experiments now
served by `reproduce_supplemental.py`, with `requirements-supplemental.txt`.
It is retained as prior evidence, not as validation of the new default.

The standalone CPU presets were tested with Python 3.12.14, PyTorch 2.5.1,
torchvision 0.20.1, NumPy 1.26.4, SciPy 1.14.1 and Pillow 11.0.0 on macOS
arm64, using one PyTorch thread. The torch/torchvision pair follows the
[official version pairing](https://pytorch.org/get-started/previous-versions/).
Exact runtime versions are included in `validation/environment.json`.

## Real-data checks

Both official datasets were read from existing local files; no images or
evaluation coefficients were supplied by the public checkpoints. All data
fingerprints matched the historical artifacts. Each final command completed
successfully, with all five coefficient seeds inside the preset's published
numerical comparison tolerances.

| Dataset | Route | RMSE, coefficient seed 57 | Mean retained per patch | Global active atoms |
| --- | --- | ---: | ---: | ---: |
| CIFAR-100 | Released checkpoint | 0.105435044 | 15.024 | 100 |
| CIFAR-100 | Train from scratch, Beta(1, 3.5) | 0.105435207 | 15.022 | 100 |
| SVHN | Released checkpoint | 0.072996609 | 14.784 | 90 |
| SVHN | Train from scratch, Beta(1, 3.5) | 0.072997674 | 14.784 | 90 |

Per-seed results are in `validation/verified_runs.csv`. Checkpoint evaluation
and from-scratch training share the same sampled test patches. Training checks
cover **one dictionary seed per dataset**; the five coefficient seeds do not
establish performance variance across independent training seeds.

The reconstructed CIFAR-100 dictionary differs from the released D by at most
3.55e-5 per entry; the SVHN difference is at most 6.53e-5. These checks support
numerical reproduction, not bit-identical training across devices or libraries.
CUDA, other platforms, fresh independently sampled validation sets, and the
paper's full-data tables have not been validated here.

## Initialization audit

The existing legacy class defaults to training Beta(1, 3.7), while archived
artifacts record β=3.5 without separate fit/predict initialization fields.
Using 3.7 failed to match the historical dictionary metrics in both the
Python 3.8.8 / PyTorch 1.8.1 x86_64 environment and the new environment.
With default coefficient seed 57 in the new environment:

| Dataset | Training β | RMSE | Mean retained per patch | Global active atoms |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | 3.7 | 0.104877248 | 15.688 | 109 |
| CIFAR-100 | 3.5 | 0.105435207 | 15.022 | 100 |
| SVHN | 3.7 | 0.072950549 | 15.174 | 102 |
| SVHN | 3.5 | 0.072997674 | 14.784 | 90 |

The 3.7 route has comparable RMSE but different dictionary use. The release
explicitly uses 3.5 to recover the audited experiment rather than relying on
the class default. The audit rows are in `validation/initialization_audit.csv`.
The original working-repository implementation was not edited.

## Tests and diagnostic flow

- Standalone release: 42 unit tests passed with
  `python -m unittest discover -s tests -v`.
- Original working repository: 61 tests passed in the isolated new environment
  with the additional dependencies in `requirements-validation.txt` and
  `python -m unittest discover -s rebuttal_tools/tests -v`.
- `python -m pip check` found no broken requirements in that environment.
- All release Python sources parsed successfully; source and checkpoint
  SHA-256 manifests matched.
- The public paired-statistics tool rejects missing or empty data fingerprints
  for either method, as well as mismatches; its regression test exercises all
  three missing-data cases before a paired statistic can be produced.
- The real CIFAR-100 Algorithm 1 diagnostic smoke command in
  `REPRODUCIBILITY.md` produced distinct warm-up and evaluation NPZ artifacts.
  The diagnostics tool accepted their matched provenance; the metrics CSV
  contains exactly one row. This is a data-flow check, not performance evidence.

The inherited paired-statistics fixture emits a SciPy precision-loss warning
because its synthetic paired differences are constant. That warning is recorded;
the suite still passes. Synthetic unit-test results are not evidence for
CIFAR-100 or SVHN reconstruction performance.

See `REPRODUCIBILITY.md` for the distinction between the historical direct-gated
variant and Algorithm 1, metric definitions, data conditions, and coverage.
