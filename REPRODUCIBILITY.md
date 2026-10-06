# Reproduction contract

## Default: the confirmed historical CIFAR experiment

`reproduce.py` packages the historical runner already verified on 3 August
2026. It preserves the DataLoader shuffle and NumPy crop sampling, including
the original exclusive crop-coordinate upper bound. All random generators are
seeded once with 57 before data loading; coefficient prediction continues the
training run's random-generator state rather than being reseeded separately.

The legacy split uses 200 images from CIFAR-100's training partition, ten
random crops per image, 1,800 training patches and 200 evaluation patches.
This is a split of the sampled images, not the official CIFAR test partition.
The confirmed model uses beta=3.5, delta=0.44, temperature 20, K=128,
400 outer iterations, 20 R and 10 D updates, and 100 prediction updates.
Beta regularization remains enabled. Patch normalization uses sample standard
deviation with correction=1 and an added 1e-8.

The historical execution used MPS and the dependency versions now pinned in
`requirements.txt`. Device and runtime are recorded in every output manifest.
Changes in dependencies or devices are not assumed to reproduce identical
training or metric values. The numerical comparison reports actual differences.

## SVHN extension

`reproduce.py --dataset svhn` uses the same random-patch model and metric
helpers, with SVHN's existing threshold delta=0.49. It samples the official
SVHN **training** partition; the last 20 sampled images provide the 200
held-out random crops. It does not load the official SVHN test partition.
The configuration was fixed before the new real-data measurement. No seed,
threshold or iteration sweep was used to select this reported result.

The recipe is stored in `profiles/historical_svhn.json`. Its first measurement
is preserved in `validation/svhn_initial_measurement.json`; the subsequent
repeatability check is `validation/svhn_legacy_run.json`. Both use the same
seed and split, so they are verification executions of one experimental case,
not two independent statistical runs. Reference metric tolerances remain
RMSE 0.0005, PSNR 0.05, and proxy SSIM 0.005.

## Historical image metrics

RMSE uses all 200 raw evaluation patches. The historical PSNR and SSIM reshape
consecutive crops into 64×64 mosaics: 64 patches per mosaic, three complete
mosaics, with the final eight patches omitted from those two proxy metrics.
The mosaics do not reconstruct complete CIFAR images. Their intensity range is
the joint minimum/maximum over ground-truth and reconstructed mosaics, as in
the historical metric functions.

Mean retained coefficients counts nonzero gated coefficients per patch.
Global active atoms are reported separately using the strict row rule
`max_i |R_ki| > delta`. Neither quantity is inferred from the other.

## Two distinct algorithms

Coefficients have shape `[K, N]` in both implementations.

`padl_algorithm1_reference.py` implements one-shot warm-up, Beta(1, β) fitting,
λ1=β−1, λ2=1, and reinitialization of both D and R. Its activation penalty is
`abs(R).max(dim=1).values.sum()`. Reconstruction uses `D @ R` and its active set
uses the strict row rule `max_i |R_ki| > δ`.

`legacy_direct_gate.py` preserves the historical direct-gated experiment:

- R is represented by logits and converted through sigmoid.
- Training uses `R * sigmoid(20 * (R - δ))` inside reconstruction.
- Evaluation uses the elementwise hard gate `R * (R >= δ)`.
- The R objective adds 1.33 times mean L1 and 1.01 times the sum of **column**
  maxima, plus the historical Beta regularizer with weight 0.1.
- D updates use 10 times the reconstruction loss; D is clipped to [0, 1]
  and normalized by columns after each outer iteration.

The historical variant has a different objective and threshold semantics.
Its checkpoint results are not evidence that Algorithm 1 reproduces the paper's
main-table numbers. Source origins and the small packaging-only change to the
historical module are recorded in `SOURCE_MANIFEST.json`.

## Metrics and statistical unit

Pixel RMSE is `sqrt(mean((X_test_raw - X_reconstruction)**2))` over the sampled
8×8 grayscale patches in the original [0, 1] intensity scale.
`mean_retained_per_patch` counts nonzero effective coefficients per column.
`global_active_atoms` counts rows with any nonzero effective coefficient.
These quantities have different meanings; a mean of about 15 coefficients
per patch does not imply a globally 15-atom dictionary.

The five supplemental coefficient seeds share one dictionary and one test subset.
They are descriptive repeats, not independent runs for statistical tests of
training performance. Comparisons between methods require identical dataset,
seed, split ID and a nonempty identical data fingerprint. Independent training
seeds, rather than patches or coefficient repeats, are the statistical unit.

## Reduced official-split supplemental presets

The older `profiles/cifar100.json` and `profiles/svhn.json` describe separate
2,000-training / 500-test-patch experiments using official splits and
non-overlapping grayscale 8×8 patches. Their entry point is
`reproduce_supplemental.py`; use `requirements-supplemental.txt` for the
previously validated CPU environment. The CIFAR and SVHN seed-57 checkpoint
RMSEs are approximately 0.105435 and 0.072997 respectively.

These presets use a seed-57 dictionary, beta=3.5, 400 training iterations,
20 R / 10 D steps, and 1,100 prediction steps. Delta is 0.40 for CIFAR and 0.49
for SVHN. Five coefficient seeds share one dictionary and sampled evaluation
set. They measure initialization sensitivity rather than independent training
variance. Their existing provenance, checkpoints and verification CSVs remain
available; they do not replace the historical default or the manuscript tables.

## Coverage

The default recipe covers the previously confirmed historical CIFAR case;
`--dataset svhn` covers the verified fixed-configuration SVHN extension.
It is a direct-gated experiment, separate from the Algorithm 1 reference.
The Algorithm 1 reference and theory diagnostics retain their existing
mathematical and data-flow checks. Neither those tests nor the reduced
supplemental results establish the manuscript's full repeated-run performance.
See `PAPER_REPRODUCTION_STATUS.md` and `VALIDATION.md` for exact coverage.
