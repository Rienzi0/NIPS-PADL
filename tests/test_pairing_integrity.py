"""Reject paired claims when matching data has not been established."""

import unittest

from rebuttal_tools.paired_run_statistics import analyze


class PairingIntegrityTests(unittest.TestCase):
    def test_paired_statistics_require_nonempty_fingerprints_from_both_methods(self):
        for reference_fingerprint, baseline_fingerprint in ((None, None), ("", ""), ("abc", "")):
            with self.subTest(reference=reference_fingerprint, baseline=baseline_fingerprint):
                rows = []
                for seed in (1, 2):
                    for method, fingerprint, value in (("padl", reference_fingerprint, .1), ("baseline", baseline_fingerprint, .2)):
                        row = {"dataset": "synthetic", "seed": str(seed), "split_id": "fixed", "method": method, "rmse": str(value + .01 * seed)}
                        if fingerprint is not None:
                            row["data_fingerprint"] = fingerprint
                        rows.append(row)
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    analyze(rows, reference="padl", comparators=["baseline"], metrics=["rmse"], unpaired=False)


if __name__ == "__main__":
    unittest.main()
