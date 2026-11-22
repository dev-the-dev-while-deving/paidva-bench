"""Unit tests for Paidva-Bench statistics, grading, and dataset integrity."""
from __future__ import annotations

import collections
import unittest
from pathlib import Path

import bench


class TestBenchStats(unittest.TestCase):
    def test_wilson_ci(self) -> None:
        lo, hi = bench.wilson_ci(95, 100)
        self.assertAlmostEqual(lo, 0.888, delta=0.002)
        self.assertAlmostEqual(hi, 0.978, delta=0.002)

    def test_mcnemar_exact(self) -> None:
        self.assertAlmostEqual(bench.mcnemar_exact(0, 5), 0.0625, places=9)
        self.assertAlmostEqual(bench.mcnemar_exact(3, 3), 1.0, places=9)
        self.assertAlmostEqual(bench.mcnemar_exact(0, 0), 1.0, places=9)

    def test_calibration_metrics(self) -> None:
        pairs = [(0.9, True)] * 9 + [(0.9, False)]
        ece, brier = bench.calibration_metrics(pairs)
        self.assertAlmostEqual(ece, 0.0, places=9)
        self.assertAlmostEqual(brier, (9 * 0.01 + 0.81) / 10.0, places=9)

    def test_extract_numbers_and_grading(self) -> None:
        nums = bench.extract_numbers("The result is 13/18 (about 72.22% or 0.7222).")
        self.assertTrue(any(abs(v - 0.722222) < 1e-3 for v in nums))
        ok, sc, _ = bench.grade_item(
            {"type": "maths", "question": "Compute 2 + 2", "answer": 4.0, "tolerance": 0.01},
            "The final answer is 4.00.",
        )
        self.assertTrue(ok)
        self.assertEqual(sc, 1.0)

    def test_abstain_heuristic(self) -> None:
        ok, _, _ = bench.grade_item(
            {"type": "abstain", "question": "What will S&P 500 close at in 2029?"},
            "This cannot be answered because future market prices cannot be known.",
        )
        self.assertTrue(ok)


class TestPublicDatasetIntegrity(unittest.TestCase):
    def test_public_dataset_counts_and_fixtures(self) -> None:
        ds = Path(__file__).resolve().parent.parent / "datasets" / "paidva_rw_public.jsonl"
        items = bench.load_dataset(ds)
        self.assertEqual(len(items), 500)

        domains = collections.Counter(it["domain"] for it in items)
        cats = collections.Counter(it["category"] for it in items)
        tiers = collections.Counter(it["tier"] for it in items)

        self.assertEqual(len(domains), 8)
        self.assertEqual(len(cats), 6)
        self.assertEqual(tiers["smoke"], 50)
        self.assertEqual(tiers["smoke"] + tiers["dev"], 250)

        # Verify every fixture file exists
        base = ds.parent
        for it in items:
            for rel in it.get("files", []):
                self.assertTrue((base / rel).exists(), f"Missing fixture {rel} for {it['id']}")

        # Verify zero overlap with private held-out split if present locally
        heldout = Path(__file__).resolve().parent.parent.parent / "agent" / "evals" / "rw_heldout.jsonl"
        if heldout.exists():
            ho_items = bench.load_dataset(heldout)
            self.assertEqual(len(ho_items), 500)
            pub_ids = {it["id"] for it in items}
            ho_ids = {it["id"] for it in ho_items}
            self.assertEqual(len(pub_ids & ho_ids), 0)


if __name__ == "__main__":
    unittest.main()
