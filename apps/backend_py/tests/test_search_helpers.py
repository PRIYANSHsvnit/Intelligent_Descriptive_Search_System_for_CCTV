from __future__ import annotations

import unittest

import numpy as np

from app.engine import _aggregate, _dedup, _percentiles


class SearchHelpersTest(unittest.TestCase):
    def test_crop_aggregations(self):
        values = np.array([0.2, 0.8, 0.6], dtype=np.float32)
        self.assertAlmostEqual(_aggregate(values, "max"), 0.8, places=6)
        self.assertAlmostEqual(_aggregate(values, "top2_mean"), 0.7, places=6)
        self.assertAlmostEqual(_aggregate(values, "best_single"), 0.2, places=6)

    def test_percentiles_preserve_order(self):
        out = _percentiles(np.array([0.4, 0.1, 0.7], dtype=np.float32))
        self.assertEqual(int(np.argmax(out)), 2)
        self.assertEqual(float(out[1]), 0.0)

    def test_overlapping_lookalikes_are_not_deduplicated(self):
        base = {
            "scene": "S", "camera_id": "c1", "entity_type": "person",
            "subtype": "person", "global_id": None, "plate_text": None,
            "dedup_vector": np.array([1.0, 0.0]),
        }
        rows = [
            {**base, "tracklet_id": "a", "ts_start_s": 1.0, "ts_end_s": 3.0},
            {**base, "tracklet_id": "b", "ts_start_s": 2.0, "ts_end_s": 4.0},
        ]
        self.assertEqual(len(_dedup(rows)), 2)


if __name__ == "__main__":
    unittest.main()
