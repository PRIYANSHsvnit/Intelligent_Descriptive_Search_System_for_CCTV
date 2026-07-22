from __future__ import annotations

import unittest

import numpy as np

from pipeline.cfg import K_CROPS
from pipeline.detect_track import _TrackAcc


class CropSelectionTest(unittest.TestCase):
    def test_selection_is_bounded_and_temporally_spread(self):
        acc = _TrackAcc(17)
        rng = np.random.default_rng(17)
        for frame in range(1, 61):
            crop = rng.integers(0, 256, size=(48 + frame % 5, 30, 3), dtype=np.uint8)
            acc.add(frame, 0.9, 100, crop)
        records = acc.best_crop_records()
        frames = [record["frame_no"] for record in records]
        self.assertEqual(len(records), K_CROPS)
        self.assertLess(min(frames), 21)
        self.assertGreater(max(frames), 40)
        self.assertEqual(len(frames), len(set(frames)))


if __name__ == "__main__":
    unittest.main()
