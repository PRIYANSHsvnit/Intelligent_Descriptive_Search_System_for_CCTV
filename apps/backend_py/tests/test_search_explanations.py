from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from app import engine
from app.query_components import QueryPlan


class SearchExplanationTests(unittest.TestCase):
    def test_each_component_keeps_its_own_supporting_crop(self):
        rows = [
            {
                "tracklet_id": "SUR01_c001_p1",
                "score": 0.9,
                "prompt_scores": [0.42, 0.38, 0.31],
                "prompt_crop_refs": ["full.jpg", "shirt.jpg", "cap.jpg"],
            },
            {
                "tracklet_id": "SUR01_c001_p2",
                "score": 0.7,
                "prompt_scores": [0.30, 0.22, 0.20],
                "prompt_crop_refs": ["full2.jpg", "shirt2.jpg", "cap2.jpg"],
            },
        ]
        output = {
            "results": rows,
            "search_mode": "multi_crop",
            "aggregation": "top2_mean",
            "composition": "weighted",
        }
        plan = QueryPlan(
            full_caption="a CCTV image of a man wearing a yellow shirt and black cap",
            components=(
                "a CCTV image of a person wearing a yellow shirt",
                "a CCTV image of a person wearing a black cap",
            ),
            entity_hint="person",
        )
        with (
            mock.patch.object(engine.query_rewrite, "rewrite", return_value="rewritten"),
            mock.patch.object(engine.query_components, "build_query_plan", return_value=plan),
            mock.patch.object(engine, "encode_texts", return_value=np.zeros((3, 1152))),
            mock.patch.object(engine, "_search_with_vectors", return_value=output),
        ):
            result = engine.search("yellow shirt black cap", "person", "SUR01", None, None, 2)

        components = result["results"][0]["component_scores"]
        self.assertEqual("/files/shirt.jpg", components[1]["supporting_crop_url"])
        self.assertEqual("/files/cap.jpg", components[2]["supporting_crop_url"])
        self.assertEqual("component", components[1]["kind"])
        self.assertEqual("strong", components[1]["match_strength"])
        self.assertNotIn("prompt_crop_refs", result["results"][0])


if __name__ == "__main__":
    unittest.main()
