from __future__ import annotations

import unittest

from app.query_components import build_query_plan


class QueryComponentsTest(unittest.TestCase):
    def test_composite_person_query(self):
        plan = build_query_plan("a photo of a man in a yellow shirt and black cap")
        self.assertEqual(plan.entity_hint, "person")
        self.assertEqual(len(plan.components), 2)
        self.assertTrue(any("yellow shirt" in value for value in plan.components))
        self.assertTrue(any("black cap" in value for value in plan.components))

    def test_negated_attribute_is_not_positive_component(self):
        plan = build_query_plan("a photo of a rider without a black helmet")
        self.assertIn("without a black helmet", plan.full_caption)
        self.assertFalse(any("helmet" in value for value in plan.components))

    def test_vehicle_prompt(self):
        plan = build_query_plan("a photo of a white SUV")
        self.assertEqual(plan.entity_hint, "vehicle")
        self.assertIn("CCTV image", plan.full_caption)
        self.assertTrue(any("white suv" in value for value in plan.components))


if __name__ == "__main__":
    unittest.main()
