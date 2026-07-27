"""Trilingual report: translation coverage, verbatim facts, and wall-clock rendering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import forensics, i18n

SUMMARY = {
    "case_id": "FIR-2026-42",
    "export_id": "8c1f0c3e-2b77-4a51-9f0d-71a0c2d4e5b6",
    "officer": "PSI 4471",
    "created_at": "2026-07-26T12:41:09Z",
    "query": "સફેદ કાર",
    "filters": {"scene": "SUR01", "camera_id": "c004"},
    "retrieval_method": "multi_crop top2_mean",
    "searched_at": "2026-07-26T12:39:55Z",
    "tracklet_id": "SUR01_c004_v1832",
    "camera_label": "Railway Station – Bismillah",
    "camera_id": "c004",
    "scene": "SUR01",
    "ts_start_s": 22450.05,
    "ts_end_s": 22456.9,
    "wall_start": "2025-11-14 06:14:10.050",
    "wall_end": "2025-11-14 06:14:16.900",
    "video_start_s": 88.05,
    "video_end_s": 94.9,
    "entity_type": "vehicle",
    "subtype": "suv",
    "color": "white",
    "plate": "GJ05KL1234",
    "source_sha256": "3f9c1b" + "a" * 58,
    "signing_key_id": "sha256:" + "7d" * 32,
}


class CatalogTests(unittest.TestCase):
    def test_every_string_is_translated_into_every_language(self):
        for key, entry in i18n._STRINGS.items():
            for lang in i18n.LANGUAGES:
                with self.subTest(key=key, lang=lang):
                    self.assertTrue(entry.get(lang, "").strip(),
                                    f"{key} has no {lang} translation")

    def test_vocabulary_covers_both_indic_languages(self):
        for token, entry in i18n._VOCAB.items():
            for lang in ("hi", "gu"):
                with self.subTest(token=token, lang=lang):
                    self.assertTrue(entry.get(lang, "").strip())

    def test_clock_note_placeholder_survives_translation(self):
        for lang in i18n.LANGUAGES:
            rendered = i18n.t("clock_note", lang).format(tz="IST")
            self.assertIn("IST", rendered)
            self.assertNotIn("{tz}", rendered)

    def test_unknown_detector_token_degrades_to_raw_token(self):
        self.assertEqual("quadricycle", i18n.term("quadricycle", "hi"))
        self.assertIn("quadricycle", i18n.entity_phrase(None, "quadricycle", "vehicle", "gu"))

    def test_entity_phrase_keeps_english_tokens_alongside_translation(self):
        gujarati = i18n.entity_phrase("white", "suv", "vehicle", "gu")
        self.assertIn("white suv (vehicle)", gujarati)
        self.assertIn("સફેદ", gujarati)
        self.assertEqual("white suv (vehicle)", i18n.entity_phrase("white", "suv", "vehicle", "en"))


class WallClockTests(unittest.TestCase):
    def test_scene_seconds_render_as_configured_date_and_time(self):
        with mock.patch.object(forensics.config, "recording_date",
                               return_value=("2025-11-14", True)):
            self.assertEqual("2025-11-14 06:14:10.050",
                             forensics._wall_clock("SUR01", 22450.05))

    def test_seconds_past_midnight_roll_the_date_forward(self):
        with mock.patch.object(forensics.config, "recording_date",
                               return_value=("2025-11-14", True)):
            self.assertEqual("2025-11-15 00:00:30.000",
                             forensics._wall_clock("SUR01", 86430.0))

    def test_malformed_configured_date_falls_back_to_the_ingest_base(self):
        with mock.patch.object(forensics.config, "recording_date",
                               return_value=("not-a-date", True)):
            self.assertTrue(forensics._wall_clock("SUR01", 0.0).startswith(
                forensics.config.INGEST_BASE_DATE))


class ReportRenderingTests(unittest.TestCase):
    def _rows(self, lang: str) -> list[tuple[str, str, str]]:
        return forensics._report_rows(SUMMARY, lang)

    def test_recorded_facts_are_verbatim_in_every_language(self):
        verbatim = [
            SUMMARY["case_id"], SUMMARY["export_id"], SUMMARY["officer"],
            SUMMARY["tracklet_id"], SUMMARY["source_sha256"], SUMMARY["signing_key_id"],
            SUMMARY["wall_start"], SUMMARY["wall_end"], SUMMARY["query"],
            SUMMARY["plate"], SUMMARY["camera_id"],
        ]
        for lang in i18n.LANGUAGES:
            blob = "\n".join(f"{left}\t{right}" for _, left, right in self._rows(lang))
            for fact in verbatim:
                with self.subTest(lang=lang, fact=fact):
                    self.assertIn(fact, blob)

    def test_labels_actually_differ_between_languages(self):
        labels = {lang: [left for kind, left, _ in self._rows(lang) if kind == "kv"]
                  for lang in i18n.LANGUAGES}
        self.assertNotEqual(labels["en"], labels["hi"])
        self.assertNotEqual(labels["hi"], labels["gu"])
        self.assertEqual(len(labels["en"]), len(labels["gu"]))

    def test_report_has_one_page_per_language_and_embeds_indic_fonts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.pdf"
            forensics._create_report(path, SUMMARY)
            raw = path.read_bytes()
        self.assertTrue(raw.startswith(b"%PDF-"))
        self.assertEqual(len(i18n.LANGUAGES), raw.count(b"/Type /Page\n"))
        for family in ("NotoSans", "NotoSansDevanagari", "NotoSansGujarati"):
            self.assertIn(family.encode(), raw, f"{family} was not embedded")

    def test_missing_optional_fields_do_not_break_rendering(self):
        summary = dict(SUMMARY, plate=None, color=None, query="", searched_at=None)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.pdf"
            forensics._create_report(path, summary)
            self.assertGreater(path.stat().st_size, 1000)
        for lang in i18n.LANGUAGES:
            rows = forensics._report_rows(summary, lang)
            values = [right for _, _, right in rows]
            self.assertIn(i18n.t("query_absent", lang), values)
            self.assertNotIn(i18n.t("plate", lang), [left for _, left, _ in rows])


if __name__ == "__main__":
    unittest.main()
