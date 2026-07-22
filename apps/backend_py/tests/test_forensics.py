from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image

from app import forensics


class ForensicExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "camera.mp4"
        self.source.write_bytes(b"unaltered source recording\x00" * 100)
        self.key_dir = self.root / "keys"
        self.export_dir = self.root / "exports"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _fake_clip(source: Path, target: Path, start_s: float, end_s: float) -> None:
        target.write_bytes(b"derived clip:" + source.read_bytes()[:80])

    @staticmethod
    def _fake_frame(source, target, evidence, at_s):
        Image.new("RGB", (160, 90), (25, 35, 50)).save(target, "JPEG")
        return {
            "video_local_timestamp_s": at_s,
            "bounding_box_xyxy": [10, 10, 80, 70],
            "annotation": "derived overlay; source recording remains unmodified",
        }

    def _create(self) -> forensics.ExportResult:
        evidence = {
            "tracklet_id": "SUR01_c001_p7",
            "scene": "SUR01",
            "camera_id": "c001",
            "entity_type": "person",
            "subtype": "person",
            "color": None,
            "ts_start_s": 100.0,
            "ts_end_s": 104.0,
            "video_ref": "SUR01/c001/media/c001.mp4",
            "crop_refs": ["SUR01/c001/crops/p7_0.jpg"],
            "plate_text": None,
            "plate_conf": None,
            "global_id": None,
            "crops": [{
                "crop_index": 0,
                "crop_ref": "SUR01/c001/crops/p7_0.jpg",
                "frame_no": 101,
                "quality": 0.9,
            }],
        }
        request = {
            "tracklet_id": evidence["tracklet_id"],
            "case_id": "FIR-2026-42",
            "officer": "Badge 17",
            "selected_crop_ref": evidence["crops"][0]["crop_ref"],
            "search": {
                "original_query": "person in yellow shirt",
                "timestamp_utc": "2026-07-23T12:00:00Z",
                "filters": {"scene": "SUR01", "camera_id": "c001"},
            },
            "retrieval": {
                "method": "multi_crop",
                "aggregation": "top2_mean",
                "component_scores": [{"caption": "yellow shirt", "score": 0.42}],
            },
        }
        with (
            mock.patch.object(forensics, "_load_evidence", return_value=evidence),
            mock.patch.object(forensics, "_source_path", return_value=self.source),
            mock.patch.object(forensics, "_create_selected_clip", side_effect=self._fake_clip),
            mock.patch.object(forensics, "_create_annotated_frame", side_effect=self._fake_frame),
            mock.patch.object(forensics, "_record_export") as record,
            mock.patch.object(forensics.config, "FORENSIC_EXPORT_ROOT", self.export_dir),
            mock.patch.object(forensics.config, "FORENSIC_KEY_DIR", self.key_dir),
            mock.patch.object(forensics.config, "camera_offset", return_value=0.0),
        ):
            result = forensics.create_export(request)
        record.assert_called_once()
        return result

    def test_export_contains_required_files_and_verifies_with_trusted_key(self):
        result = self._create()
        _, trusted_key = forensics._key_paths(self.key_dir)
        verification = forensics.verify_package(result.package_path, trusted_key)

        self.assertEqual("VALID", verification["status"])
        self.assertTrue(verification["valid"])
        self.assertEqual([], verification["errors"])
        with zipfile.ZipFile(result.package_path) as archive:
            names = {Path(name).name for name in archive.namelist()}
            self.assertEqual(forensics.REQUIRED_FILES, names)
            manifest = json.loads(archive.read("case-export/manifest.json"))
            source_bytes = archive.read("case-export/original_or_source_clip.mp4")
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(
            hashlib.sha256(self.source.read_bytes()).hexdigest(),
            manifest["integrity"]["source_recording_sha256"],
        )
        self.assertEqual("source", manifest["artifacts"]["original_or_source_clip.mp4"]["role"])
        self.assertEqual("derived", manifest["artifacts"]["annotated_frame.jpg"]["role"])

    def test_modified_artifact_is_reported_as_tampered(self):
        result = self._create()
        tampered = self.root / "tampered.zip"
        forensics.create_tampered_demo(result.package_path, tampered)

        _, trusted_key = forensics._key_paths(self.key_dir)
        verification = forensics.verify_package(tampered, trusted_key)
        self.assertEqual("TAMPERED", verification["status"])
        self.assertIn("checksum mismatch: selected_clip.mp4", verification["errors"])

    def test_cli_style_unpinned_verification_warns(self):
        result = self._create()
        verification = forensics.verify_package(result.package_path)
        self.assertTrue(verification["valid"])
        self.assertIn("signer identity was not pinned", verification["warnings"][0])

    def test_case_bundle_verifies_parent_and_nested_export(self):
        child = self._create()
        item = {
            "tracklet_id": "SUR01_c001_p7",
            "status": "pinned",
            "note": "Confirmed by investigator",
            "result_snapshot": {},
        }
        with (
            mock.patch.object(forensics, "create_export", return_value=child),
            mock.patch.object(forensics.config, "FORENSIC_EXPORT_ROOT", self.export_dir),
            mock.patch.object(forensics.config, "FORENSIC_KEY_DIR", self.key_dir),
        ):
            bundle = forensics.create_case_bundle("FIR-2026-42", "Badge 17", [item])
            _, trusted_key = forensics._key_paths(self.key_dir)
            verification = forensics.verify_package(bundle.package_path, trusted_key)

        self.assertEqual("VALID", verification["status"])
        self.assertEqual(bundle.bundle_id, verification["export_id"])
        with zipfile.ZipFile(bundle.package_path) as archive:
            manifest = json.loads(archive.read("case-bundle/manifest.json"))
        self.assertEqual("only case items whose current status was pinned",
                         manifest["selection_policy"])
        self.assertEqual(child.export_id, manifest["evidence"][0]["child_export_id"])


if __name__ == "__main__":
    unittest.main()
