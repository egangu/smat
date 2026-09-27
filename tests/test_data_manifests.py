"""The released indices must remain disjoint and match the paper's counts."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_data", ROOT / "scripts/prepare_data.py"
)
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


class SplitManifestTest(unittest.TestCase):
    def test_frozen_indices_are_disjoint_and_have_expected_sizes(self):
        expected_trace = {
            "C-STANCE": (4800, 200),
            "FOMC": (4404, 190),
            "MeetingBank": (4799, 200),
            "ScienceQA": (4800, 200),
            "NumGLUE-cm": (4121, 50),
            "NumGLUE-ds": (4297, 102),
            "20Minuten": (4800, 200),
        }
        manifests = ROOT / "manifests/splits"
        for task, counts in expected_trace.items():
            data = json.loads(
                (manifests / "trace" / task / "split_manifest.json").read_text()
            )
            self.assertEqual(
                (len(data["train_indices"]), len(data["dev_indices"])), counts
            )
            self.assertFalse(set(data["train_indices"]) & set(data["dev_indices"]))
            self.assertFalse(set(data["train_group_ids"]) & set(data["dev_group_ids"]))
        vision = list((manifests / "clip8").glob("*/split_manifest.json"))
        self.assertEqual(len(vision), 8)
        for path in vision:
            data = json.loads(path.read_text())
            self.assertEqual(len(data["splits"]["dev"]["indices"]), 200)
            self.assertFalse(
                set(data["splits"]["train"]["indices"])
                & set(data["splits"]["dev"]["indices"])
            )
            for split in ("train", "dev"):
                selection = data["splits"][split]
                length = sum(
                    entry["num_rows"]
                    for entry in data["source_files"][selection["source"]]
                )
                self.assertTrue(
                    all(0 <= index < length for index in selection["indices"])
                )

    def test_preparation_checks_bytes_and_refuses_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw/trace/fixture"
            source.mkdir(parents=True)
            for split in ("train", "eval", "test"):
                prepare_data.write_json(
                    source / f"{split}.json", [{"prompt": "q", "answer": "a"}]
                )
            expected = root / "expected.json"
            prepare_data.write_json(expected, [{"prompt": "q", "answer": "a"}])
            empty = root / "empty.json"
            prepare_data.write_json(empty, [])
            manifest = {
                "source_sha256": {
                    s: prepare_data.sha256(source / f"{s}.json")
                    for s in ("train", "eval", "test")
                },
                "train_indices": [0],
                "train_unique_indices": [0],
                "dev_indices": [],
                "output_sha256": {
                    s: prepare_data.sha256(empty if s == "dev" else expected)
                    for s in ("train", "train_unique", "dev", "eval")
                },
            }
            folder = root / "manifests/trace/fixture"
            folder.mkdir(parents=True)
            prepare_data.write_json(folder / "split_manifest.json", manifest)
            with patch.object(prepare_data, "MANIFESTS", root / "manifests"):
                prepare_data.prepare(root / "raw", root / "prepared")
                self.assertEqual(
                    (root / "prepared/trace/fixture/eval.json").read_bytes(),
                    (source / "eval.json").read_bytes(),
                )
                with self.assertRaises(FileExistsError):
                    prepare_data.prepare(root / "raw", root / "prepared")
                (source / "train.json").write_text("[]")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    prepare_data.prepare(root / "raw", root / "corrupt")


if __name__ == "__main__":
    unittest.main()
