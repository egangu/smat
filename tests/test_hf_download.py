"""Check pinned downloads, experiment paths, and protection of existing results."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("download_hf", Path(__file__).parents[1] / "scripts/download_hf.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def checksum(payload):
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class DownloadTests(TestCase):
    def test_dataset_task_filter_and_pinned_revision(self):
        payload = b'[{"prompt":"p","answer":"a"}]'
        record = {"repo_id": "owner/data", "revision": "fixed-sha", "files": {
            "trace/FOMC/train.json": checksum(payload),
            "trace/ScienceQA/train.json": checksum(payload)}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fetch(repo, name, **kw):
                self.assertEqual((repo, kw["revision"], kw["repo_type"]), ("owner/data", "fixed-sha", "dataset"))
                path = kw["local_dir"] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            with patch.object(download, "hf_hub_download", side_effect=fetch) as mock:
                download.download_data(record, root, ["FOMC"])
                download.download_data(record, root, ["FOMC"])
                self.assertEqual(mock.call_count, 1)
            path = root / "trace/FOMC/train.json"
            path.write_bytes(b'corrupt')
            with patch.object(download, "hf_hub_download") as mock, self.assertRaises(ValueError):
                download.download_data(record, root, ["FOMC"])
            mock.assert_not_called()

    def test_checkpoint_resume_and_protect_training_results(self):
        payload = b'weights'
        record = {"repo_id": "owner/model", "revision": "fixed-sha", "files": {"model.safetensors": checksum(payload)}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "expert"
            def fetch(repo, **kw):
                self.assertEqual((repo, kw["revision"]), ("owner/model", "fixed-sha"))
                (kw["local_dir"] / "model.safetensors").write_bytes(payload)
            with patch.object(download, "snapshot_download", side_effect=fetch):
                download.download_expert(record, root)
                download.download_expert(record, root)
            (root / "model.safetensors").write_bytes(b'bad')
            with patch.object(download, "snapshot_download") as mock, self.assertRaises(ValueError):
                download.download_expert(record, root)
            mock.assert_not_called()
            (root / ".hf-release.json").unlink()
            with self.assertRaises(ValueError):
                download.download_expert(record, root)

    def test_vision_checkpoint_path_and_task_validation(self):
        record = {"case": "b32_adam_smat_shared_scale01_s42", "task": "MNIST"}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            (root / "manifests/hf_release.json").write_text(json.dumps({"models": [record]}))
            argv = ["download_hf.py", "experts", "--model", "vitb32", "--method", "smat", "--tasks", "MNIST", "--output-root", str(root)]
            with patch.object(download, "ROOT", root), patch.object(download, "download_expert") as fetch, patch("sys.argv", argv):
                download.main()
            fetch.assert_called_once_with(record, root / "training/experts/vitb32_adam_smat/MNIST")
            argv[7] = "FOMC"
            with patch.object(download, "ROOT", root), patch.object(download, "download_expert") as fetch, patch("sys.argv", argv), self.assertRaises(SystemExit):
                download.main()
            fetch.assert_not_called()

    def test_summary_without_private_timing_metadata(self):
        spec = importlib.util.spec_from_file_location("summarize", download.ROOT / "scripts/summarize.py")
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            record = {"name": "test", "scores": {"expert": {"macro": 0.5}}, "training_metadata": {"FOMC": {"seed": 42}}, "tasks": ["FOMC"], "evaluation": {}}
            path.write_text(json.dumps(record))
            result = summary.summarize(path)
            self.assertEqual(result["scores_percent"]["expert"], 50)
            self.assertIsNone(result["suite_training_seconds"])
            record["training_metadata"]["FOMC"]["loop_elapsed_seconds"] = 12.5
            path.write_text(json.dumps(record))
            self.assertEqual(summary.summarize(path)["suite_training_seconds"], 12.5)

    def test_cli_places_smat_expert_where_config_expects(self):
        record = {"case": "1b_adamw_smat_shared_rms2e3_s42", "task": "FOMC"}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            (root / "manifests/hf_release.json").write_text(json.dumps({"models": [record]}))
            with patch.object(download, "ROOT", root), patch.object(download, "download_expert") as fetch, patch("sys.argv", ["download_hf.py", "experts", "--model", "llama1b", "--method", "smat", "--tasks", "FOMC", "--output-root", str(root)]):
                download.main()
            fetch.assert_called_once_with(record, root / "training/experts/llama1b_adamw_smat/FOMC")


if __name__ == "__main__":
    main()
