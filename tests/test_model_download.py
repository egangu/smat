import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_models import download_llama


class ModelDownloadTest(unittest.TestCase):
    def test_pinned_download_is_verified_before_final_name_is_written(self):
        content = b"fixed model asset"
        expected = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        source = {"repository": "upstream/model", "revision": "a" * 40}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with patch(
                "scripts.download_models.urlopen", return_value=io.BytesIO(content)
            ) as request:
                download_llama(path, source, expected)
            self.assertEqual(path.read_bytes(), content)
            self.assertFalse(path.with_suffix(".json.partial").exists())
            url = request.call_args.args[0]
            self.assertIn("Revision=" + "a" * 40, url)
            self.assertIn("FilePath=config.json", url)

    def test_checksum_mismatch_never_becomes_a_usable_model_asset(self):
        expected = {"bytes": 5, "sha256": hashlib.sha256(b"right").hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with patch(
                "scripts.download_models.urlopen", return_value=io.BytesIO(b"wrong")
            ):
                with self.assertRaisesRegex(ValueError, "checksum differs"):
                    download_llama(
                        path,
                        {"repository": "upstream/model", "revision": "a" * 40},
                        expected,
                    )
            self.assertFalse(path.exists())
