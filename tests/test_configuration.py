"""Configuration errors must be explicit before launching an experiment."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smat.config import load_config
from smat.train.updates import validate_settings

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTest(unittest.TestCase):
    def test_every_released_recipe_resolves_to_ft_or_smat(self):
        with patch.dict(
            os.environ,
            {"MODEL_ROOT": "/models", "DATA_ROOT": "/data", "OUTPUT_ROOT": "/outputs"},
        ):
            for path in (ROOT / "configs").glob("*/*.json"):
                config = load_config(path)
                validate_settings(config["train"])
                self.assertIn(config["train"]["method"], ("ft", "smat"))
                self.assertTrue(config["base_model"].startswith("/models/"))

    def test_unset_asset_variable_is_not_treated_as_a_literal_path(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
        ):
            path = Path(directory) / "recipe.json"
            path.write_text(
                json.dumps(
                    {"experiment": "trace_llm", "base_model": "${MODEL_ROOT}/model"}
                )
            )
            with self.assertRaisesRegex(ValueError, "unresolved environment"):
                load_config(path)

    def test_removed_or_misspelled_options_do_not_silently_change_a_run(self):
        with self.assertRaisesRegex(ValueError, "unsupported training"):
            validate_settings({"method": "ft", "scale": {"alpha_min": 0.2}})
        with self.assertRaisesRegex(ValueError, "unsupported training"):
            validate_settings({"method": "smat", "learing_rate": 1e-5})


if __name__ == "__main__":
    unittest.main()
