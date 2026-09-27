import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from smat.experiments import trace_llm


class TraceEvaluationSplitTest(unittest.TestCase):
    def evaluate(self, task="NumGLUE-cm", split="eval", legacy=False):
        config = {"seed": 42, "evaluation": {"split": split, "save_samples": False}}
        if legacy:
            config["evaluation"]["numglue_cm_include_test"] = True
        rows = [{"prompt": "1+1", "answer": "2"}]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(trace_llm, "_rows", return_value=rows) as loader,
            patch.object(
                trace_llm, "_sglang_generate_many", new_callable=AsyncMock
            ) as generate,
            patch.object(trace_llm, "_result", return_value=1.0) as result,
        ):
            generate.return_value = ["2", "2"] if legacy else ["2"]
            trace_llm.evaluate(
                config,
                Path(directory),
                task,
                Path(directory) / "result.json",
                endpoint="http://unused",
            )
            return [call.args[2] for call in loader.call_args_list], result.call_args

    def test_numglue_eval_does_not_read_test(self):
        splits, result = self.evaluate()
        self.assertEqual(splits, ["eval"])
        self.assertEqual(len(result.args[4]), 1)
        self.assertFalse(result.kwargs["numglue_cm_include_test"])

    def test_explicit_test_is_not_duplicated(self):
        splits, _ = self.evaluate(split="test")
        self.assertEqual(splits, ["test"])

    def test_other_tasks_still_use_requested_split(self):
        splits, _ = self.evaluate(task="FOMC")
        self.assertEqual(splits, ["eval"])
