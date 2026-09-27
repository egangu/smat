import unittest
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from smat.experiments import trace_llm


class TraceLoaderTest(unittest.TestCase):
    def loader(self, append_eos=None):
        backend = Tokenizer(
            WordLevel(
                {"<unk>": 0, "<eos>": 1, "short": 2, "long": 3, "answer": 4},
                unk_token="<unk>",
            )
        )
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            eos_token="<eos>",
            pad_token="<eos>",
        )
        config = {
            "seed": 42,
            "train": {
                "padding": "dynamic",
                "max_length": 64,
                "batch_size": 2,
                "append_eos": append_eos,
            },
        }
        if append_eos is None:
            config["train"].pop("append_eos")
        rows = [
            {"prompt": "short", "answer": "answer"},
            {"prompt": "long long long", "answer": "answer"},
        ]
        with (
            patch.object(trace_llm, "_tokenizer", return_value=tokenizer),
            patch.object(trace_llm, "_rows", return_value=rows),
        ):
            return next(iter(trace_llm.build_loader(config, "NumGLUE-ds")))

    def test_answer_eos_is_supervised_even_when_padding_uses_same_token(self):
        ids, mask, labels = self.loader(True)
        for row in range(2):
            end = int(mask[row].sum()) - 1
            self.assertEqual(int(ids[row, end]), 1)
            self.assertEqual(int(labels[row, end]), 1)
        self.assertTrue((labels[mask == 0] == -100).all())

    def test_eos_is_enabled_by_default(self):
        ids, mask, labels = self.loader()
        for row in range(2):
            end = int(mask[row].sum()) - 1
            self.assertEqual(int(labels[row, end]), 1)


if __name__ == "__main__":
    unittest.main()
