import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from smat.experiments.clip8_vit import (
    _checkpoint_file,
    _zero_shot_class_weights,
    evaluate,
    evaluate_many,
)


class ClipCheckpointPathTest(unittest.TestCase):
    def test_partial_directory_is_not_mistaken_for_a_weight_file(self):
        directory = Path("/tmp/.weight_average.partial-deadbeef")
        self.assertEqual(_checkpoint_file(directory), directory / "encoder.pt")

    def test_per_class_zero_shot_weights_are_normalized(self):
        names = ("one", "two", "three")
        templates = ("a {classname}", "the {classname}")
        prompts = [
            template.format(classname=name) for name in names for template in templates
        ]
        vectors = {
            prompt: torch.tensor((index + 1.0, (index + 1.0) ** 2))
            for index, prompt in enumerate(prompts)
        }
        calls = []

        def text_features(_model, _tokenizer, selected, _device):
            calls.append(tuple(selected))
            return torch.stack([vectors[prompt] for prompt in selected])

        with patch(
            "smat.experiments.clip8_vit._text_features", side_effect=text_features
        ):
            actual = _zero_shot_class_weights(
                object(), object(), names, templates, torch.device("cpu")
            )
        self.assertEqual(
            calls,
            [tuple(prompts[index : index + 2]) for index in range(0, 6, 2)],
        )

        expected = []
        for name in names:
            embeddings = torch.stack(
                [vectors[template.format(classname=name)] for template in templates]
            )
            expected.append(F.normalize(embeddings, dim=-1).mean(dim=0))
        expected = F.normalize(torch.stack(expected), dim=-1)
        self.assertEqual(actual.shape, (len(names), 2))
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(len(names)))
        torch.testing.assert_close(actual, expected)

    def test_evaluate_many_reuses_one_task_session_and_reloads_a_b_a(self):
        class FakeEncoder:
            def __init__(self):
                self.loads = []
                self.checkpoint = ""

            def load_state_dict(self, state, *, strict):
                self.checkpoint = state["checkpoint"]
                self.loads.append((self.checkpoint, strict))

        class FakeModel:
            def __init__(self):
                self.encoder = FakeEncoder()
                self.head_tasks = []
                self.eval_calls = 0

            def eval(self):
                self.eval_calls += 1
                return self

            def _build_head(self, task):
                self.head_tasks.append(task)

            def logits(self, _task, images):
                if self.encoder.checkpoint == "b.pt":
                    return torch.tensor([[0.0, 2.0]]).repeat(images.shape[0], 1)
                return torch.tensor([[2.0, 0.0]]).repeat(images.shape[0], 1)

        config = {"evaluation": {"split": "test"}}
        model = FakeModel()
        loader = [
            {"task": "Cars", "images": torch.zeros((1, 1)), "labels": torch.tensor([0])}
        ]
        payloads = {}
        requests = (
            (Path("a.pt"), Path("a-first.json")),
            (Path("b.pt"), Path("b.json")),
            (Path("a.pt"), Path("a-last.json")),
        )

        with (
            patch(
                "smat.experiments.clip8_vit.build_model", return_value=model
            ) as build_model,
            patch(
                "smat.experiments.clip8_vit.build_loader", return_value=loader
            ) as build_loader,
            patch("smat.experiments.clip8_vit.register_loader") as register_loader,
            patch(
                "smat.experiments.clip8_vit._device", return_value=torch.device("cpu")
            ),
            patch(
                "smat.experiments.clip8_vit._image_encoder_state",
                side_effect=lambda checkpoint: {"checkpoint": str(checkpoint)},
            ),
            patch(
                "smat.experiments.clip8_vit.atomic_write_json",
                side_effect=lambda path, payload: payloads.__setitem__(
                    Path(path), payload
                ),
            ),
        ):
            scores = evaluate_many(config, "Cars", requests)

        self.assertEqual(scores, [1.0, 0.0, 1.0])
        build_model.assert_called_once_with(config, training=False)
        build_loader.assert_called_once_with(
            config, "Cars", split="test", max_samples=None
        )
        register_loader.assert_called_once_with(model, loader)
        self.assertEqual(model.head_tasks, ["Cars"])
        self.assertEqual(
            model.encoder.loads, [("a.pt", True), ("b.pt", True), ("a.pt", True)]
        )
        self.assertEqual(payloads[Path("a-first.json")], payloads[Path("a-last.json")])
        self.assertEqual(payloads[Path("b.json")]["correct"], 0)
        self.assertEqual(payloads[Path("b.json")]["protocol"], "clip8")
        self.assertEqual(payloads[Path("b.json")]["num_samples"], 1)

    def test_evaluate_delegates_to_the_multi_checkpoint_path(self):
        config = {"evaluation": {"split": "test"}}
        with patch(
            "smat.experiments.clip8_vit.evaluate_many", return_value=[0.75]
        ) as evaluate_many_mock:
            score = evaluate(config, Path("model.pt"), "Cars", Path("output.json"))
        self.assertEqual(score, 0.75)
        evaluate_many_mock.assert_called_once_with(
            config,
            "Cars",
            ((Path("model.pt"), Path("output.json")),),
            None,
        )


if __name__ == "__main__":
    unittest.main()
