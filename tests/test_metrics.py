import unittest

from rouge import Rouge

from smat.eval.metrics import rouge_l, sari


class RougeEmptySentenceTests(unittest.TestCase):
    def test_sentence_empty_outputs_still_count_in_denominator(self):
        predictions = ["Exact words", "...", ". \n .", "\t", ""]
        references = ["Exact words"] * len(predictions)
        matched = Rouge(metrics=["rouge-l"]).get_scores(
            "Exact words",
            "Exact words",
            avg=True,
        )["rouge-l"]["f"]
        self.assertEqual(rouge_l(predictions, references), matched / 5)

    def test_sentence_empty_references_score_zero(self):
        self.assertEqual(rouge_l(["valid", "valid"], ["...", ""]), 0.0)

    def test_existing_sentence_scores_are_unchanged(self):
        predictions = ["A brief summary. Another item.", "北京 会议 结束.", "?"]
        references = ["A summary. One item.", "北京 会议.", "no punctuation"]
        scorer = Rouge(metrics=["rouge-l"])
        expected = sum(
            scorer.get_scores(reference, prediction, avg=True)["rouge-l"]["f"]
            for prediction, reference in zip(predictions, references)
        ) / len(predictions)
        self.assertEqual(rouge_l(predictions, references), expected)

    def test_empty_batch_scores_zero(self):
        self.assertEqual(rouge_l([], []), 0.0)


class SariTest(unittest.TestCase):
    def test_perfect_simplification_and_empty_input(self):
        self.assertEqual(sari(["the cat sat"], ["the cat"], ["the cat"]), 1.0)
        self.assertEqual(sari([], [], []), 0.0)


if __name__ == "__main__":
    unittest.main()
