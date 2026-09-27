import unittest

from smat.eval.metrics import extract_label, label_accuracy, numeric_accuracy


class TraceMetricTest(unittest.TestCase):
    def test_extract_label_prefers_explicit_answer_phrases(self):
        self.assertEqual(extract_label("The answer is C."), "C")
        self.assertEqual(extract_label("Answer: D"), "D")
        self.assertEqual(extract_label("I choose D."), "D")
        self.assertEqual(extract_label("答案是B"), "B")

    def test_extract_label_accepts_option_tokens_but_not_letters_inside_words(self):
        self.assertEqual(extract_label("C"), "C")
        self.assertEqual(extract_label("(d)"), "D")
        self.assertEqual(extract_label("(A) Because it follows from the premise."), "A")
        self.assertEqual(extract_label("I believe B is correct."), "B")
        self.assertEqual(extract_label("This is a result."), "")
        self.assertEqual(extract_label("There is no valid option."), "")

    def test_accuracy_counts_unparseable_predictions_as_incorrect(self):
        self.assertEqual(label_accuracy(["A", ""], ["A", "B"]), 0.5)
        self.assertEqual(numeric_accuracy(["1", ""], ["1", "2"]), 0.5)

    def test_numeric_accuracy_prefers_an_explicit_answer(self):
        self.assertEqual(
            numeric_accuracy(["There are 2 steps. Answer: 4"], ["4"]),
            1.0,
        )
        self.assertEqual(numeric_accuracy(["-3"], ["-3"]), 1.0)


if __name__ == "__main__":
    unittest.main()
