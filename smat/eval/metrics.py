"""TRACE metrics with strict answer parsing around the public definitions."""

from __future__ import annotations

import re
from collections import Counter

from packaging import version


def extract_label(text: str) -> str:
    patterns = (
        r"\b(?:final\s+)?answer\s*(?:is|:|=)?\s*[\(\[（]?\s*([A-D])\b",
        r"\b(?:choose|choice\s+is|option)\s*[\(\[（]?\s*([A-D])\b",
        r"(?:答案|选择|选项|选)\s*(?:是|为|:|：)?\s*[\(\[（]?\s*([A-D])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.fullmatch(
        r"\s*[\(\[（]?\s*([A-D])\s*[\)\]）\.、,:：\-]?\s*",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    match = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text)
    if match:
        return match.group(1)
    return ""


def label_accuracy(
    predictions: list[str], references: list[str], first_character: bool = False
) -> float:
    correct = 0
    for prediction, reference in zip(predictions, references):
        prediction = prediction[:1] if first_character else prediction
        predicted = extract_label(prediction.strip())
        target = reference.strip().upper()[:1]
        correct += bool(predicted and target and predicted == target)
    return correct / len(references) if references else 0.0


def numeric_accuracy(predictions: list[str], references: list[str]) -> float:
    correct = 0
    for prediction, reference in zip(predictions, references):
        match = re.search(
            r"\b(?:final\s+)?answer\s*(?:is|:|=)?\s*([-+]?\d+(?:\.\d+)?)",
            prediction,
            re.IGNORECASE,
        )
        if not match:
            match = re.search(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)(?![\w.])", prediction)
        predicted = match.group(1) if match else prediction.strip()
        correct += predicted == reference.strip()
    return correct / len(references) if references else 0.0


def rouge_l(predictions: list[str], references: list[str]) -> float:
    from rouge import Rouge

    scorer = Rouge(metrics=["rouge-l"])
    scores = [
        scorer.get_scores(reference, prediction, avg=True)["rouge-l"]["f"]
        for prediction, reference in zip(predictions, references)
        # Rouge splits sentences on periods. Dot-only or whitespace-only text
        # has no content to score; count it as zero in the original denominator.
        if prediction.replace(".", "").strip() and reference.replace(".", "").strip()
    ]
    return sum(scores) / len(predictions) if predictions else 0.0


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _normalize(sentence: str) -> str:
    import sacrebleu

    sentence = sentence.lower()
    if version.parse(sacrebleu.__version__).major >= 2:
        return sacrebleu.metrics.bleu._get_tokenizer("13a")()(sentence)
    return sacrebleu.TOKENIZERS["13a"]()(sentence)


def _sari_ngram(source, candidate, references, num_references):
    references_all = [gram for reference in references for gram in reference]
    reference_counter = Counter(references_all)
    source_counter = Counter(source)
    source_repeated = Counter(
        {gram: count * num_references for gram, count in source_counter.items()}
    )
    candidate_counter = Counter(candidate)
    candidate_repeated = Counter(
        {gram: count * num_references for gram, count in candidate_counter.items()}
    )

    keep = source_repeated & candidate_repeated
    keep_good = keep & reference_counter
    keep_all = source_repeated & reference_counter
    keep_precision = (
        1.0
        if not keep
        else sum(keep_good[gram] / keep[gram] for gram in keep_good) / len(keep)
    )
    keep_recall = (
        1.0 if not keep_all else sum(keep_good.values()) / sum(keep_all.values())
    )
    keep_score = (
        2 * keep_precision * keep_recall / (keep_precision + keep_recall)
        if keep_precision > 0 or keep_recall > 0
        else 0.0
    )

    deleted = source_repeated - candidate_repeated
    deleted_good = deleted - reference_counter
    delete_precision = (
        1.0
        if not deleted
        else sum(deleted_good[gram] / deleted[gram] for gram in deleted_good)
        / len(deleted)
    )

    added = set(candidate_counter) - set(source_counter)
    added_good = added & set(reference_counter)
    added_all = set(reference_counter) - set(source_counter)
    add_precision = 1.0 if not added else len(added_good) / len(added)
    add_recall = 1.0 if not added_all else len(added_good) / len(added_all)
    add_score = (
        2 * add_precision * add_recall / (add_precision + add_recall)
        if add_precision > 0 or add_recall > 0
        else 0.0
    )
    return keep_score, delete_precision, add_score


def sari(sources: list[str], predictions: list[str], references: list[str]) -> float:
    total = 0.0
    for source, prediction, reference in zip(sources, predictions, references):
        source_tokens = _normalize(source).split(" ")
        candidate_tokens = _normalize(prediction).split(" ")
        reference_tokens = _normalize(reference).split(" ")
        keep = delete = add = 0.0
        for ngram in range(1, 5):
            scores = _sari_ngram(
                _ngrams(source_tokens, ngram),
                _ngrams(candidate_tokens, ngram),
                [_ngrams(reference_tokens, ngram)],
                1,
            )
            keep += scores[0]
            delete += scores[1]
            add += scores[2]
        total += (keep / 4 + delete / 4 + add / 4) / 3
    return total / len(predictions) if predictions else 0.0


def task_metric(
    task: str, inputs: list[str], predictions: list[str], references: list[str]
) -> float:
    if task in {"C-STANCE", "FOMC"}:
        return label_accuracy(predictions, references)
    if task == "ScienceQA":
        return label_accuracy(predictions, references, first_character=True)
    if task in {"NumGLUE-cm", "NumGLUE-ds"}:
        return numeric_accuracy(predictions, references)
    if task == "MeetingBank":
        return rouge_l(predictions, references)
    if task == "20Minuten":
        return sari(inputs, predictions, references)
    raise ValueError(task)
