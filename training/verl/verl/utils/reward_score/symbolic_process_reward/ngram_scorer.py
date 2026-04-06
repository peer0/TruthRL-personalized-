import re


def _extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


class NGramRepetitionScorer:
    """Penalizes repetitive reasoning chains using n-gram analysis.

    Args:
        ngram_size: Size of n-grams to check. Default 4.
        tolerance: Repetition ratio below this threshold incurs no penalty. Default 0.1.
        penalty_scale: Maximum penalty magnitude (negative). Default -1.0.
    """

    def __init__(self, ngram_size: int = 4, tolerance: float = 0.1, penalty_scale: float = -1.0):
        self.ngram_size = ngram_size
        self.tolerance = tolerance
        self.penalty_scale = penalty_scale

    def score(self, text: str) -> float:
        content = _extract_think_content(text)
        if not content.strip():
            return 0.0

        words = content.lower().split()
        if len(words) < self.ngram_size:
            return 0.0

        ngrams = [tuple(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))

        repetition_ratio = 1.0 - (unique / total)
        excess = max(0.0, repetition_ratio - self.tolerance)

        if (1.0 - self.tolerance) > 0:
            normalized = excess / (1.0 - self.tolerance)
        else:
            normalized = 0.0

        return normalized * self.penalty_scale
