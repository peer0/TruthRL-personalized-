import re


def _extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


class EchoDetector:
    """Detects prompt copying in reasoning chains.

    Uses sliding-window n-gram matching between prompt and reasoning content.

    Args:
        window_size: Size of n-gram window for matching. Default 8.
        threshold: Echo ratio above which penalty applies. Default 0.3.
        penalty_scale: Maximum penalty magnitude (negative). Default -1.0.
    """

    def __init__(self, window_size: int = 8, threshold: float = 0.3, penalty_scale: float = -1.0):
        self.window_size = window_size
        self.threshold = threshold
        self.penalty_scale = penalty_scale

    def score(self, text: str, prompt: str | None = None) -> float:
        if not prompt or not prompt.strip():
            return 0.0

        content = _extract_think_content(text)
        if not content.strip():
            return 0.0

        prompt_words = prompt.lower().split()
        reasoning_words = content.lower().split()

        if len(prompt_words) < self.window_size or len(reasoning_words) < self.window_size:
            return 0.0

        # Build set of all prompt n-grams
        prompt_ngrams = set()
        for i in range(len(prompt_words) - self.window_size + 1):
            prompt_ngrams.add(tuple(prompt_words[i : i + self.window_size]))

        # Check each reasoning window against prompt set
        matched = [False] * len(reasoning_words)
        for i in range(len(reasoning_words) - self.window_size + 1):
            window = tuple(reasoning_words[i : i + self.window_size])
            if window in prompt_ngrams:
                for j in range(i, i + self.window_size):
                    matched[j] = True

        matched_count = sum(matched)
        echo_ratio = matched_count / len(reasoning_words)

        if echo_ratio <= self.threshold:
            return 0.0

        normalized = (echo_ratio - self.threshold) / (1.0 - self.threshold)
        return normalized * self.penalty_scale
