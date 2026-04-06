import re
from enum import Enum, auto


class Phase(Enum):
    ANALYSIS = auto()
    SYNTHESIS = auto()
    CONCLUSION = auto()


_VALID_TRANSITIONS = {
    (Phase.ANALYSIS, Phase.SYNTHESIS),
    (Phase.ANALYSIS, Phase.CONCLUSION),
    (Phase.SYNTHESIS, Phase.CONCLUSION),
}

_DEFAULT_MARKERS = {
    Phase.ANALYSIS: [
        "the question asks",
        "looking at",
        "given that",
        "we need to",
        "let me consider",
        "we know",
        "according to",
    ],
    Phase.SYNTHESIS: [
        "therefore",
        "this means",
        "which suggests",
        "combining",
        "based on this",
        "this indicates",
    ],
    Phase.CONCLUSION: [
        "the answer is",
        "in conclusion",
        "thus",
        "hence",
        "finally",
    ],
}


def _extract_think_content(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


class PhaseTransitionScorer:
    """Scores reasoning chain structural quality by detecting logical phase transitions.

    Args:
        alpha: Weight between coverage (alpha) and ordering (1-alpha). Default 0.5.
        markers: Optional custom marker dict mapping Phase -> list[str].
    """

    def __init__(self, alpha: float = 0.5, markers: dict | None = None):
        self.alpha = alpha
        self.markers = markers or _DEFAULT_MARKERS

    def _detect_phases(self, content: str) -> list[tuple[int, Phase]]:
        content_lower = content.lower()
        occurrences = []
        for phase, marker_list in self.markers.items():
            for marker in marker_list:
                start = 0
                while True:
                    idx = content_lower.find(marker, start)
                    if idx == -1:
                        break
                    occurrences.append((idx, phase))
                    start = idx + len(marker)
        occurrences.sort(key=lambda x: x[0])
        return occurrences

    def _compute_coverage(self, phases: list[Phase]) -> float:
        if not phases:
            return 0.0
        unique = len(set(phases))
        return unique / 3.0

    def _compute_ordering(self, phases: list[Phase]) -> float:
        if len(phases) <= 1:
            return 1.0
        valid_count = 0
        total = len(phases) - 1
        for i in range(total):
            if (phases[i], phases[i + 1]) in _VALID_TRANSITIONS:
                valid_count += 1
        return valid_count / total

    def score(self, text: str) -> float:
        content = _extract_think_content(text)
        if not content.strip():
            return 0.0

        occurrences = self._detect_phases(content)
        if not occurrences:
            return 0.0

        phases = [phase for _, phase in occurrences]
        coverage = self._compute_coverage(phases)
        ordering = self._compute_ordering(phases)
        return self.alpha * coverage + (1 - self.alpha) * ordering
