from .format_gate import FormatGateDFA
from .phase_scorer import PhaseTransitionScorer
from .ngram_scorer import NGramRepetitionScorer
from .echo_detector import EchoDetector


class SymbolicProcessReward:
    """Orchestrates symbolic process reward components.

    Pipeline: hard gate -> soft scorers -> weighted combination.

    Args:
        w_phase: Weight for phase transition score. Default 0.5.
        w_ngram: Weight for n-gram repetition penalty. Default 0.3.
        w_echo: Weight for echo detection penalty. Default 0.2.
        ngram_size: N-gram size for repetition check. Default 4.
        ngram_tolerance: Repetition tolerance before penalty. Default 0.1.
        echo_window: Window size for echo detection. Default 8.
        echo_threshold: Echo ratio threshold before penalty. Default 0.3.
        phase_alpha: Coverage vs ordering weight in phase scorer. Default 0.5.
    """

    def __init__(
        self,
        w_phase: float = 0.5,
        w_ngram: float = 0.3,
        w_echo: float = 0.2,
        ngram_size: int = 4,
        ngram_tolerance: float = 0.1,
        echo_window: int = 8,
        echo_threshold: float = 0.3,
        phase_alpha: float = 0.5,
    ):
        self.w_phase = w_phase
        self.w_ngram = w_ngram
        self.w_echo = w_echo

        self.gate = FormatGateDFA()
        self.phase_scorer = PhaseTransitionScorer(alpha=phase_alpha)
        self.ngram_scorer = NGramRepetitionScorer(
            ngram_size=ngram_size, tolerance=ngram_tolerance,
        )
        self.echo_detector = EchoDetector(
            window_size=echo_window, threshold=echo_threshold,
        )

    def score(self, solution_str: str, prompt: str | None = None) -> dict:
        """Score a solution string using all symbolic process reward components.

        Args:
            solution_str: The full model output including <think> and \\boxed{}.
            prompt: The input prompt (optional, used for echo detection).

        Returns:
            Dict with keys: process_score, gate_pass, phase_score,
            ngram_penalty, echo_penalty.
        """
        gate_pass = self.gate.check(solution_str)

        if not gate_pass:
            return {
                "process_score": 0.0,
                "gate_pass": False,
                "phase_score": 0.0,
                "ngram_penalty": 0.0,
                "echo_penalty": 0.0,
            }

        phase_score = self.phase_scorer.score(solution_str)
        ngram_penalty = self.ngram_scorer.score(solution_str)
        echo_penalty = self.echo_detector.score(solution_str, prompt)

        raw = (
            self.w_phase * phase_score
            + self.w_ngram * ngram_penalty
            + self.w_echo * echo_penalty
        )
        process_score = max(0.0, min(1.0, raw))

        return {
            "process_score": process_score,
            "gate_pass": True,
            "phase_score": phase_score,
            "ngram_penalty": ngram_penalty,
            "echo_penalty": echo_penalty,
        }
