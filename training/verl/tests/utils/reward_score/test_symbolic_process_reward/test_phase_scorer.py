import pytest
from verl.utils.reward_score.symbolic_process_reward.phase_scorer import PhaseTransitionScorer


class TestPhaseTransitionScorer:

    def setup_method(self):
        self.scorer = PhaseTransitionScorer(alpha=0.5)

    def test_full_asc_chain_high_score(self):
        text = (
            "<think>\n"
            "The question asks about the capital of France. "
            "Given that France is a European country, "
            "therefore the capital is Paris. "
            "The answer is Paris.\n"
            "</think>\n\\boxed{Paris}"
        )
        result = self.scorer.score(text)
        assert result > 0.8

    def test_analysis_only_low_coverage(self):
        text = (
            "<think>\n"
            "The question asks about X. "
            "Looking at the data, we know that Y. "
            "Given that Z is true, we need to consider W.\n"
            "</think>\n\\boxed{answer}"
        )
        result = self.scorer.score(text)
        assert result < 0.5
        assert result > 0.0

    def test_backward_transition_low_ordering(self):
        text = (
            "<think>\n"
            "The answer is X. "
            "The question asks about Y. "
            "Therefore Z.\n"
            "</think>\n\\boxed{X}"
        )
        result = self.scorer.score(text)
        assert result < 0.8

    def test_no_markers_returns_zero(self):
        text = (
            "<think>\n"
            "Some random text without any recognized markers at all.\n"
            "</think>\n\\boxed{answer}"
        )
        result = self.scorer.score(text)
        assert result == 0.0

    def test_single_marker_coverage_only(self):
        text = (
            "<think>\n"
            "Therefore X is true.\n"
            "</think>\n\\boxed{X}"
        )
        result = self.scorer.score(text)
        assert result == pytest.approx(0.5 * (1 / 3) + 0.5 * 1.0, abs=0.01)

    def test_custom_alpha(self):
        scorer_coverage = PhaseTransitionScorer(alpha=1.0)
        scorer_ordering = PhaseTransitionScorer(alpha=0.0)
        text = (
            "<think>\n"
            "The question asks about X. Therefore Y.\n"
            "</think>\n\\boxed{Y}"
        )
        cov_result = scorer_coverage.score(text)
        ord_result = scorer_ordering.score(text)
        assert cov_result != ord_result

    def test_returns_float_in_range(self):
        text = (
            "<think>\n"
            "Given that X. This means Y. In conclusion Z.\n"
            "</think>\n\\boxed{Z}"
        )
        result = self.scorer.score(text)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
