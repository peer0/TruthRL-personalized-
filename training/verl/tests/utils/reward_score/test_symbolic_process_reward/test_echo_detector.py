import pytest
from verl.utils.reward_score.symbolic_process_reward.echo_detector import EchoDetector


class TestEchoDetector:

    def setup_method(self):
        self.detector = EchoDetector(window_size=8, threshold=0.3, penalty_scale=-1.0)

    def test_original_reasoning_no_penalty(self):
        prompt = "What is the capital of France? The query time is 01/01/2024."
        text = (
            "<think>\n"
            "Paris has been the administrative center since the medieval era. "
            "It serves as the seat of government and cultural hub.\n"
            "</think>\n\\boxed{Paris}"
        )
        result = self.detector.score(text, prompt)
        assert result >= -0.1

    def test_copied_prompt_strong_penalty(self):
        prompt = "The references state that the annual revenue was approximately fifty billion dollars in the fiscal year ending December according to the latest report"
        text = (
            "<think>\n"
            "The references state that the annual revenue was approximately fifty billion dollars "
            "in the fiscal year ending December according to the latest report. "
            "The references state that the annual revenue was approximately fifty billion dollars "
            "in the fiscal year ending December according to the latest report.\n"
            "</think>\n\\boxed{fifty billion}"
        )
        result = self.detector.score(text, prompt)
        assert result < -0.3

    def test_no_prompt_graceful_skip(self):
        text = "<think>\nSome reasoning here.\n</think>\n\\boxed{answer}"
        result = self.detector.score(text, prompt=None)
        assert result == 0.0

    def test_empty_prompt_graceful_skip(self):
        text = "<think>\nSome reasoning here.\n</think>\n\\boxed{answer}"
        result = self.detector.score(text, prompt="")
        assert result == 0.0

    def test_no_think_tags_no_penalty(self):
        prompt = "What is X?"
        text = "Just some text without think tags."
        result = self.detector.score(text, prompt)
        assert result == 0.0

    def test_returns_float_in_range(self):
        prompt = "A long prompt with many words that could potentially be echoed by the model"
        text = (
            "<think>\n"
            "A long prompt with many words that could potentially be echoed by the model "
            "and this is some additional original content.\n"
            "</think>\n\\boxed{X}"
        )
        result = self.detector.score(text, prompt)
        assert isinstance(result, float)
        assert -1.0 <= result <= 0.0
