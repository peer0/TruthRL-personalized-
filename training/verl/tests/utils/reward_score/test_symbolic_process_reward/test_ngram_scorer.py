import pytest
from verl.utils.reward_score.symbolic_process_reward.ngram_scorer import NGramRepetitionScorer


class TestNGramRepetitionScorer:

    def setup_method(self):
        self.scorer = NGramRepetitionScorer(ngram_size=4, tolerance=0.1, penalty_scale=-1.0)

    def test_clean_text_no_penalty(self):
        text = (
            "<think>\n"
            "The question asks about the capital of France. "
            "Paris is a major European city known for its culture. "
            "It has been the capital since medieval times.\n"
            "</think>\n\\boxed{Paris}"
        )
        result = self.scorer.score(text)
        assert result >= -0.1

    def test_heavily_repeated_text_strong_penalty(self):
        repeated = "the model says the answer is correct. "
        text = (
            "<think>\n"
            + repeated * 20
            + "\n</think>\n\\boxed{answer}"
        )
        result = self.scorer.score(text)
        assert result < -0.3

    def test_short_text_no_penalty(self):
        text = "<think>\nYes.\n</think>\n\\boxed{Y}"
        result = self.scorer.score(text)
        assert result == 0.0

    def test_no_think_tags_no_penalty(self):
        text = "Some text without think tags"
        result = self.scorer.score(text)
        assert result == 0.0

    def test_returns_float_in_range(self):
        repeated = "this is a repeated phrase that keeps going. "
        text = "<think>\n" + repeated * 15 + "\n</think>\n\\boxed{X}"
        result = self.scorer.score(text)
        assert isinstance(result, float)
        assert -1.0 <= result <= 0.0

    def test_custom_ngram_size(self):
        scorer_2gram = NGramRepetitionScorer(ngram_size=2, tolerance=0.1, penalty_scale=-1.0)
        repeated = "yes no " * 20
        text = "<think>\n" + repeated + "\n</think>\n\\boxed{X}"
        result = scorer_2gram.score(text)
        assert result < -0.3
