import pytest
from verl.utils.reward_score.symbolic_process_reward.composer import SymbolicProcessReward


class TestSymbolicProcessReward:

    def setup_method(self):
        self.reward = SymbolicProcessReward()

    def test_well_formed_structured_reasoning(self):
        text = (
            "<think>\n"
            "The question asks about the population of Tokyo. "
            "Given that Tokyo is the capital of Japan, "
            "therefore it is one of the most populous cities. "
            "The answer is approximately 14 million.\n"
            "</think>\n\\boxed{14 million}"
        )
        result = self.reward.score(text)
        assert result["gate_pass"] is True
        assert result["process_score"] > 0.0
        assert result["phase_score"] > 0.0
        assert result["ngram_penalty"] >= -1.0
        assert result["echo_penalty"] >= -1.0

    def test_malformed_output_zero_score(self):
        text = "No think tags here. Just an answer."
        result = self.reward.score(text)
        assert result["gate_pass"] is False
        assert result["process_score"] == 0.0
        assert result["phase_score"] == 0.0
        assert result["ngram_penalty"] == 0.0
        assert result["echo_penalty"] == 0.0

    def test_repetitive_reasoning_penalized(self):
        repeated = "the data shows the value is important. "
        text = (
            "<think>\n"
            "The question asks about X. "
            + repeated * 20
            + "The answer is Y.\n"
            "</think>\n\\boxed{Y}"
        )
        result = self.reward.score(text)
        assert result["gate_pass"] is True
        assert result["ngram_penalty"] < 0.0

    def test_echo_detected_with_prompt(self):
        prompt = "The references state that the annual revenue of the company was approximately fifty billion dollars in the fiscal year ending December"
        text = (
            "<think>\n"
            "The references state that the annual revenue of the company was approximately fifty billion dollars "
            "in the fiscal year ending December. "
            "The references state that the annual revenue of the company was approximately fifty billion dollars "
            "in the fiscal year ending December.\n"
            "</think>\n\\boxed{fifty billion}"
        )
        result = self.reward.score(text, prompt=prompt)
        assert result["gate_pass"] is True
        assert result["echo_penalty"] < 0.0

    def test_no_prompt_echo_zero(self):
        text = (
            "<think>\n"
            "The question asks about X. Therefore Y. The answer is Z.\n"
            "</think>\n\\boxed{Z}"
        )
        result = self.reward.score(text, prompt=None)
        assert result["echo_penalty"] == 0.0

    def test_output_dict_has_all_keys(self):
        text = "<think>\nSome reasoning.\n</think>\n\\boxed{X}"
        result = self.reward.score(text)
        expected_keys = {"process_score", "gate_pass", "phase_score", "ngram_penalty", "echo_penalty"}
        assert set(result.keys()) == expected_keys

    def test_process_score_clamped_to_unit(self):
        text = (
            "<think>\n"
            "The question asks about X. Given that Y. "
            "Therefore Z. The answer is W.\n"
            "</think>\n\\boxed{W}"
        )
        result = self.reward.score(text)
        assert 0.0 <= result["process_score"] <= 1.0

    def test_custom_weights(self):
        reward_phase_only = SymbolicProcessReward(w_phase=1.0, w_ngram=0.0, w_echo=0.0)
        text = (
            "<think>\n"
            "The question asks about X. Therefore Y. The answer is Z.\n"
            "</think>\n\\boxed{Z}"
        )
        result = reward_phase_only.score(text)
        assert result["process_score"] == pytest.approx(result["phase_score"], abs=0.01)

    def test_perfect_score_reaches_one(self):
        """With default weights, perfect phase + no penalties should normalize to ~1.0."""
        text = (
            "<think>\n"
            "The question asks about the capital of France. "
            "Given that France is a European country, "
            "we know that Paris has been the capital for centuries. "
            "Therefore, based on this information, "
            "the answer is Paris.\n"
            "</think>\n\\boxed{Paris}"
        )
        result = self.reward.score(text)
        assert result["gate_pass"] is True
        assert result["phase_score"] > 0.5
        assert result["ngram_penalty"] == pytest.approx(0.0, abs=0.01)
        # After normalization, high phase + no penalties maps well above 0.7
        assert result["process_score"] > 0.7

    def test_max_penalties_reaches_zero(self):
        """Max penalties with no phase signal should normalize near 0.0."""
        repeated = "word "
        text = (
            "<think>\n"
            + repeated * 200
            + "\n</think>\n\\boxed{X}"
        )
        result = self.reward.score(text)
        assert result["gate_pass"] is True
        assert result["process_score"] < 0.3

    def test_zero_weights_return_zero(self):
        """All weights zero should safely return 0.0."""
        reward = SymbolicProcessReward(w_phase=0.0, w_ngram=0.0, w_echo=0.0)
        text = "<think>\nSome reasoning.\n</think>\n\\boxed{X}"
        result = reward.score(text)
        assert result["process_score"] == 0.0
