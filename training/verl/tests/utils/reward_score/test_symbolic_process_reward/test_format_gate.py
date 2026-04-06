from verl.utils.reward_score.symbolic_process_reward.format_gate import FormatGateDFA


class TestFormatGateDFA:

    def setup_method(self):
        self.gate = FormatGateDFA()

    def test_well_formed_output_passes(self):
        text = "<think>\nThe question asks about X. Therefore, the answer is Y.\n</think>\n\\boxed{Y}"
        assert self.gate.check(text) is True

    def test_well_formed_with_multiline_think(self):
        text = "<think>\nStep 1: analyze.\nStep 2: synthesize.\nStep 3: conclude.\n</think>\n\\boxed{42}"
        assert self.gate.check(text) is True

    def test_missing_think_open_fails(self):
        text = "Some reasoning\n</think>\n\\boxed{Y}"
        assert self.gate.check(text) is False

    def test_missing_think_close_fails(self):
        text = "<think>\nSome reasoning\n\\boxed{Y}"
        assert self.gate.check(text) is False

    def test_missing_boxed_fails(self):
        text = "<think>\nSome reasoning\n</think>\nThe answer is Y"
        assert self.gate.check(text) is False

    def test_empty_think_content_fails(self):
        text = "<think></think>\n\\boxed{Y}"
        assert self.gate.check(text) is False

    def test_empty_boxed_content_fails(self):
        text = "<think>\nSome reasoning\n</think>\n\\boxed{}"
        assert self.gate.check(text) is False

    def test_duplicate_think_tags_fails(self):
        text = "<think>\nFirst\n</think>\n<think>\nSecond\n</think>\n\\boxed{Y}"
        assert self.gate.check(text) is False

    def test_wrong_order_boxed_before_think_close_fails(self):
        text = "<think>\nReasoning\n\\boxed{Y}\n</think>"
        assert self.gate.check(text) is False

    def test_nested_think_tags_fails(self):
        text = "<think>\nOuter\n<think>\nInner\n</think>\n</think>\n\\boxed{Y}"
        assert self.gate.check(text) is False

    def test_i_dont_know_answer_passes(self):
        text = "<think>\nI cannot determine the answer from the given information.\n</think>\n\\boxed{I don't know}"
        assert self.gate.check(text) is True
