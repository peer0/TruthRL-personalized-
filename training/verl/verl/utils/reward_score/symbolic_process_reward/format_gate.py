import re
from enum import Enum, auto


class _State(Enum):
    START = auto()
    IN_THINK = auto()
    AFTER_THINK = auto()
    IN_BOXED = auto()
    DONE = auto()


class FormatGateDFA:
    """Hard gate that validates output envelope well-formedness.

    Checks:
    1. Exactly one <think> and one </think> tag
    2. Exactly one \\boxed{...} with non-empty content
    3. Tags appear in order: <think> -> </think> -> \\boxed{}
    4. Neither think content nor boxed content is empty
    5. No nested or duplicate <think> tags
    """

    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"
    _BOXED_PATTERN = re.compile(r"\\boxed\{")

    def check(self, text: str) -> bool:
        state = _State.START
        pos = 0
        think_content_start = -1
        think_content_end = -1

        while pos < len(text):
            if state == _State.START:
                idx = text.find(self._THINK_OPEN, pos)
                if idx == -1:
                    return False
                state = _State.IN_THINK
                pos = idx + len(self._THINK_OPEN)
                think_content_start = pos

            elif state == _State.IN_THINK:
                # Check for nested <think> — invalid
                nested_open = text.find(self._THINK_OPEN, pos)
                close_idx = text.find(self._THINK_CLOSE, pos)

                if close_idx == -1:
                    return False
                if nested_open != -1 and nested_open < close_idx:
                    return False

                think_content_end = close_idx
                think_content = text[think_content_start:think_content_end].strip()
                if len(think_content) == 0:
                    return False

                state = _State.AFTER_THINK
                pos = close_idx + len(self._THINK_CLOSE)

            elif state == _State.AFTER_THINK:
                # Check no more <think> tags
                if self._THINK_OPEN in text[pos:]:
                    return False

                # Find \boxed{
                match = self._BOXED_PATTERN.search(text, pos)
                if match is None:
                    return False

                # Extract boxed content by counting braces
                brace_start = match.end() - 1  # position of opening {
                content_start = match.end()
                depth = 1
                i = content_start
                while i < len(text) and depth > 0:
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                    i += 1

                if depth != 0:
                    return False

                boxed_content = text[content_start : i - 1].strip()
                if len(boxed_content) == 0:
                    return False

                state = _State.DONE
                pos = i

            elif state == _State.DONE:
                break

        return state == _State.DONE
