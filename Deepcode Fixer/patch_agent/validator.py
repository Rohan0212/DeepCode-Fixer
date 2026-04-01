import difflib
import re
from dataclasses import dataclass
from typing import Tuple

from .config import PatchAgentConfig


@dataclass
class PatchValidationResult:
    is_valid: bool
    reason: str = ""
    normalized_patch: str = ""


class PatchValidator:
    def __init__(self, config: PatchAgentConfig):
        self.config = config

    def normalize_patch(self, text: str) -> str:
        """Extract code block if present, otherwise return stripped text."""
        if "```" in text:
            blocks = re.findall(r"```(?:[a-zA-Z+]+)?\n(.*?)```", text, flags=re.S)
            if blocks:
                return blocks[0].strip()
        return text.strip()

    def validate(self, patch_text: str) -> PatchValidationResult:
        if not patch_text:
            return PatchValidationResult(False, "Empty patch output.")
        if len(patch_text) > self.config.max_patch_chars:
            return PatchValidationResult(False, "Patch exceeds char limit.")
        if not self._balanced_braces(patch_text):
            return PatchValidationResult(False, "Unbalanced braces detected.")
        return PatchValidationResult(True, normalized_patch=patch_text)

    def diff(self, before: str, after: str) -> str:
        diff_lines = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="func_before",
            tofile="patch_generated",
            lineterm="",
        )
        return "\n".join(diff_lines)

    @staticmethod
    def _balanced_braces(text: str) -> bool:
        stack = []
        pairs = {"{": "}", "(": ")", "[": "]"}
        closing = set(pairs.values())
        for char in text:
            if char in pairs:
                stack.append(pairs[char])
            elif char in closing:
                if not stack or stack.pop() != char:
                    return False
        return not stack


