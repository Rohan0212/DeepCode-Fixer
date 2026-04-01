"""
Comprehensive validation modules for the Patch Agent pipeline.

This module provides validators for:
- Input data validation
- RCA output validation
- Retrieval validation
- LLM response validation
- Enhanced patch validation
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Generic validation result."""
    is_valid: bool
    reason: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class InputDataValidator:
    """Validates input data structure and required fields."""

    REQUIRED_FIELDS = ["func_before"]
    OPTIONAL_BUT_RECOMMENDED = ["rca_generated", "commit_msg", "func_name", "file_path"]

    @staticmethod
    def validate(entry: Dict) -> ValidationResult:
        """Validate that entry has required fields and reasonable values."""
        warnings = []
        
        # Check required fields
        missing = [field for field in InputDataValidator.REQUIRED_FIELDS if not entry.get(field)]
        if missing:
            return ValidationResult(
                False,
                f"Missing required fields: {', '.join(missing)}",
                warnings
            )

        # Check recommended fields
        missing_recommended = [
            field for field in InputDataValidator.OPTIONAL_BUT_RECOMMENDED
            if not entry.get(field)
        ]
        if missing_recommended:
            warnings.append(f"Missing recommended fields: {', '.join(missing_recommended)}")

        # Validate func_before is not empty
        func_before = entry.get("func_before", "")
        if not func_before or not func_before.strip():
            return ValidationResult(False, "func_before is empty", warnings)

        # Check reasonable length
        if len(func_before) > 50000:  # Very large function
            warnings.append("func_before is unusually large (>50k chars)")

        if len(func_before) < 10:  # Very small function
            warnings.append("func_before is unusually small (<10 chars)")

        return ValidationResult(True, "", warnings)


class RCAOutputValidator:
    """Validates RCA (Root Cause Analysis) output quality."""

    MIN_LENGTH = 20
    MAX_LENGTH = 2000
    ERROR_MARKERS = ["[ERROR]", "error:", "exception:", "failed:", "traceback"]

    @staticmethod
    def validate(rca_text: str) -> ValidationResult:
        """Validate RCA output quality."""
        warnings = []

        if not rca_text:
            return ValidationResult(False, "RCA output is empty", warnings)

        rca_text = rca_text.strip()

        # Check for error markers
        rca_lower = rca_text.lower()
        for marker in RCAOutputValidator.ERROR_MARKERS:
            if marker.lower() in rca_lower:
                return ValidationResult(
                    False,
                    f"RCA output contains error marker: {marker}",
                    warnings
                )

        # Check length
        if len(rca_text) < RCAOutputValidator.MIN_LENGTH:
            return ValidationResult(
                False,
                f"RCA output too short (<{RCAOutputValidator.MIN_LENGTH} chars)",
                warnings
            )

        if len(rca_text) > RCAOutputValidator.MAX_LENGTH:
            warnings.append(f"RCA output very long (>{RCAOutputValidator.MAX_LENGTH} chars)")

        # Check for reasonable content (not just whitespace or single word)
        words = rca_text.split()
        if len(words) < 5:
            warnings.append("RCA output has very few words (<5)")

        # Check for common RCA indicators
        rca_indicators = [
            "root cause", "vulnerability", "issue", "problem", "fix", "security",
            "exploit", "attack", "buffer", "overflow", "injection"
        ]
        has_indicator = any(indicator in rca_lower for indicator in rca_indicators)
        if not has_indicator:
            warnings.append("RCA output may lack security-related terminology")

        return ValidationResult(True, "", warnings)


class RetrievalValidator:
    """Validates retrieval results."""

    MIN_EXAMPLES = 1
    MAX_EXAMPLES = 20

    @staticmethod
    def validate(retrieved: List[Dict], expected_count: int) -> ValidationResult:
        """Validate that retrieval returned reasonable results."""
        warnings = []

        if not retrieved:
            return ValidationResult(False, "No examples retrieved", warnings)

        if len(retrieved) < RetrievalValidator.MIN_EXAMPLES:
            return ValidationResult(
                False,
                f"Too few examples retrieved ({len(retrieved)} < {RetrievalValidator.MIN_EXAMPLES})",
                warnings
            )

        if len(retrieved) > RetrievalValidator.MAX_EXAMPLES:
            warnings.append(f"Many examples retrieved ({len(retrieved)} > {RetrievalValidator.MAX_EXAMPLES})")

        if len(retrieved) != expected_count:
            warnings.append(
                f"Retrieved count ({len(retrieved)}) differs from expected ({expected_count})"
            )

        # Check that examples have required fields
        for i, ex in enumerate(retrieved):
            if not ex.get("func_before") and not ex.get("func_after"):
                warnings.append(f"Retrieved example {i} missing func_before/func_after")

        # Check similarity scores if present
        scores = [ex.get("score", 0.0) for ex in retrieved if "score" in ex]
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.3:
                warnings.append(f"Low average similarity score: {avg_score:.3f}")

        return ValidationResult(True, "", warnings)


class LLMResponseValidator:
    """Validates LLM response format and structure."""

    @staticmethod
    def validate_patch_response(response: Dict) -> ValidationResult:
        """Validate LLM patch generation response."""
        warnings = []

        if not response:
            return ValidationResult(False, "Empty LLM response", warnings)

        patch = response.get("patch", "")
        if not patch:
            return ValidationResult(False, "No patch in LLM response", warnings)

        if not patch.strip():
            return ValidationResult(False, "Patch is empty/whitespace only", warnings)

        # Check for common LLM artifacts
        if patch.startswith("```") and not patch.endswith("```"):
            warnings.append("Patch may have unclosed code block markers")

        # Check for error messages in response
        error_indicators = ["error", "cannot", "unable", "failed", "exception"]
        patch_lower = patch.lower()
        if any(indicator in patch_lower[:200] for indicator in error_indicators):
            warnings.append("Patch may contain error messages")

        return ValidationResult(True, "", warnings)

    @staticmethod
    def validate_rca_response(response: str) -> ValidationResult:
        """Validate LLM RCA generation response."""
        warnings = []

        if not response:
            return ValidationResult(False, "Empty RCA response", warnings)

        if not response.strip():
            return ValidationResult(False, "RCA response is whitespace only", warnings)

        # Check for reasonable length
        if len(response) < 20:
            warnings.append("RCA response is very short")

        if len(response) > 2000:
            warnings.append("RCA response is very long")

        return ValidationResult(True, "", warnings)


class EnhancedPatchValidator:
    """Enhanced patch validation beyond basic syntax."""

    def __init__(self, max_length: int = 50000):
        self.max_length = max_length

    def validate_syntax(self, code: str) -> ValidationResult:
        """Enhanced syntax validation."""
        warnings = []

        if not code:
            return ValidationResult(False, "Code is empty", warnings)

        # Basic checks
        if len(code) > self.max_length:
            return ValidationResult(False, f"Code exceeds max length ({self.max_length})", warnings)

        # Brace balance (already in PatchValidator, but included here for completeness)
        if not self._balanced_braces(code):
            return ValidationResult(False, "Unbalanced braces/parentheses/brackets", warnings)

        # Check for common syntax issues
        # Unclosed strings (simple check)
        if code.count('"') % 2 != 0:
            warnings.append("Possible unclosed double-quoted string")

        if code.count("'") % 2 != 0:
            warnings.append("Possible unclosed single-quoted string")

        # Check for common incomplete patterns
        incomplete_patterns = [
            r"if\s*\([^)]*$",  # Unclosed if condition
            r"for\s*\([^)]*$",  # Unclosed for loop
            r"while\s*\([^)]*$",  # Unclosed while loop
        ]
        for pattern in incomplete_patterns:
            if re.search(pattern, code, re.MULTILINE):
                warnings.append(f"Possible incomplete construct: {pattern}")

        return ValidationResult(True, "", warnings)

    def validate_structure(self, code: str) -> ValidationResult:
        """Validate code structure (function-like)."""
        warnings = []

        # Check for function-like structure
        has_function_keywords = any(
            keyword in code for keyword in ["function", "def ", "static", "void ", "int ", "return"]
        )
        if not has_function_keywords:
            warnings.append("Code may not be a function (no function keywords found)")

        # Check for reasonable line count
        lines = code.splitlines()
        if len(lines) < 2:
            warnings.append("Code has very few lines (<2)")

        if len(lines) > 1000:
            warnings.append("Code has many lines (>1000)")

        return ValidationResult(True, "", warnings)

    @staticmethod
    def _balanced_braces(text: str) -> bool:
        """Check if braces, parentheses, and brackets are balanced."""
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

