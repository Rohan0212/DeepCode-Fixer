"""
Validation modules for the RCA Agent.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


# Re-implement validators here to avoid circular imports
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

        # Validate func_before is not empty
        func_before = entry.get("func_before", "")
        if not func_before or not func_before.strip():
            return ValidationResult(False, "func_before is empty", warnings)

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

        return ValidationResult(True, "", warnings)


@dataclass
class RCAValidationResult:
    """RCA validation result."""
    is_valid: bool
    reason: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class RCAAgentValidator:
    """Validator for RCA Agent inputs and outputs."""

    @staticmethod
    def validate_input(entry: Dict) -> RCAValidationResult:
        """Validate RCA Agent input entry."""
        result = InputDataValidator.validate(entry)
        return RCAValidationResult(
            result.is_valid,
            result.reason,
            result.warnings
        )

    @staticmethod
    def validate_output(rca_text: str) -> RCAValidationResult:
        """Validate RCA Agent output."""
        result = RCAOutputValidator.validate(rca_text)
        return RCAValidationResult(
            result.is_valid,
            result.reason,
            result.warnings
        )

