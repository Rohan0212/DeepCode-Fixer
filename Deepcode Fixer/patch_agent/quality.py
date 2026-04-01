from dataclasses import dataclass
from typing import Dict, List, Tuple


UNSAFE_CALLS = [
    "strcpy",
    "strcat",
    "sprintf(",
    "gets(",
    "strncpy(",
    "memcpy(",
]

SAFE_CALLS = [
    "strncpy",
    "snprintf",
    "memmove",
    "strlcpy",
    "memcpy",
]


@dataclass
class QualityResult:
    score: float
    unsafe_removed: bool
    safe_added: bool
    bounds_check_added: bool
    added_conditions: int
    deleted_lines: int
    added_lines: int

    def as_dict(self) -> Dict:
        return {
            "score": round(self.score, 3),
            "unsafe_removed": self.unsafe_removed,
            "safe_added": self.safe_added,
            "bounds_check_added": self.bounds_check_added,
            "added_conditions": self.added_conditions,
            "added_lines": self.added_lines,
            "deleted_lines": self.deleted_lines,
        }


class PatchQualityEvaluator:
    """Heuristic-based patch quality metrics."""

    def evaluate(self, func_before: str, patch: str, diff: str) -> QualityResult:
        added, removed = self._extract_diff_lines(diff)

        unsafe_removed = self._detect_removed_calls(removed, func_before)
        safe_added = self._detect_safe_calls(added)
        bounds_check_added = self._detect_bounds_checks(added)
        added_conditions = sum("if " in line for line in added)

        components = [unsafe_removed, safe_added, bounds_check_added]
        score = sum(1 for flag in components if flag) / len(components)

        return QualityResult(
            score=score,
            unsafe_removed=unsafe_removed,
            safe_added=safe_added,
            bounds_check_added=bounds_check_added,
            added_conditions=added_conditions,
            added_lines=len(added),
            deleted_lines=len(removed),
        )

    @staticmethod
    def _extract_diff_lines(diff: str) -> Tuple[List[str], List[str]]:
        added, removed = [], []
        for line in diff.splitlines():
            if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
                continue
            if line.startswith("+"):
                added.append(line[1:].strip())
            elif line.startswith("-"):
                removed.append(line[1:].strip())
        return added, removed

    @staticmethod
    def _detect_removed_calls(removed: List[str], func_before: str) -> bool:
        return any(
            call in func_before and any(call in line for line in removed)
            for call in UNSAFE_CALLS
        )

    @staticmethod
    def _detect_safe_calls(added: List[str]) -> bool:
        return any(any(call in line for call in SAFE_CALLS) for line in added)

    @staticmethod
    def _detect_bounds_checks(added: List[str]) -> bool:
        keywords = ["len", "size", "count", "capacity"]
        comparators = ["<", "<=", ">=", ">"]
        for line in added:
            if "if" not in line:
                continue
            if any(key in line for key in keywords) and any(
                comp in line for comp in comparators
            ):
                return True
        return False


