from typing import Dict, List, Tuple

from .config import PatchAgentConfig
from .utils import truncate_text


def _format_example(idx: int, example: Dict, char_limit: int) -> str:
    vuln = truncate_text(example.get("func_before", ""), char_limit)
    patch = truncate_text(example.get("func_after") or example.get("func", ""), char_limit)
    rca = truncate_text(example.get("rca_explanation", ""), char_limit)
    commit_msg = truncate_text(example.get("commit_msg") or example.get("commit_message", ""), 200)
    return (
        f"Example {idx} Commit: {commit_msg}\n"
        f"Before:\n{vuln}\n"
        f"After:\n{patch}\n"
        f"RCA: {rca}\n"
    )


def build_patch_prompt(
    entry: Dict,
    retrieved_examples: List[Dict],
    config: PatchAgentConfig,
) -> Tuple[str, str]:
    """Create the user prompt and lightweight trace string."""
    commit_msg = entry.get("commit_msg") or entry.get("commit_message") or "(missing)"
    rca = entry.get("rca_generated") or "(missing)"
    func_before = entry.get("func_before") or ""

    examples_text = (
        "\n".join(
            _format_example(i + 1, ex, config.retrieval_example_char_limit)
            for i, ex in enumerate(retrieved_examples)
        )
        or "No close-matching examples were found."
    )

    instruction = f"""
Root Cause:
{rca}

Vulnerable Function:
{func_before}

Retrieved Reference Examples:
{examples_text}

Requirements:
- Return ONLY the fixed function first.
- Preserve the original function signature unless a security fix requires an additive change.
- Keep logic changes strictly scoped to addressing the root cause.
- If additional helpers are essential, include them directly under the function.
- Follow up with a short 'Patch Rationale:' paragraph explaining the changes.
"""

    prompt = (
        f"You are an expert security engineer. "
        f"Given a function and its root-cause analysis, generate a secure patch.\n\n"
        f"Commit Context: {commit_msg}\n"
        f"{instruction.strip()}"
    )
    return prompt, examples_text


