#!/usr/bin/env python3
"""
Run the complete security automation pipeline: RCA Agent → Patch Agent → Report Generation.

This script orchestrates the entire workflow:
1. Generate RCA explanations for vulnerable code
2. Generate secure patches based on RCA
3. Generate a comprehensive PDF report

Usage:
    python run_full_pipeline.py [--rca-only] [--patch-only] [--report-only]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_rca_agent() -> bool:
    """Run the RCA Agent to generate root cause analyses."""
    logger.info("=" * 60)
    logger.info("Step 1: Running RCA Agent")
    logger.info("=" * 60)
    
    # Set default paths if not already set via environment variables
    if not os.getenv("RCA_AGENT_BIGVUL_PATH"):
        os.environ["RCA_AGENT_BIGVUL_PATH"] = "processed_datasets/rca_prompts/rca_bigvul_augmented.jsonl"
    if not os.getenv("RCA_AGENT_MEGAVUL_PATH"):
        os.environ["RCA_AGENT_MEGAVUL_PATH"] = "processed_datasets/rca_prompts/rca_megavul_cleaned.jsonl"
    if not os.getenv("RCA_AGENT_OUTPUT_PATH"):
        os.environ["RCA_AGENT_OUTPUT_PATH"] = "rca_agent/outputs/rca_megavul_generated.jsonl"
    
    try:
        from rca_agent import rca_agent

        rca_agent.main()
        logger.info("✅ RCA Agent completed successfully")
        return True
    except Exception as exc:
        logger.error("❌ RCA Agent failed: %s", exc)
        return False


def run_patch_agent() -> bool:
    """Run the Patch Agent to generate secure patches."""
    logger.info("=" * 60)
    logger.info("Step 2: Running Patch Agent")
    logger.info("=" * 60)
    try:
        from patch_agent import run_patch_agent

        run_patch_agent.run()
        logger.info("✅ Patch Agent completed successfully")
        return True
    except Exception as exc:
        logger.error("❌ Patch Agent failed: %s", exc)
        return False


def generate_report() -> bool:
    """Generate PDF report from patch outputs."""
    logger.info("=" * 60)
    logger.info("Step 3: Generating PDF Report")
    logger.info("=" * 60)
    try:
        from patch_agent.report import main as report_main
        import sys as sys_module

        # Temporarily override sys.argv for the report module
        original_argv = sys_module.argv
        try:
            sys_module.argv = [
                "patch_agent.report",
                "--input",
                "patch_agent/outputs/patch_megavul_generated.jsonl",
            ]
            report_main()
            logger.info("✅ PDF report generated successfully")
            return True
        finally:
            sys_module.argv = original_argv
    except Exception as exc:
        logger.error("❌ Report generation failed: %s", exc)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete RCA → Patch → Report pipeline"
    )
    parser.add_argument(
        "--rca-only",
        action="store_true",
        help="Only run the RCA Agent",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Only run the Patch Agent (assumes RCA output exists)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate the report (assumes patch output exists)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation after patch agent",
    )
    args = parser.parse_args()

    success = True

    if args.report_only:
        success = generate_report()
    elif args.patch_only:
        success = run_patch_agent()
        if success and not args.skip_report:
            success = generate_report()
    elif args.rca_only:
        success = run_rca_agent()
    else:
        # Full pipeline
        success = run_rca_agent()
        if success:
            success = run_patch_agent()
            if success and not args.skip_report:
                success = generate_report()

    if success:
        logger.info("=" * 60)
        logger.info("✅ Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("Output files:")
        logger.info("  - RCA: rca_agent/outputs/rca_megavul_generated.jsonl")
        logger.info("  - Patches: patch_agent/outputs/patch_megavul_generated.jsonl")
        logger.info("  - Report: patch_agent/outputs/patch_report.pdf")
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error("❌ Pipeline failed. Check logs above for details.")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

