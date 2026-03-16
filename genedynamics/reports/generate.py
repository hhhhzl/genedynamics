"""
Report generation CLI and API.

Usage:
    python -m genedynamics.reports --input results --output reports
    python -m genedynamics.reports -i results -o reports -f html pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from genedynamics.reports.collectors import collect_all
from genedynamics.reports.schema import ReportData
from genedynamics.reports.renderer import render_html, render_pdf


def generate_report(
    results_root: Path,
    output_dir: Path,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """
    Generate report from results directory.

    Args:
        results_root: Path to results/ directory
        output_dir: Output directory for report files
        formats: List of "html", "pdf" (default: ["html"])

    Returns:
        Dict of format -> output path
    """
    if formats is None:
        formats = ["html"]

    data = collect_all(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    html_path = output_dir / "report.html"

    if "html" in formats:
        render_html(data, html_path)
        out["html"] = html_path

    if "pdf" in formats:
        pdf_path = output_dir / "report.pdf"
        p = render_pdf(data, pdf_path, html_path=html_path if "html" in formats else None)
        if p is not None:
            out["pdf"] = p

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate industrial-grade experiment report from results"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="results",
        help="Path to results directory (default: results)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reports",
        help="Output directory for report (default: reports)",
    )
    parser.add_argument(
        "--format",
        "-f",
        nargs="+",
        choices=["html", "pdf"],
        default=["html"],
        help="Output format(s) (default: html)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results_root = Path(args.input)
    if not results_root.is_absolute():
        results_root = root / results_root
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    try:
        out = generate_report(results_root, output_dir, formats=args.format)
        for fmt, path in out.items():
            print(f"Report ({fmt}): {path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
