"""Keep development-report exports outside the manuscript source tree."""
import argparse
from pathlib import Path


MANUSCRIPT_ROOT = Path(__file__).resolve().parents[3] / "latex"


def report_export_directory(value):
    """Resolve an explicit export directory and reject manuscript destinations."""
    destination = Path(value).resolve()
    if destination.is_relative_to(MANUSCRIPT_ROOT.resolve()):
        raise argparse.ArgumentTypeError(
            "experiment reports must stay outside latex/; use a results/_reports "
            "directory or docs/development/cross-ebdm-experiment-reports"
        )
    return destination


def preflight_report_exports(destination, relative_paths):
    """Check all output directories and files before any export is written."""
    destination = report_export_directory(destination)
    for relative in relative_paths:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise argparse.ArgumentTypeError("report output paths must stay inside the export directory")
        candidate = destination
        for part in relative.parts:
            candidate = candidate / part
            report_export_directory(candidate)
    return destination


def add_report_export_argument(parser):
    parser.add_argument(
        "--report-export-dir", "--paper-dir", dest="report_export_dir",
        type=report_export_directory, metavar="DIR",
        help=("optional development-report LaTeX export outside latex/; "
              "--paper-dir is a compatibility alias, not a manuscript update; "
              "default: keep reports in the experiment's _reports directory"),
    )
