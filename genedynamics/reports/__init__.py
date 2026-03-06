"""
Industrial-grade report generation pipeline.

Unified report from experiment results, deploy runs, and visualizations.
"""

from genedynamics.reports.schema import (
    DeploySection,
    ExperimentSection,
    ReportData,
)
from genedynamics.reports.generate import generate_report

__all__ = [
    "ReportData",
    "ExperimentSection",
    "DeploySection",
    "generate_report",
]
