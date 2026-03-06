"""
Report renderer: HTML and optional PDF output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from genedynamics.reports.schema import ReportData


def _to_template_context(data: ReportData) -> Dict[str, Any]:
    """Convert ReportData to template context."""
    return {
        "title": data.title,
        "generated_at": data.generated_at,
        "project_root": data.project_root,
        "experiment_sections": data.experiment_sections,
        "deploy_sections": data.deploy_sections,
        "meta": data.meta,
    }


def render_html(data: ReportData, output_path: Path) -> Path:
    """Render report to HTML file."""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        raise ImportError("jinja2 required for report generation. pip install jinja2")

    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    html = template.render(**_to_template_context(data))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_pdf(data: ReportData, output_path: Path, html_path: Optional[Path] = None) -> Optional[Path]:
    """Render report to PDF (optional, requires weasyprint or similar)."""
    try:
        import weasyprint
    except ImportError:
        return None

    if html_path is None:
        html_path = output_path.with_suffix(".html")
    render_html(data, html_path)
    weasyprint.HTML(filename=str(html_path)).write_pdf(str(output_path))
    return Path(output_path)
