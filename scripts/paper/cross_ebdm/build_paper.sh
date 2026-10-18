#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT/latex/cross_ebdm"
latexmk -pdf -interaction=nonstopmode -halt-on-error intervention_calibrated_sim2real_blueprint.tex
