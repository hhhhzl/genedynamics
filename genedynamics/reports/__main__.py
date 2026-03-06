"""
Entry point for `python -m genedynamics.reports`.

Avoids RuntimeWarning from running `python -m genedynamics.reports.generate`.
"""

from genedynamics.reports.generate import main
import sys

if __name__ == "__main__":
    sys.exit(main())
