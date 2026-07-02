"""Shared pytest fixtures and path setup.

Ensures the project root is importable so `import core`, `import config`, etc.
resolve when pytest is invoked from anywhere.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
