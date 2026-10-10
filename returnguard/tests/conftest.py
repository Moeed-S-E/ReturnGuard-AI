"""Make the returnguard package root importable from tests.

Ensures `import policy_engine`, `import train_model`, etc. resolve regardless
of the directory pytest is invoked from.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # returnguard/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
