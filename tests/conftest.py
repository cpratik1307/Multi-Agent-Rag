"""Add src/ to sys.path so tests can import multi_agent_rag without pip install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
