"""Put the repo root on sys.path so tests can import `capture`, `retarget`
and `viz` without each file doing its own sys.path surgery."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
