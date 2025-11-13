"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import fixtures from fixtures module
pytest_plugins = ["tests.fixtures.korean_fixtures"]
