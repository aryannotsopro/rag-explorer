"""
tests/conftest.py - Shared pytest fixtures and configuration.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so tests can import 'config', 'models', etc.
sys.path.insert(0, str(Path(__file__).parent.parent))
