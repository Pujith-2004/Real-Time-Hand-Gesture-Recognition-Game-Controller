"""
Pytest configuration file.
Ensures project root directory is added to sys.path so 'src' and 'scripts'
are importable when running `pytest -v tests/` from any runner.
"""

import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
