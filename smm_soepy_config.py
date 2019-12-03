"""This module provides some configuration for the package."""
import sys
from pathlib import Path

import numpy as np

# We only support modern Python.
np.testing.assert_equal(sys.version_info[:2] >= (3, 6), True)

# We rely on relative paths throughout the package.
PACKAGE_DIR = Path(__file__).parent.absolute()
INIT_FILES_DIR = PACKAGE_DIR / "smm_estimagic" / "init_files"
LOG_FILES_DIR = PACKAGE_DIR / "smm_estimagic" / "logging"
