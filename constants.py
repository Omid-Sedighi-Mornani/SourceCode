"""
Constants for directory paths used across the project.
"""

from pathlib import Path

# Project root directory (parent of helpers/)
PROJECT_ROOT = Path(__file__)

# Main directories
DATA_FOLDER = PROJECT_ROOT / "data"
PROCESSED_DATA_FOLDER = DATA_FOLDER / "processed"
STAN_MODEL_FOLDER = DATA_FOLDER / "stan_code"
FITTED_MODEL_FOLDER = PROJECT_ROOT / "models"
