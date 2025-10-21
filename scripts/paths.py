from pathlib import Path

# Project root = two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Other directories
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "configs"

def get_path(relative_path: str) -> Path:
    """
    Turn a relative path (from project root) into an absolute Path object.
    Example: get_path("data/test.json") → /full/path/to/project/data/test.json
    """
    return PROJECT_ROOT / relative_path