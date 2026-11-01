"""Shared pytest fixtures.

Tests must pass on a fresh clone, where data/raw/ is empty because the Kaggle
CSVs are gitignored. The `synthetic_config` fixture writes small synthetic CSVs
to a temporary directory and returns the project config pointing at them.
Paths are resolved from this file, so tests do not depend on the working
directory.
"""

from pathlib import Path

import pytest

from src.data_processing import load_config
from tests.synthetic_data import make_raw_tables

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "config.yaml")


@pytest.fixture(scope="session")
def synthetic_config(tmp_path_factory) -> dict:
    """Project config whose raw_files point at synthetic CSVs."""
    config = load_config(CONFIG_PATH)
    raw_dir = tmp_path_factory.mktemp("synthetic_raw")
    tables = make_raw_tables(config["data"]["expected_row_counts"])
    raw_files = {}
    for name, df in tables.items():
        path = raw_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        raw_files[name] = str(path)
    config["data"]["raw_files"] = raw_files
    return config
