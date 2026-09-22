# paths.py

from pathlib import Path

def _get_data_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data"

def get_raw_path() -> Path:
    return _get_data_root() / "raw"

def get_features_path() -> Path:
    return _get_data_root() / "features"

def get_outputs_path() -> Path:
    return _get_data_root() / "outputs"

