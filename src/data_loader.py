"""Data loader module for loading raw agricultural and meteorological records."""
from pathlib import Path
import pandas as pd
from typing import Optional
from src.config import load_config


def get_raw_data_path(custom_path: Optional[str | Path] = None) -> Path:
    """Resolve the path to the raw dataset, checking standard candidate paths."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p

    config = load_config()
    candidates = [
        Path(config["paths"]["raw_data"]),
        Path(config["paths"]["raw_data_fallback"]),
        Path("data/raw/Data(1).csv"),
        Path("data/raw/Data.csv"),
        Path("Data.csv"),
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        f"Raw data file not found. Checked candidates: {[str(c) for c in candidates]}"
    )


def load_raw_data(
    file_path: Optional[str | Path] = None,
    encoding: str = "latin1",
) -> pd.DataFrame:
    """Load raw dataset with verified encoding and whitespace-stripped columns.
    
    Parameters
    ----------
    file_path : Optional[str | Path]
        Path to CSV file. If None, resolves from config / standard locations.
    encoding : str
        File encoding, default 'latin1' to handle non-breaking space bytes.
        
    Returns
    -------
    pd.DataFrame
        Raw dataset DataFrame.
    """
    resolved_path = get_raw_data_path(file_path)
    df = pd.read_csv(resolved_path, encoding=encoding)
    df.columns = df.columns.str.strip()
    return df
