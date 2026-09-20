"""Build and execute the Phase 2 specification and architecture notebook."""
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor


def create_05_crop_calendar_and_lag_nb() -> nbformat.NotebookNode:
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell(
        "# Phase 2 & 3: Agro-Climatic Reference Calendar & Lag-Aware Machine Learning Specification\n\n"
        "**Project:** Rainfall-Lag-Aware Crop Yield Forecasting in India Using Explainable Temporal Machine Learning\n\n"
        "**Objective:**\n"
        "Establish the official phenological reference calendar for India's 15 Agro-Climatic Planning Zones, "
        "formulate the 16-week pre-harvest rainfall lag sequence ($w_{-16}$ to $w_{-1}$), and specify the lag-aware "
        "tabular (Lag-RF, Lag-XGBoost) and recurrent sequence (LSTM, GRU) architectures."
    ))

    cells.append(new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import torch\n\n"
        "sys.path.insert(0, str(Path.cwd()))\n\n"
        "from src.crop_calendar import build_crop_calendar_dataframe, get_crop_reference_week, get_pre_harvest_lag_weeks\n"
        "from src.lag_construction import extract_16week_lag_sequence, LAG_COLUMN_NAMES\n"
        "from src.lag_models import LagAwareTabularPipeline, extract_lag_importance_profile\n"
        "from src.sequence_models import CropYieldSequenceDataset, RiceLSTM, RiceGRU\n\n"
        "print(f'PyTorch Version: {torch.__version__}')\n"
        "print('Phase 2 & 3 modules loaded successfully.')"
    ))

    cells.append(new_markdown_cell(
        "## 1. Agro-Climatic Reference Periods\n"
        "Documenting phenological windows across India's 15 Agro-Climatic Zones based on Ministry of Agriculture / ICAR standards."
    ))

    cells.append(new_code_cell(
        "cal_df = build_crop_calendar_dataframe()\n"
        "cal_df[['Zone_ID', 'Agro_Climatic_Zone', 'Representative_States', 'Harvest_Reference_Week', 'Harvest_Month', '16_Week_Lag_Window']]"
    ))

    cells.append(new_markdown_cell(
        "## 2. 16-Week Pre-Harvest Lag Sequence Extraction & Causality Verification\n"
        "Demonstrating the exact causal backward-looking extraction: $w_{-16}$ through $w_{-1}$ where $w_{-1} < T_{\\text{ref}}$."
    ))

    cells.append(new_code_cell(
        "for state in ['Punjab', 'Kerala', 'West Bengal', 'Bihar', 'Tamil Nadu']:\n"
        "    t_ref = get_crop_reference_week(state)\n"
        "    lag_wks = get_pre_harvest_lag_weeks(t_ref, lag_window_size=16)\n"
        "    print(f'{state:15s} | Harvest Week T_ref: {t_ref:2d} | 16-Week Lag Range: Weeks {lag_wks[0]:2d} to {lag_wks[-1]:2d} (Strictly < {t_ref})')"
    ))

    cells.append(new_markdown_cell(
        "## 3. Deep Sequence Architectures: PyTorch RiceLSTM & RiceGRU\n"
        "Validating multi-input sequential tensor fusion (16-week rainfall sequence + static spatial fixed effects)."
    ))

    cells.append(new_code_cell(
        "B = 4   # Batch size\n"
        "L = 16  # Pre-harvest sequence length\n"
        "F_seq = 1\n"
        "F_static = 20\n\n"
        "dummy_seq = torch.randn(B, L, F_seq)\n"
        "dummy_static = torch.randn(B, F_static)\n\n"
        "lstm_model = RiceLSTM(seq_input_dim=F_seq, static_input_dim=F_static, hidden_dim=32, num_layers=2)\n"
        "gru_model = RiceGRU(seq_input_dim=F_seq, static_input_dim=F_static, hidden_dim=32, num_layers=2)\n\n"
        "lstm_out = lstm_model(dummy_seq, dummy_static)\n"
        "gru_out = gru_model(dummy_seq, dummy_static)\n\n"
        "print(f'LSTM Output Shape: {lstm_out.shape} -> Expected (B={B}, 1)')\n"
        "print(f'GRU Output Shape:  {gru_out.shape} -> Expected (B={B}, 1)')\n"
        "print(f'Sample LSTM Predictions: {lstm_out.squeeze().detach().numpy()}')"
    ))

    nb.cells = cells
    return nb


def main():
    root_dir = Path(__file__).resolve().parent.parent
    nb_dir = root_dir / "notebooks"
    nb_path = nb_dir / "05_crop_calendar_and_lag_spec.ipynb"

    nb = create_05_crop_calendar_and_lag_nb()
    print(f"Executing {nb_path.name}...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="venv_kernel")
    ep.preprocess(nb, {"metadata": {"path": str(root_dir)}})

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved: {nb_path}")


if __name__ == "__main__":
    main()
