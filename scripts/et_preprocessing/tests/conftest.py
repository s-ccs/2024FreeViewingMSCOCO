"""Shared fixtures for the et_preprocessing test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_TSV = (
    REPO_ROOT
    / "data"
    / "derivatives"
    / "custom-preprocessing"
    / "sub-022"
    / "ses-001"
    / "eeg"
    / "sub-022_ses-001_task-freeviewing_run-1_events.tsv"
)

# Minimal column set required by the preprocessing functions.
ROW_DEFAULTS = {
    "onset": 0.0,
    "duration": 0.1,
    "trial_type": "fixation",
    "end_time": 0.1,
    "eye": "L",
    "fix_avg_x": 500.0,
    "fix_avg_y": 400.0,
    "fix_avg_pupil_size": 1000.0,
    "sacc_start_x": np.nan,
    "sacc_start_y": np.nan,
    "sacc_end_x": np.nan,
    "sacc_end_y": np.nan,
    "sacc_visual_angle": np.nan,
}


@pytest.fixture
def make_events():
    """Factory: build an events DataFrame from row dicts, filling defaults.

    end_time is derived from onset + duration unless given explicitly.
    """

    def _make(rows):
        out = []
        for r in rows:
            row = {**ROW_DEFAULTS, **r}
            if "end_time" not in r:
                row["end_time"] = row["onset"] + row["duration"]
            out.append(row)
        return pd.DataFrame(out)

    return _make


@pytest.fixture(scope="session")
def real_tsv_path():
    if not REAL_TSV.exists():
        pytest.skip(f"real events TSV not available: {REAL_TSV}")
    return REAL_TSV


@pytest.fixture(scope="session")
def real_events(real_tsv_path):
    from preprocessing import load_subject_tsv

    return load_subject_tsv(real_tsv_path, subject_id="022")
