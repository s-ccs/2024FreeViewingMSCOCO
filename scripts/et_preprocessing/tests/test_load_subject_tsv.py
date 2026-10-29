"""Tests for preprocessing.load_subject_tsv."""

import pandas as pd
import pytest

from preprocessing import load_subject_tsv


def test_loads_real_tsv(real_events):
    df = real_events
    assert len(df) > 0
    for col in ["onset", "end_time", "duration", "trial_type", "eye"]:
        assert col in df.columns
    assert {"fixation", "saccade", "blink"} <= set(df["trial_type"].unique())
    assert set(df["eye"].dropna().unique()) <= {"L", "R"}


def test_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_subject_tsv(tmp_path / "does_not_exist.tsv", subject_id="022")


def test_loads_minimal_tsv(tmp_path):
    p = tmp_path / "mini.tsv"
    pd.DataFrame(
        {"onset": [0.0], "end_time": [0.1], "trial_type": ["fixation"], "eye": ["L"]}
    ).to_csv(p, sep="\t", index=False)
    df = load_subject_tsv(p, subject_id="022")
    assert len(df) == 1
    assert df.iloc[0]["trial_type"] == "fixation"
