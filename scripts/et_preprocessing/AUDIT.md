# Audit: `scripts/et_preprocessing/preprocessing.py`

Audit date: 2026-10-29. Accompanied by the pytest suite in `tests/`.
Status labels: **CONFIRMED** (proven by a test, marked `xfail(strict=True)` — the
suite fails loudly once the bug is fixed, reminding you to remove the marker),
**REFUTED** (suspected during review, disproven by a test), **MINOR** (doc rot,
robustness).

## Confirmed bugs

### 1. `annotate_blink_saccades_in_df` crashes when there are no saccades
- Location: `preprocessing.py:266-267`
- `events.loc[sac_mask, "blink_saccade"] = blink_saccades` assigns an **empty
  list** when `sac_mask` selects no rows. On pandas 3.x this fails column
  validation (`np_can_hold_element` cannot put `[]` into a bool column), even
  though the mask is all-False and nothing actually needs to be written.
- Trigger: any events file without saccades (e.g. a short/broken recording, or
  a filtered slice).
- Fix: initialize `events["blink_saccade"] = False` *before* the loop (already
  done) and guard the assignment: `if blink_saccades: events.loc[...] = ...`.
- Test: `tests/test_annotate_blink_saccades_in_df.py::test_non_saccade_rows_get_false_column`

### 2. `merge_fixation_candidates` crashes when all events are dropped
- Location: `preprocessing.py:174-175`
- If every event is removed (e.g. single-eye saccades under the binocularity
  criterion, or an empty input), `rows_to_keep` is empty and
  `pd.DataFrame(rows_to_keep)` has **no columns** — the subsequent
  `sort_values(["onset", "eye"])` raises `KeyError`.
- Trigger: `binocularity_criterion=True` with no binocular saccades, or an
  empty DataFrame.
- Fix: return early (or build the DataFrame with `events.columns`) when
  `rows_to_keep` is empty.
- Test: `tests/test_merge_fixation_candidates.py::test_all_events_dropped`

### 3. Tolerance validation rejects numpy floats
- Location: `preprocessing.py:323`
- `type(tolerance_ms) not in [int, float]` uses exact type matching:
  `np.float64(20)` raises `ValueError` even though it *is* a float
  (`isinstance(np.float64(20), float)` is `True`). The module is numpy-heavy,
  so passing a numpy scalar from array math is likely.
- Fix: `isinstance(tolerance_ms, (int, float))` (and note `isinstance` accepts
  bool — add an explicit bool rejection if that matters).
- Test: `tests/test_annotate_matching_binocular_events.py::test_numpy_float_tolerance_accepted`

## Refuted during testing

### 4. "Merges across kept saccades" — NOT a bug
- Initial suspicion: the merge condition (`preprocessing.py:141-147`) checks
  only the time gap, so two fixations separated by a *retained* saccade
  (< 50 ms) might be merged across it, contradicting Hooge et al. (2022)
  (merge only across *removed* saccades).
- Reality: the merge loop only considers **consecutive rows that are both
  fixations**. A kept saccade between two fixations keeps them non-adjacent in
  the filtered DataFrame, so no merge happens. Verified by
  `test_no_merge_across_kept_saccade`.

## Minor / documentation

5. `preprocessing.py:349` — `output_df["matching_score"] = output_df["matching_score"] = ...`
   double-assignment typo. Harmless.
6. `preprocessing.py:39-41` — `load_subject_tsv` docstring says `folder_path`
   and mentions a `window_ms` parameter that does not exist. Stale.
7. `preprocessing.py:88` — `merge_fixation_candidates` docstring says "Saccades
   below both thresholds"; the drop condition is `a_min` **OR** not-binocular.
8. `preprocessing.py:83` vs `config.py:83` — the function docstring says
   "maximal time gap … for them to merge" (implies ≤), but the code uses a
   strict `<` and `config.py` documents `<`. The function docstring should say
   "strictly less than". Exact-boundary behavior is float-fragile and is not
   pinned by a test; gaps of 30 ms (merge) and 60 ms (no merge) are.
9. `config.py:25-26` — `DATA_ROOT` is assigned twice; the final value is the
   relative `./data/`, so the pipeline only works when launched from the repo
   root. Keep one assignment (or resolve relative to the file location).
10. `main.py:185-190` — if `BINOCULARITY_CRITERION` were `False`,
    `binocular_matching_tolerance` is never defined. Currently dead-safe (the
    variable is only read in the `True` branch) but fragile under refactor.
11. `main.py:465` — `--subjects all` calls `os.listdir()` without an existence
    check; crashes with an unhandled error if the input derivative directory
    is missing.
12. `preprocessing.py:22` — flat import `from config import …` means the module
    is only importable when `scripts/et_preprocessing` is on `sys.path`. The
    pytest config handles this via `pythonpath`; converting the folder to a
    package (or adding it to `src/`) would remove the constraint.
13. NaN tolerance in `annotate_blink_saccades_in_df` — blinks with NaN
    onset/end are silently skipped (NaN sorts last; comparisons are False).
    Reasonable, but undocumented and untested by design.

## Test suite

Run from the repo root:

```bash
uv sync --group dev          # or: use a venv with pytest, pandas, numpy
uv run -m pytest             # ~11 s (includes a full-chain run on sub-022)
```

- `tests/conftest.py` — synthetic-events builder + the real sub-022 TSV
  (skipped when the data file is absent).
- 36 tests across 5 files, one per public function in `preprocessing.py`.
- 3 `xfail(strict=True)` markers map 1:1 to confirmed bugs 1-3 above.
- The real-data smoke test asserts pipeline invariants: monotonic onsets,
  non-negative durations, merged fixations satisfying
  `duration == end_time - onset` (5 ms slack for file rounding), and every
  kept saccade above `a_min` with a binocular partner.
