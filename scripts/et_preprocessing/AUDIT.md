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

## Second pass — new changes on `adapt-saccade-selection-criteria` (2026-10-29)

The branch added a **spatial merge criterion** (commit 1215116), made both merge
thresholds required, dropped the binocularity columns from the merge output
(62cff5f), and added `deg_to_px` / `euclidean_distance` helpers. The tests were
re-run against this code; findings below append the first pass.

### Newly confirmed bugs

#### 6. `binocularity_criterion=False` crashes: unconditional drop of annotation columns
- Location: `preprocessing.py:182`
- `merged_events.drop(columns=["has_match", "matching_event", "matching_score"])`
  runs unconditionally, but with `binocularity_criterion=False` the annotation
  is never executed and the columns do not exist → `KeyError`.
- This is exactly what `main.py`'s non-binocular path does (`run_preprocessing`,
  else-branch), so `python main.py` with `BINOCULARITY_CRITERION = False`
  currently cannot run.
- Fix: `drop(columns=[...], errors="ignore")` (also removes the pandas
  deprecation below) or drop only when the columns exist.
- Test: `tests/test_merge_fixation_candidates.py::test_binocularity_false_unannotated_input`

#### 2. (still present, relocated) empty result crashes
- Now fails one step earlier: `pd.DataFrame([])` has no columns, so the new
  `drop(...)` call raises `KeyError` before `sort_values` is reached. Same root
  cause as audit #2 in the first pass.
- Test: `tests/test_merge_fixation_candidates.py::test_all_events_dropped`

### Review notes on the new code

7. **Units trap around `SPATIAL_MERGE_THRESHOLD`**: `config.py` documents the
   value as "(in °)" (degrees of visual angle), the merge docstring says
   "(in pixels)", and the `ValueError` message says "(px)". `main.py` converts
   config degrees → pixels via `deg_to_px` before calling, so the pipeline is
   consistent — but anyone calling `merge_fixation_candidates` directly with a
   config-style value gets a silently wrong spatial criterion. Recommend one
   unit statement everywhere (the function takes pixels; config stores degrees).
8. **Cluster semantics — distance to first fixation**: the inner merge loop
   checks `euclidean_distance(events.iloc[j], events.iloc[i])`, i.e. against
   the **first** fixation of the cluster, not the previous one. A fixation 9 px
   from a cluster member but 19 px from the cluster's first fixation is
   excluded. `main.py`'s log message ("Distance to first fixation in merging
   cluster") says this is intended; it is now pinned by
   `test_spatial_distance_measured_to_first_fixation_in_cluster`.
9. **`inplace=True` deprecation**: `DataFrame.drop(..., inplace=True)`
   (`preprocessing.py:182`) is deprecated (pandas PDEP-8) and warns on every
   call. Use `merged_events = merged_events.drop(columns=[...])`.
10. **NaN fixation positions**: `euclidean_distance` returns NaN for NaN
    coordinates → comparison False → such fixations are silently never merged.
    Reasonable, now documented by `tests/test_helpers.py::test_euclidean_distance_with_nan_coords`.
11. `deg_to_px` uses `tan` (exact for visual angle) — correct; verified against
    the closed form in `tests/test_helpers.py`.
12. Merge docstring still says "Maximal Euclidean distance … for them to merge"
    while the code uses strict `<` for both thresholds — same ambiguity as
    audit #8 in the first pass, now for the spatial threshold too.

### Test suite (second pass)

```bash
uv sync --group dev          # or: use a venv with pytest, pandas, numpy
uv run -m pytest             # ~11 s (includes a full-chain run on sub-022)
```

- 45 passed, 4 `xfail(strict=True)` (confirmed bugs 1-3 and 6), ~11 s.
- Updated for the new API: all merge calls pass both thresholds; assertions on
  the dropped binocularity columns replaced by a check that the columns are
  gone (`test_binocularity_columns_dropped_from_output`, and in the smoke test).
- New tests: required-threshold `ValueError`, spatial criterion blocking a
  merge, distance-to-first-fixation cluster semantics, `deg_to_px` (known
  value + monotonicity), `euclidean_distance` (3-4-5, same point, NaN).
- The real-data smoke test now mirrors `main.py`'s degree→pixel conversion and
  still asserts the pipeline invariants (monotonic onsets, non-negative
  durations, `duration == end_time - onset` within 5 ms, kept saccades ≥ `a_min`).

## Test suite (first pass — history)

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
