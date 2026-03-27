# Data curation

## Assess missing data

### Description

The `generate_missing_data_overview.py`script

- finds missing data segments (i.e. nan values) in the continuous EEG data per participant and saves their start and end in `sub-XXX_missing_data_segments.tsv` in a `sub-XXX` folder in the specified output directory (`output_path`).

- identifies which trials are affected for each missing data segment and collects their trial numbers (per participant). The resulting dictionary is saved as `sub-XXX_missing_trials_info.json` in the respective `sub-XXX`folder.

- aggregates the information about trials with missing data from all participants in a combined data frame which is saved as `missing_trials_info_all_participants.tsv` in `output_path`.

> [!NOTE]
> Note that the script uses the eeg data to assess missing data, but for "additional missing trials" (i.e. because of early abortion of the experiment) there is also no eye-tracking data.

### Output files

#### `sub-XXX_missing_data_segments.tsv`
| Column                      | Description                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| `start_idx`                 | Sample index where a missing data segment starts                                              |
| `end_idx`                   | Sample index where a missing data segment ends                                                |
| `start_time`                | Start time of the missing segment in seconds                                                  |
| `end_time`                  | End time of the missing segment in seconds                                                    |
| `duration`                  | Length of the missing data segment in seconds                                                 |
| `count_trials_with_missing` | Number of trials whose stimulus interval overlaps with this missing segment                   |
| `missing_trial_numbers`     | Trial numbers affected by this missing segment                                                |


#### `sub-XXX_missing_trials_info.json`
See `missing_trials_info_all_participants.tsv`

#### `missing_trials_info_all_participants.tsv`
| Column                                  | Description                                                  |
|-----------------------------------------|--------------------------------------------------------------|
| `participant_id`                        | Numeric participant ID                                       |
| `has_nans`                              | Whether any missing segments (NaNs) were found               |
| `count_trials_with_missing_in_segments` | Number of trials overlapping with missing segments           |
| `trials_with_missing_in_segments`       | List of affected trial numbers                               |
| `count_missing_onset_or_offset`         | Number of rials with missing onset or offset trigger         |
| `trials_missing_onset_or_offset`        | Trial numbers with missing onset or offset                   |
| `count_additional_missing_trials`       | Number of trials not presented or completely absent          |
| `additional_missing_trials`             | Completely missing trial numbers                             |
| `total_num_missing_trials`              | Sum of all missing/affected trials                           |
| `percentage_missing_trials`             | Proportion of missing/affected trials in % (0–100)           |

### Usage
1. Adjust the paths in `main()`:
  - `data_root_path` should point to the BIDS root.
  - `output_path` specifies where the missing data overview files will be saved (default: `data_root_path/derivatives/missing_data`).

2. Optionally adjust `subject_ids`in `main()` to specify for which subjects the script should be run. By default it is run for all subjects in `participants.tsv`.

3. Navigate to the `scripts/data_curation`folder and run:
```bash
    python generate_missing_data_overview.py
```