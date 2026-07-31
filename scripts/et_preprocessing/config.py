"""
config.py
---------
Central configuration for the free-viewing eye-tracking pipeline.

Content
--------
1. Paths
2. Subjects
3. Screen / hardware specs
4. Preprocessing
6. Visualisation — general
7. Visualisation — main sequence
8. Visualisation — fixation duration
9. Visualisation — saccade amplitude
10. Visualisation — saccade duration
11. Visualisation — angular histogram
"""

from pathlib import Path

# 1. PATHS
# =============================================================================
# Root directory of the BIDS dataset.
# DATA_ROOT = Path("/scratch/data/2024FreeViewingMSCOCO")
DATA_ROOT = Path(
     r"C:\Users\chris\Documents\ArbeitUni\VIS_S-CCS\FreeViewing\BIDS")  # TBD: Delete later

# If the ET (or EEG+ET) events input file is within a derivative folder, specify which one
# If raw data should be used set it to None
# INPUT_DERIVATIVE = "custom-preprocessing"
INPUT_DERIVATIVE = "in"
# Name of the subdirectory containing the events.tsv file: eeg vs misc
# INPUT_SUBDIR = "eeg"
INPUT_SUBDIR = "misc"
INPUT_SUFFIX = "et_events"
# INPUT_SUFFIX = "events"

OUTPUT_DERIVATIVE = "et-preprocessing"
OUTPUT_SUBDIR = "eeg"
OUTPUT_SUFFIX = "events"

PLOTS_SUBDIR = "plots"

# BIDS specs
SESSION = "ses-001"
TASK = "freeviewing"
RUN = None # optional: 'None' if 'run'-specs are NOT in the filename


# Subjects
# =============================================================================
# Default subject list processed when no --subjects argument is given: 
# Format: list of strings, e.g. ["005", "006", "007"] or ["all"] for all subjects in the dataset. 
# The subject list is used to find the input files in DATA_ROOT/sub-XXX/ses-001/misc/sub-XXX_ses-001_task-freeviewing_et_events.tsv
SUBJECTS = ["all"]
#SUBJECTS = ["007","029","061"]
# ["005"]
# [ "005", "006", "007", "009", "010", "011", "013", "016", "017", "018", "021", "022", "024", "025", "029", "030", "034", "035", "038", "043", "045", "060"]
# ["all"]


# Preprocessing
# =============================================================================
# Half-window in ms around each blink event for flagging saccades as "blink saccade" in annotate_blink_saccades_in_df().
BLINK_WINDOW_MS = 50.0

# Minimum saccade amplitude threshold (degrees)
# s. Hooge et al. (2022), Stage 1.
A_MIN = 1.0 # Saccades smaller than this AND shorter than T_min are dropped. T_min = 2.2 · A_MIN + 27
T_MIN_FIX = 60
MERGE_THRESHOLD = 100 # (in ms); only fixations with a time difference < MERGE_THRESHOLD will be merged. 'False' if you y

# Generate Report (boolean): creates an html report
REPORT = True


# Visualisation — general
# =============================================================================
# Eye selection for all plots. Options: "all", "left", "right", "binocular"
BY_EYE = "right"

# Output figure file format. Options: "svg", "pdf", "eps"
OUT_FILE_FORMAT = "svg"


# Visualisation — main sequence
# =============================================================================
# Whether to include saccades flagged as 'near a blink' in the plotting.
# TRUE = Flagged Saccades are included in the plot
# FALSE = Flagged saccades are dropped
# "highlight" = Flagges saccades are highlighted in the plot
INCLUDE_BLINK_SAC = "highlight"


# Visualisation — fixation duration
# =============================================================================
# drop implausibly short fixations. Pass 'None' for no threshold.
# FIX_DUR_MIN_MS = 60
FIX_DUR_MIN_MS = 0

# drop implausibly long fixations. Pass 'None' for no threshold.
# FIX_DUR_MAX_MS = 1000
FIX_DUR_MAX_MS = 10000


# Visualisation — saccade amplitude
# =============================================================================
# Max amplitude (deg). Pass 'None' for no threshold.
# SAC_AMP_MAX_DEG = 40
SAC_AMP_MAX_DEG = 100


# Visualisation — saccade duration
# =============================================================================
# Max duration (ms). . Pass 'None' disable clipping
# SAC_DUR_MAX_MS = 120
SAC_DUR_MAX_MS = 500


# Dropout statistics in Plotting
# =============================================================================
# If True: compute and display dropout statistics in the summary figure
DROPOUT_STATS = True
