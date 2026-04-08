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
# DATA_ROOT/sub-XXX/ses-001/misc/sub-XXX_ses-001_task-freeviewing_et_events.tsv
# TBD: Adapt later to the data
DATA_ROOT = Path(r"C:\Users\chris\Documents\ArbeitUni\VIS_S-CCS\FreeViewing\BIDS")

MISC_SUBDIR = "misc"  # raw + merged TSVs
PLOTS_SUBDIR = "plots"  # fplots

# BIDS specs
SESSION = "ses-001"
TASK = "freeviewing"


# 2. Subjects
# =============================================================================
# Default subject list processed when no --subjects argument is given.
SUBJECTS = [
    "005"
]  # [ "005", "006", "007", "009", "010", "011", "013", "016", "017", "018", "021", "022", "024", "025", "029", "030", "034", "035", "038", "043", "045", "060"] TBD: add "all" condition


# 3. Screen / hardware specs
# =============================================================================
# For more accessible documentation in config file
SCREEN_RES = (1920, 1080)
SCREEN_SIZE_M = (0.552, 0.307)  # in metres (width, height)
SCREEN_DIST_M = 0.729  # in metres


# 4. Preprocessing
# =============================================================================
# Half-window in ms around each blink event for flagging saccades as "near blink" in annotate_saccades_near_blinks_in_df().
BLINK_WINDOW_MS = 150.0

# Minimum saccade amplitude threshold (degrees)
# s. Hooge et al. (2022), Stage 1.: Saccades smaller than this AND shorter than T_min are dropped.
A_MIN = 1.0

# Minimum fixation duration (seconds)
# s. Hooge et al. (2022), Stage 2.: Fixations shorter than this are dropped before merging consecutive saccades.
T_MIN_FIX = 0.06


# 5. Visualisation — general
# =============================================================================
# Eye selection for all plots. Options: "all", "left", "right", "both"
BY_EYE = "all"

# Output figure file format. Options: "svg", "pdf", "eps"
OUT_FILE_FORMAT = "svg"


# 6. Visualisation — main sequence
# =============================================================================
# Whether to exclude saccades flagged as 'near a blink'
MS_DROP_NEAR_BLINKS = True

# Whether to run outlier detection and exclude outliers from the plot
MS_DROP_OUTLIERS = False

# if MS_DROP_OUTLIERS is True
MS_OUTLIER_MAD_THRESH = 4.3

# MAD threshold used by detect_main_sequence_outliers() standalone
MS_DETECT_MAD_THRESH = 3.0


# 7. Visualisation — fixation duration
# =============================================================================
# drop implausibly short fixations
FIX_DUR_MIN_MS = 60

# drop implausibly long fixations
FIX_DUR_MAX_MS = 1000

# Histogram bin width in ms
FIX_DUR_BIN_W = 20


# 8. Visualisation — saccade amplitude
# =============================================================================
# Max amplitude (deg)
SACC_AMP_MAX_DEG = 40


# 9. Visualisation — saccade duration
# =============================================================================
# Max duration (ms); none = disable clipping
SACC_DUR_MAX_MS = 120


# 10. Visualisation — angular histogram
# =============================================================================
# If True: weight bins by saccade amplitude & exclude microssacades < ANG_HIST_MICROSACC_MIN_DEG
ANG_HIST_REFINEMENT = False
ANG_HIST_MICROSACC_MIN_DEG = 1.0

#  no. bins (default 36 = 10° per bin)
ANG_HIST_BINS_POLAR = 36
ANG_HIST_BIN_WIDTH_CART = 10
