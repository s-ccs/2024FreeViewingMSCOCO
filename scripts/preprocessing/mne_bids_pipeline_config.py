sync_eyelink = True
#sync_eventtype_regex_et = ".*Stimulus image shown.*"
sync_eventtype_regex = ".*Stimulus image shown.*"


# Default settings for data processing and analysis.

# from collections.abc import Sequence
# from typing import Annotated, Any, Callable, Literal, Optional, Union

# from annotated_types import Ge, Interval, Len, MinLen
# from mne import Covariance
# from mne_bids import BIDSPath

#from mne_bids_pipeline.typing import (
#     ArbitraryContrast,
#     DigMontageType,
#     FloatArrayLike,
#     PathLike,
# )

# %%
# # General settings
bids_root = "data"
# bids_root: Optional[PathLike] = None
# """
# Specify the BIDS root directory. Pass an empty string or ```None` to use
# the value specified in the `BIDS_ROOT` environment variable instead.
# Raises an exception if the BIDS root has not been specified.

# ???+ example "Example"
#     ``` python
#     bids_root = '/path/to/your/bids_root'  # Use this to specify a path here.
#     bids_root = None  # Make use of the `BIDS_ROOT` environment variable.
#     ```
# """

# deriv_root: Optional[PathLike] = None
# """
# The root of the derivatives directory in which the pipeline will store
# the processing results. If `None`, this will be
# `derivatives/mne-bids-pipeline` inside the BIDS root.

# interactive: bool = False
# """
# If True, the steps will provide some interactive elements, such as
# figures. If running the steps from a notebook or Spyder,
# run `%matplotlib qt` in the command line to open the figures in a separate
# window.

# !!! info
#     Enabling interactive mode deactivates parallel processing.
# """

# sessions: Union[list, Literal["all"]] = "all"
# """
# The sessions to process. If `'all'`, will process all sessions found in the
# BIDS dataset.
# """

task = "2024FreeViewingMSCOCO"
# task: str = ""
# """
# The task to process.
# """

#subjects = "all"
subjects = ["001"]
# subjects: Union[Sequence[str], Literal["all"]] = "all"
# """
# Subjects to analyze. If `'all'`, include all subjects. To only
# include a subset of subjects, pass a list of their identifiers. Even
# if you plan on analyzing only a single subject, pass their identifier
# as a list.

# Please note that if you intend to EXCLUDE only a few subjects, you
# should consider setting `subjects = 'all'` and adding the
# identifiers of the excluded subjects to `exclude_subjects` (see next
# section).

# ???+ example "Example"
#     ```python
#     subjects = 'all'  # Include all subjects.
#     subjects = ['05']  # Only include subject 05.
#     subjects = ['01', '02']  # Only include subjects 01 and 02.
#     ```
# """

exclude_subjects = []
# exclude_subjects: Sequence[str] = []
# """
# Specify subjects to exclude from analysis. The MEG empty-room mock-subject
# is automatically excluded from regular analysis.

# ???+ info "Good Practice / Advice"
#     Keep track of the criteria leading you to exclude
#     a participant (e.g. too many movements, missing blocks, aborted experiment,
#     did not understand the instructions, etc, ...)
#     The `emptyroom` subject will be excluded automatically.
# """

ch_types = ["eeg"]
# ch_types: Annotated[Sequence[Literal["meg", "mag", "grad", "eeg"]], Len(1, 4)] = []
# """
# The channel types to consider.

# ???+ example "Example"
#     ```python
#     # Use EEG channels:
#     ch_types = ['eeg']

#     # Use magnetometer and gradiometer MEG channels:
#     ch_types = ['mag', 'grad']

#     # Use MEG and EEG channels:
#     ch_types = ['meg', 'eeg']
#     ```
# """

data_type = "eeg"
# data_type: Optional[Literal["meg", "eeg"]] = None
# """
# The BIDS data type.

# For MEG recordings, this will usually be 'meg'; and for EEG, 'eeg'.
# However, if your dataset contains simultaneous recordings of MEG and EEG,
# stored in a single file, you will typically need to set this to 'meg'.
# If `None`, we will assume that the data type matches the channel type.

# ???+ example "Example"
#     The dataset contains simultaneous recordings of MEG and EEG, and we only
#     wish to process the EEG data, which is stored inside the MEG files:

#     ```python
#     ch_types = ['eeg']
#     data_type = 'meg'
#     ```

#     The dataset contains simultaneous recordings of MEG and EEG, and we only
#     wish to process the gradiometer data:

#     ```python
#     ch_types = ['grad']
#     data_type = 'meg'  # or data_type = None
#     ```

#     The dataset contains only EEG data:

#     ```python
#     ch_types = ['eeg']
#     data_type = 'eeg'  # or data_type = None
#     ```
# """

# eog_channels: Optional[Sequence[str]] = None
# """
# Specify EOG channels to use, or create virtual EOG channels.

# Allows the specification of custom channel names that shall be used as
# (virtual) EOG channels. For example, say you recorded EEG **without** dedicated
# EOG electrodes, but with some EEG electrodes placed close to the eyes, e.g.
# Fp1 and Fp2. These channels can be expected to have captured large quantities
# of ocular activity, and you might want to use them as "virtual" EOG channels,
# while also including them in the EEG analysis. By default, MNE won't know that
# these channels are suitable for recovering EOG, and hence won't be able to
# perform tasks like automated blink removal, unless a "true" EOG sensor is
# present in the data as well. Specifying channel names here allows MNE to find
# the respective EOG signals based on these channels.

# You can specify one or multiple channel names. Each will be treated as if it
# were a dedicated EOG channel, without excluding it from any other analyses.

# If `None`, only actual EOG channels will be used for EOG recovery.

# If there are multiple actual EOG channels in your data, and you only specify
# a subset of them here, only this subset will be used during processing.

# ???+ example "Example"
#     Treat `Fp1` as virtual EOG channel:
#     ```python
#     eog_channels = ['Fp1']
#     ```

#     Treat `Fp1` and `Fp2` as virtual EOG channels:
#     ```python
#     eog_channels = ['Fp1', 'Fp2']
#     ```
# """

# eeg_bipolar_channels: Optional[dict[str, tuple[str, str]]] = None
# """
# Combine two channels into a bipolar channel, whose signal is the **difference**
# between the two combined channels, and add it to the data.
# A typical use case is the combination of two EOG channels – for example, a
# left and a right horizontal EOG – into a single, bipolar EOG channel. You need
# to pass a dictionary whose **keys** are the name of the new bipolar channel you
# wish to create, and whose **values** are tuples consisting of two strings: the
# name of the channel acting as anode and the name of the channel acting as
# cathode, i.e. `{'ch_name': ('anode', 'cathode')}`. You can request
# to construct more than one bipolar channel by specifying multiple key/value
# pairs. See the examples below.

# Can also be `None` if you do not want to create bipolar channels.

# !!! info
#     The channels used to create the bipolar channels are **not** automatically
#     dropped from the data. To drop channels, set `drop_channels`.

# ???+ example "Example"
#     Combine the existing channels `HEOG_left` and `HEOG_right` into a new,
#     bipolar channel, `HEOG`:
#     ```python
#     eeg_add_bipolar_channels = {'HEOG': ('HEOG_left', 'HEOG_right')}
#     ```

#     Create two bipolar channels, `HEOG` and `VEOG`:
#     ```python
#     eeg_add_bipolar_channels = {'HEOG': ('HEOG_left', 'HEOG_right'),
#                                 'VEOG': ('VEOG_lower', 'VEOG_upper')}
#     ```
# """

task_is_rest = True
rest_epochs_overlap = 0
rest_epochs_duration = 1
epochs_tmin = 0
baseline = None

run_source_estimation = False

eeg_reference = "average"
# eeg_reference: Union[Literal["average"], str, Sequence["str"]] = "average"
# """
# The EEG reference to use. If `average`, will use the average reference,
# i.e. the average across all channels. If a string, must be the name of a single
# channel. To use multiple channels as reference, set to a list of channel names.

# ???+ example "Example"
#     Use the average reference:
#     ```python
#     eeg_reference = 'average'
#     ```

#     Use the `P9` channel as reference:
#     ```python
#     eeg_reference = 'P9'
#     ```

#     Use the average of the `P9` and `P10` channels as reference:
#     ```python
#     eeg_reference = ['P9', 'P10']
#     ```
# """

eeg_template_montage = "standard_1005"
# eeg_template_montage: Optional[Union[str, DigMontageType]] = None
# """
# In situations where you wish to process EEG data and no individual
# digitization points (measured channel locations) are available, you can apply
# a "template" montage. This means we will assume the EEG cap was placed
# either according to an international system like 10/20, or as suggested by
# the cap manufacturers in their respective manual.

# Please be aware that the actual cap placement most likely deviated somewhat
# from the template, and, therefore, source reconstruction may be impaired.

# If `None`, do not apply a template montage. If a string, must be the
# name of a built-in template montage in MNE-Python.
# You can find an overview of supported template montages at
# https://mne.tools/stable/generated/mne.channels.make_standard_montage.html

# ???+ example "Example"
#     Do not apply template montage:
#     ```python
#     eeg_template_montage = None
#     ```

#     Apply 64-channel Biosemi 10/20 template montage:
#     ```python
#     eeg_template_montage = 'biosemi64'
#     ```
# """

drop_channels = ["Bipolar64","Bipolar65","sampleNumber"]
# drop_channels: Sequence[str] = []
# """
# Names of channels to remove from the data. This can be useful, for example,
# if you have added a new bipolar channel via `eeg_bipolar_channels` and now wish
# to remove the anode, cathode, or both.

# ???+ example "Example"
#     Exclude channels `Fp1` and `Cz` from processing:
#     ```python
#     drop_channels = ['Fp1', 'Cz]
#     ```
# """

# analyze_channels = ["Pz"]
# analyze_channels: Union[
#     Literal["all", "ch_types"], Annotated[Sequence["str"], MinLen(1)]
# ] = "ch_types"
# """
# The names of the channels to analyze during ERP/ERF and time-frequency analysis
# steps. For certain paradigms, e.g. EEG ERP research, it is common to constrain
# sensor-space analysis to only a few specific sensors. If `'all'`, do not
# exclude any channels (except for those selected for removal via the
# `drop_channels` setting; use with caution as this can include things like STIM
# channels during the decoding step). If 'ch_types' (default), restrict to the
# channels listed in the `ch_types` parameter. The constraint will be applied to
# all sensor-level analyses after the preprocessing stage, but not to the
# preprocessing stage itself, nor to the source analysis stage.

# ???+ example "Example"
#     Only use channel `Pz` for ERP, evoked contrasts, time-by-time
#     decoding, and time-frequency analysis:
#     ```python
#     analyze_channels = ['Pz']
#     ```
# """

reader_extra_params = {"units": "uV"}
# reader_extra_params: dict = {}
# """
# Parameters to be passed to `read_raw_bids()` calls when importing raw data.

# ???+ example "Example"
#     Enforce units for EDF files:
#     ```python
#     reader_extra_params = {"units": "uV"}
#     ```
# """

# read_raw_bids_verbose: Optional[Literal["error"]] = None
# """
# Verbosity level to pass to `read_raw_bids(..., verbose=read_raw_bids_verbose)`.
# If you know your dataset will contain files that are not perfectly BIDS
# compliant (e.g., "Did not find any meg.json..."), you can set this to
# `'error'` to suppress warnings emitted by read_raw_bids.
# """

# plot_psd_for_runs: Union[Literal["all"], Sequence[str]] = "all"
# """
# For which runs to add a power spectral density (PSD) plot to the generated
# report. This can take a considerable amount of time if you have many long
# runs. In this case, specify the runs, or pass an empty list to disable raw PSD
# plotting.
# """

random_state = 42
# random_state: Optional[int] = 42
# """
# You can specify the seed of the random number generator (RNG).
# This setting is passed to the ICA algorithm and to the decoding function,
# ensuring reproducible results. Set to `None` to avoid setting the RNG
# to a defined state.
# """

# %%
# # Preprocessing

# ## Break detection

# find_breaks: bool = False
# """
# During an experimental run, the recording might be interrupted by breaks of
# various durations, e.g. to allow the participant to stretch, blink, and swallow
# freely. During these periods, large-scale artifacts are often picked up by the
# recording system. These artifacts can impair certain stages of processing, e.g.
# the peak-detection algorithms we use to find EOG and ECG activity. In some
# cases, even the bad channel detection algorithms might not function optimally.
# It is therefore advisable to mark such break periods for exclusion at early
# processing stages.

# If `True`, try to mark breaks by finding segments of the data where no
# experimental events have occurred. This will then add annotations with the
# description `BAD_break` to the continuous data, causing these segments to be
# ignored in all following processing steps.

# ???+ example "Example"
#     Automatically find break periods, and annotate them as `BAD_break`.
#     ```python
#     find_breaks = True
#     ```

#     Disable break detection.
#     ```python
#     find_breaks = False
#     ```
# """

# min_break_duration: float = 15.0
# """
# The minimal duration (in seconds) of a data segment without any experimental
# events for it to be considered a "break". Note that the minimal duration of the
# generated `BAD_break` annotation will typically be smaller than this, as by
# default, the annotation will not extend across the entire break.
# See [`t_break_annot_start_after_previous_event`][mne_bids_pipeline._config.t_break_annot_start_after_previous_event]
# and [`t_break_annot_stop_before_next_event`][mne_bids_pipeline._config.t_break_annot_stop_before_next_event]
# to control this behavior.

# ???+ example "Example"
#     Periods between two consecutive experimental events must span at least
#     `15` seconds for this period to be considered a "break".
#     ```python
#     min_break_duration = 15.
#     ```
# """  # noqa : E501

# t_break_annot_start_after_previous_event: float = 5.0
# """
# Once a break of at least
# [`min_break_duration`][mne_bids_pipeline._config.min_break_duration]
# seconds has been discovered, we generate a `BAD_break` annotation that does not
# necessarily span the entire break period. Instead, you will typically want to
# start it some time after the last event before the break period, as to not
# unnecessarily discard brain activity immediately following that event.

# This parameter controls how much time (in seconds) should pass after the last
# pre-break event before we start annotating the following segment of the break
# period as bad.

# ???+ example "Example"
#     Once a break period has been detected, add a `BAD_break` annotation to it,
#     starting `5` seconds after the latest pre-break event.
#     ```python
#     t_break_annot_start_after_previous_event = 5.
#     ```

#     Start the `BAD_break` annotation immediately after the last pre-break
#     event.
#     ```python
#     t_break_annot_start_after_previous_event = 0.
#     ```
# """

# t_break_annot_stop_before_next_event: float = 5.0
# """
# Similarly to how
# [`t_break_annot_start_after_previous_event`][mne_bids_pipeline._config.t_break_annot_start_after_previous_event]
# controls the "gap" between beginning of the break period and `BAD_break`
# annotation onset,  this parameter controls how far the annotation should extend
# toward the first experimental event immediately following the break period
# (in seconds). This can help not to waste a post-break trial by marking its
# pre-stimulus period as bad.

# ???+ example "Example"
#     Once a break period has been detected, add a `BAD_break` annotation to it,
#     starting `5` seconds after the latest pre-break event.
#     ```python
#     t_break_annot_start_after_previous_event = 5.
#     ```

#     Start the `BAD_break` annotation immediately after the last pre-break
#     event.
#     ```python
#     t_break_annot_start_after_previous_event = 0.
#     ```
# """

# %%
# ## Filtering & resampling

# ### Filtering
#
# It is typically better to set your filtering properties on the raw data so
# as to avoid what we call border (or edge) effects.
#
# If you use this pipeline for evoked responses, you could consider
# a low-pass filter cut-off of h_freq = 40 Hz
# and possibly a high-pass filter cut-off of l_freq = 1 Hz
# so you would preserve only the power in the 1Hz to 40 Hz band.
# Note that highpass filtering is not necessarily recommended as it can
# distort waveforms of evoked components, or simply wash out any low
# frequency that can may contain brain signal. It can also act as
# a replacement for baseline correction in Epochs. See below.
#
# If you use this pipeline for time-frequency analysis, a default filtering
# could be a high-pass filter cut-off of l_freq = 1 Hz
# a low-pass filter cut-off of h_freq = 120 Hz
# so you would preserve only the power in the 1Hz to 120 Hz band.
#
# If you need more fancy analysis, you are already likely past this kind
# of tips! 😇

l_freq = 0.1
# l_freq: Optional[float] = None
# """
# The low-frequency cut-off in the highpass filtering step.
# Keep it `None` if no highpass filtering should be applied.
# """

h_freq = 100
# h_freq: Optional[float] = 40.0
# """
# The high-frequency cut-off in the lowpass filtering step.
# Keep it `None` if no lowpass filtering should be applied.
# """

# l_trans_bandwidth: Union[float, Literal["auto"]] = "auto"
# """
# Specifies the transition bandwidth of the
# highpass filter. By default it's `'auto'` and uses default MNE
# parameters.
# """

# h_trans_bandwidth: Union[float, Literal["auto"]] = "auto"
# """
# Specifies the transition bandwidth of the
# lowpass filter. By default it's `'auto'` and uses default MNE
# parameters.
# """

# notch_freq: Optional[Union[float, Sequence[float]]] = None
# """
# Notch filter frequency. More than one frequency can be supplied, e.g. to remove
# harmonics. Keep it `None` if no notch filter should be applied.

# !!! info
#     The notch filter will be applied before high- and lowpass filtering.

# ???+ example "Example"
#     Remove line noise at 50 Hz:
#     ```python
#     notch_freq = 50
#     ```
#     Remove line noise at 50 Hz and its (sub-)harmonics
#     ```python
#     notch_freq = [25, 50, 100, 150]
#     ```
# """

# notch_trans_bandwidth: float = 1.0
# """
# Specifies the transition bandwidth of the notch filter. The default is `1.`.
# """

# notch_widths: Optional[Union[float, Sequence[float]]] = None
# """
# Specifies the width of each stop band. `None` uses the MNE default.
# """

# ### Resampling
#
# If you have acquired data with a very high sampling frequency (e.g. 2 kHz)
# you will likely want to downsample to lighten up the size of the files you
# are working with (pragmatics)
# If you are interested in typical analysis (up to 120 Hz) you can typically
# resample your data down to 500 Hz without preventing reliable time-frequency
# exploration of your data.

raw_resample_sfreq = 250
# raw_resample_sfreq: Optional[float] = None
# """
# Specifies at which sampling frequency the data should be resampled.
# If `None`, then no resampling will be done.

# ???+ example "Example"
#     ```python
#     raw_resample_sfreq = None  # no resampling
#     raw_resample_sfreq = 500  # resample to 500Hz
#     ```
# """

# ## Artifact removal

# ### SSP, ICA, and artifact regression

spatial_filter = "ica"
# spatial_filter: Optional[Literal["ssp", "ica"]] = None
# """
# Whether to use a spatial filter to detect and remove artifacts. The BIDS
# Pipeline offers the use of signal-space projection (SSP) and independent
# component analysis (ICA).

# Use `'ssp'` for SSP, `'ica'` for ICA, and `None` if you do not wish to apply
# a spatial filter for artifact removal.

# The Pipeline will try to automatically discover EOG and ECG artifacts. For SSP,
# it will then produce projection vectors that remove ("project out") these
# artifacts from the data. For ICA, the independent components related to
# EOG and ECG activity will be omitted during the signal reconstruction step in
# order to remove the artifacts. The ICA procedure can be configured in various
# ways using the configuration options you can find below.
# """

ica_reject = {'eeg': 800e-6}
# ica_reject: Optional[Union[dict[str, float], Literal["autoreject_local"]]] = None
# """
# Peak-to-peak amplitude limits to exclude epochs from ICA fitting. This allows you to
# remove strong transient artifacts from the epochs used for fitting ICA, which could
# negatively affect ICA performance.

# The parameter values are the same as for [`reject`][mne_bids_pipeline._config.reject],
# but `"autoreject_global"` is not supported. `"autoreject_local"` here behaves
# differently, too: it is only used to exclude bad epochs from ICA fitting; we do not
# perform any interpolation.

# ???+ info
#     We don't support `"autoreject_global"` here (as opposed to
#     [`reject`][mne_bids_pipeline._config.reject]) because in the past, we found that
#     rejection thresholds were too strict before running ICA, i.e., too many epochs
#     got rejected. `"autoreject_local"`, however, usually performed nicely.
#     The `autoreject` documentation
#     [recommends][https://autoreject.github.io/stable/auto_examples/plot_autoreject_workflow.html]
#     running local `autoreject` before and after ICA, which can be achieved by setting
#     both, `ica_reject` and [`reject`][mne_bids_pipeline._config.reject], to
#     `"autoreject_local"`.

# If passing a dictionary, the rejection limits will also be applied to the ECG and EOG
# epochs created to find heart beats and ocular artifacts.

# ???+ info
#     MNE-BIDS-Pipeline will automatically try to detect EOG and ECG artifacts in
#     your data, and remove them. For this to work properly, it is recommended
#     to **not** specify rejection thresholds for EOG and ECG channels here –
#     otherwise, ICA won't be able to "see" these artifacts.

# ???+ info
#     This setting is applied only to the epochs that are used for **fitting** ICA. The
#     goal is to make it easier for ICA to produce a good decomposition. After fitting,
#     ICA is applied to the epochs to be analyzed, usually with one or more components
#     removed (as to remove artifacts). But even after ICA cleaning, some epochs may still
#     contain large-amplitude artifacts. Those epochs can then be rejected by using
#     the [`reject`][mne_bids_pipeline._config.reject] parameter.

# ???+ example "Example"
#     ```python
#     ica_reject = {'grad': 10e-10, 'mag': 20e-12, 'eeg': 400e-6}
#     ica_reject = {'grad': 15e-10}
#     ica_reject = None  # no rejection before fitting ICA
#     ica_reject = "autoreject_global"  # find global (per channel type) PTP thresholds before fitting ICA
#     ica_reject = "autoreject_local"  # find local (per channel) thresholds and repair epochs before fitting ICA
#     ```
# """  # noqa: E501

ica_algorithm = "picard-extended_infomax"
# ica_algorithm: Literal[
#     "picard", "fastica", "extended_infomax", "picard-extended_infomax"
# ] = "picard"
# """
# The ICA algorithm to use. `"picard-extended_infomax"` operates `picard` such that the
# generated ICA decomposition is identical to the one generated by the extended Infomax
# algorithm (but may converge in less time).
# """

ica_l_freq = 1.0
# ica_l_freq: Optional[float] = 1.0
# """
# The cutoff frequency of the high-pass filter to apply before running ICA.
# Using a relatively high cutoff like 1 Hz will remove slow drifts from the
# data, yielding improved ICA results. Must be set to 1 Hz or above.

# Set to `None` to not apply an additional high-pass filter.

# !!! info
#       The filter will be applied to raw data which was already filtered
#       according to the `l_freq` and `h_freq` settings. After filtering, the
#       data will be epoched, and the epochs will be submitted to ICA.

# !!! info
#     The Pipeline will only allow you to perform ICA on data that has been
#     high-pass filtered with a 1 Hz cutoff or higher. This is a conscious,
#     opinionated (but partially data-driven) decision made by the developers.
#     If you have reason to challenge this behavior, please get in touch with
#     us so we can discuss.
# """

ica_max_iterations = 500
# ica_max_iterations: int = 500
# """
# Maximum number of iterations to decompose the data into independent
# components. A low number means to finish earlier, but the consequence is
# that the algorithm may not have finished converging. To ensure
# convergence, pick a high number here (e.g. 3000); yet the algorithm will
# terminate as soon as it determines that is has successfully converged, and
# not necessarily exhaust the maximum number of iterations. Note that the
# default of 200 seems to be sufficient for Picard in many datasets, because
# it converges quicker than the other algorithms; but e.g. for FastICA, this
# limit may be too low to achieve convergence.
# """

ica_n_components = 0.9999999999
# ica_n_components: Optional[Union[float, int]] = None
# """
# MNE conducts ICA as a sort of a two-step procedure: First, a PCA is run
# on the data (trying to exclude zero-valued components in rank-deficient
# data); and in the second step, the principal components are passed
# to the actual ICA. You can select how many of the total principal
# components to pass to ICA – it can be all or just a subset. This determines
# how many independent components to fit, and can be controlled via this
# setting.

# If int, specifies the number of principal components that are passed to the
# ICA algorithm, which will be the number of independent components to
# fit. It must not be greater than the rank of your data (which is typically
# the number of channels, but may be less in some cases).

# If float between 0 and 1, all principal components with cumulative
# explained variance less than the value specified here will be passed to
# ICA.

# If `None` (default), `0.999999` will be used to avoid issues when working with
# rank-deficient data.

# This setting may drastically alter the time required to compute ICA.
# """

# ica_decim: Optional[int] = None
# """
# The decimation parameter to compute ICA. If 5 it means
# that 1 every 5 sample is used by ICA solver. The higher the faster
# it is to run but the less data you have to compute a good ICA. Set to
# `1` or `None` to not perform any decimation.
# """
ica_use_icalabel = True
icalabel_include = ["brain", "other"]
ica_use_eog_detection = False
ica_use_ecg_detection = False

# ### Amplitude-based artifact rejection
#
# ???+ info "Good Practice / Advice"
#     Have a look at your raw data and train yourself to detect a blink, a heart
#     beat and an eye movement.
#     You can do a quick average of blink data and check what the amplitude looks
#     like.

reject = {"eeg": 200e-6}
# reject: Optional[
#     Union[dict[str, float], Literal["autoreject_global", "autoreject_local"]]
# ] = None
# """
# Peak-to-peak amplitude limits to mark epochs as bad. This allows you to remove
# epochs with strong transient artifacts.

# !!! info
#       The rejection is performed **after** SSP or ICA, if any of those methods
#       is used. To reject epochs **before** fitting ICA, see the
#       [`ica_reject`][mne_bids_pipeline._config.ica_reject] setting.

# If `None` (default), do not apply artifact rejection.

# If a dictionary, manually specify rejection thresholds (see examples).
# The thresholds provided here must be at least as stringent as those in
# [`ica_reject`][mne_bids_pipeline._config.ica_reject] if using ICA. In case of
# `'autoreject_global'`, thresholds for any channel that do not meet this
# requirement will be automatically replaced with those used in `ica_reject`.

# If `"autoreject_global"`, use [`autoreject`](https://autoreject.github.io) to find
# suitable "global" rejection thresholds for each channel type, i.e., `autoreject`
# will generate a dictionary with (hopefully!) optimal thresholds for each
# channel type.

# If `"autoreject_local"`, use "local" `autoreject` to detect (and potentially repair) bad
# channels in each epoch. Use [`autoreject_n_interpolate`][mne_bids_pipeline._config.autoreject_n_interpolate]
# to control how many channels are allowed to be bad before an epoch gets dropped.

# ???+ example "Example"
#     ```python
#     reject = {"grad": 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
#     reject = {"eeg": 100e-6, "eog": 250e-6}
#     reject = None  # no rejection based on PTP amplitude
#     reject = "autoreject_global"  # find global (per channel type) PTP thresholds
#     reject = "autoreject_local"  # find local (per channel) thresholds and repair epochs
#     ```
# """  # noqa: E501

# reject_tmin: Optional[float] = None
# """
# Start of the time window used to reject epochs. If `None`, the window will
# start with the first time point. Has no effect if
# [`reject`][mne_bids_pipeline._config.reject] has been set to `"autoreject_local"`.

# ???+ example "Example"
#     ```python
#     reject_tmin = -0.1  # 100 ms before event onset.
#     ```
# """

# reject_tmax: Optional[float] = None
# """
# End of the time window used to reject epochs. If `None`, the window will end
# with the last time point. Has no effect if
# [`reject`][mne_bids_pipeline._config.reject] has been set to `"autoreject_local"`.

# ???+ example "Example"
#     ```python
#     reject_tmax = 0.3  # 300 ms after event onset.
#     ```
# """

# autoreject_n_interpolate: FloatArrayLike = [4, 8, 16]
# """
# The maximum number of bad channels in an epoch that `autoreject` local will try to
# interpolate. The optimal number among this list will be estimated using a
# cross-validation procedure; this means that the more elements are provided here, the
# longer the `autoreject` run will take. If the number of bad channels in an epoch
# exceeds this value, the channels won't be interpolated and the epoch will be dropped.

# !!! info
#     This setting only takes effect if [`reject`][mne_bids_pipeline._config.reject] has
#     been set to `"autoreject_local"`.

# !!! info
#     Channels marked as globally bad in the BIDS dataset (in `*_channels.tsv)`) will not
#     be considered (i.e., will remain marked as bad and not analyzed by autoreject).
# """

# %%
# # Reports

# ## Report generation

# report_evoked_n_time_points: Optional[int] = None
# """
# Specifies the number of time points to display for each evoked
# in the report. If `None`, it defaults to the current default in MNE-Python.

# ???+ example "Example"
#     Only display 5 time points per evoked
#     ```python
#     report_evoked_n_time_points = 5
#     ```
# """

# report_add_epochs_image_kwargs: Optional[dict] = None
# """
# Specifies the limits for the color scales of the epochs_image in the report.
# If `None`, it defaults to the current default in MNE-Python.

# ???+ example "Example"
#     Set vmin and vmax to the epochs rejection thresholds (with unit conversion):

#     ```python
#     report_add_epochs_image_kwargs = {
#         "grad": {"vmin": 0, "vmax": 1e13 * reject["grad"]},  # fT/cm
#         "mag": {"vmin": 0, "vmax": 1e15 * reject["mag"]},  # fT
#     }
#     ```
# """

# %%
# # Execution
#
# These options control how the pipeline is executed but should not affect
# what outputs get produced.

# n_jobs: int = 1
# """
# Specifies how many subjects you want to process in parallel. If `1`, disables
# parallel processing.
# """

# parallel_backend: Literal["loky", "dask"] = "loky"
# """
# Specifies which backend to use for parallel job execution. `loky` is the
# default backend used by `joblib`. `dask` requires [`Dask`](https://dask.org) to
# be installed. Ignored if [`n_jobs`][mne_bids_pipeline._config.n_jobs] is set to
# `1`.
# """

# dask_open_dashboard: bool = False
# """
# Whether to open the Dask dashboard in the default webbrowser automatically.
# Ignored if `parallel_backend` is not `'dask'`.
# """

# dask_temp_dir: Optional[PathLike] = None
# """
# The temporary directory to use by Dask. Dask places lock-files in this
# directory, and also uses it to "spill" RAM contents to disk if the amount of
# free memory in the system hits a critical low. It is recommended to point this
# to a location on a fast, local disk (i.e., not a network-attached storage) to
# ensure good performance. The directory needs to be writable and will be created
# if it does not exist.

# If `None`, will use `.dask-worker-space` inside of
# [`deriv_root`][mne_bids_pipeline._config.deriv_root].
# """

# dask_worker_memory_limit: str = "10G"
# """
# The maximum amount of RAM per Dask worker.
# """

mne_log_level = "info"
# log_level: Literal["info", "error"] = "info"
# """
# Set the pipeline logging verbosity.
# """

# mne_log_level: Literal["info", "error"] = "error"
# """
# Set the MNE-Python logging verbosity.
# """

on_error = "debug"
# on_error: Literal["continue", "abort", "debug"] = "abort"
# """
# Whether to abort processing as soon as an error occurs, continue with all other
# processing steps for as long as possible, or drop you into a debugger in case
# of an error.

# !!! info
#     Enabling debug mode deactivates parallel processing.
# """

# memory_location: Optional[Union[PathLike, bool]] = True
# """
# If not None (or False), caching will be enabled and the cache files will be
# stored in the given directory. The default (True) will use a
# `"_cache"` subdirectory (name configurable via the
# [`memory_subdir`][mne_bids_pipeline._config.memory_subdir]
# variable) in the BIDS derivative root of the dataset.
# """

# memory_subdir: str = "_cache"
# """
# The caching directory name to use if `memory_location` is `True`.
# """

# memory_file_method: Literal["mtime", "hash"] = "mtime"
# """
# The method to use for cache invalidation (i.e., detecting changes). Using the
# "modified time" reported by the filesystem (`'mtime'`, default) is very fast
# but requires that the filesystem supports proper mtime reporting. Using file
# hashes (`'hash'`) is slower and requires reading all input files but should
# work on any filesystem.
# """

# memory_verbose: int = 0
# """
# The verbosity to use when using memory. The default (0) does not print, while
# 1 will print the function calls that will be cached. See the documentation for
# the joblib.Memory class for more information."""

# config_validation: Literal["raise", "warn", "ignore"] = "raise"
# """
# How strictly to validate the configuration. Errors are always raised for
# invalid entries (e.g., not providing `ch_types`). This setting controls
# how to handle *possibly* or *likely* incorrect entries, such as likely
# misspellings (e.g., providing `session` instead of `sessions`).
# """
