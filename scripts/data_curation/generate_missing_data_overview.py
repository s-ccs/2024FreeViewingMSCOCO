import mne
import numpy as np
import pandas as pd
from itertools import chain
import json
import os.path
import warnings

def calculate_missing_data(subject_id, data_root_path, output_path):
    #----
    # Load data

    #subject_id = int(sys.argv[1]) # Use command line argument
    padded_subject_id = f"{subject_id:03}"

    # TODO: Generalize to multiple runs
    path = os.path.join(data_root_path, f"sub-{padded_subject_id}/ses-001/eeg/sub-{padded_subject_id}_ses-001_task-freeviewing_run-1_eeg.set")
    raw = mne.io.read_raw_eeglab(path, preload=True)

    #----
    # Find missing data segments in the EEG data

    eeg_data = raw.get_data(units="uV")
    # TODO: Sample numbers look unexpected (check whether they were filtered or resampled in LSLAutoBIDS)
    sample_nr_idx = raw.ch_names.index('sampleNumber') # Find index of the sample_nr "channel"
    sample_nr = eeg_data[sample_nr_idx,:] 

    def find_missing_data_segments(data):
        start_missing = []
        end_missing = []

        # Check whether first sample is nan (if yes add to start)
        if np.isnan(data[0]):
            start_missing.append(0)
            if (not np.isnan(data[1])):
                end_missing.append(0)

        # Find transitions between existing values and nan-values
        for i in range(1,len(data)-1):
            if (np.isnan(data[i]) and not np.isnan(data[i-1])):
                start_missing.append(i)
            if (np.isnan(data[i]) and not np.isnan(data[i+1])):
                end_missing.append(i)

        # Check whether last sample is nan (if yes add to end)
        if np.isnan(data[-1]):
            end_missing.append(len(data)-1)
            if (not np.isnan(data[-2])):
                start_missing.append(len(data)-1)

        return start_missing, end_missing

    # Checks missing data based on sample_nr
    # TODO: Cannot find cases in which the sample_nr is not nan but the eeg channels are
    start_idx, end_idx = find_missing_data_segments(sample_nr)

    missing_segments = pd.DataFrame({
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_time': raw.times[start_idx],
        'end_time': raw.times[end_idx]})

    missing_segments['duration'] = missing_segments['end_time']-missing_segments['start_time']

    #----
    # Count number of trials (images shown) with missing eeg data i.e. nan values + store their trial numbers

    events = raw.annotations.to_data_frame(time_format=None) # load events/trigger messages

    # Find all "Stimulus image shown" and "Stimulus end" events
    stim_shown_df = events[events["description"].str.contains(r"trigger=02|trigger=08")]
    stim_shown_df.reset_index(inplace=True)

    # Function to extract the trial info (trigger, block, trial and image) from the event description
    def extract_trial_info(description_string):
        parts = [part.strip() for part in description_string.split('|')]
        
        trial_info = {}
        for part in parts:
            key,value = part.split("=",1)
            if "trigger" in key:
                key = "trigger"
            
            trial_info[key] = value
        
        return trial_info
    
    trial_info_df = pd.DataFrame(list(stim_shown_df["description"].apply(extract_trial_info)))
    trial_info_df["block"] = trial_info_df["block"].astype(int)
    trial_info_df["trial"] = trial_info_df["trial"].astype(int)

    # Add the additional trial info to the "Stimulus image shown" and "Stimulus end" events df
    stim_shown_df = pd.concat([stim_shown_df, trial_info_df], axis=1)

    # Split data frame in stimulus onset and offset events
    stim_onset_events = stim_shown_df.query("trigger == '02 Stimulus image shown'").copy()
    stim_offset_events = stim_shown_df.query("trigger == '08 Stimulus end'").copy()

    # Initialise list 
    trials_missing_onset_or_offset = []

    # Check whether the number of stimulus onset and offset events matches
    # If a trial is missing onset or offset, its number is saved and afterwards dropped from the df for the calculation
    if len(stim_onset_events) != len(stim_offset_events):
        warning_msg = "The number of stimulus onset and offset events does not match. " \
        "Check whether all triggers have been send correctly."

        trials_without_offset = set(stim_onset_events.trial) - set(stim_offset_events.trial)
        trials_without_onset = set(stim_offset_events.trial) - set(stim_onset_events.trial)

        incomplete_trials_msg = ""
        if trials_without_offset:
            incomplete_trials_msg += f" The following trials have an onset event but no offset event: {trials_without_offset}."
            trials_missing_onset_or_offset.extend(trials_without_offset)
            stim_onset_events.drop(stim_onset_events[stim_onset_events["trial"].isin(trials_without_offset)].index, inplace=True)
        
        if trials_without_onset:
            incomplete_trials_msg += f" The following trials have an offset event but no onset event: {trials_without_onset}."
            trials_missing_onset_or_offset.extend(trials_without_onset)
            stim_offset_events.drop(stim_offset_events[stim_offset_events["trial"].isin(trials_without_onset)].index, inplace=True)

        warnings.warn(warning_msg + incomplete_trials_msg)

    # Check that the stimulus onset/offset pairs have the same trial number
    assert (stim_onset_events["trial"].reset_index(drop=True) == stim_offset_events["trial"].reset_index(drop=True)).all(), \
        "The trial numbers of the stimulus onset and offset events do not match for at least one pair."

    # Combine onset and offset events in one df which one row per trial
    trial_df = pd.DataFrame({
        "block": stim_onset_events["block"].values,
        "trial": stim_onset_events["trial"].values,
        "stim_onset": stim_onset_events["onset"].values,
        "stim_offset": stim_offset_events["onset"].values
    })

    # Function to find the trial numbers of trials which have missing EEG data + count them
    # Applied per missing data segment (i.e. its start_time and end_time)
    def find_trials_with_missing(start_time, end_time, trial_df):
        trials_with_missing_df = trial_df.query(
            "(@start_time <= stim_onset <= @end_time) | \
            (@start_time <= stim_offset <= @end_time) | \
            (stim_onset <= @start_time <= stim_offset) | \
            (stim_onset <= @end_time <= stim_offset)")
        count_trials_with_missing = len(trials_with_missing_df)
        missing_trial_numbers = list(trials_with_missing_df["trial"])
        return count_trials_with_missing, missing_trial_numbers
    
    if missing_segments.empty: # If there are no missing segments, add empty columns
        missing_segments = missing_segments.assign(
            count_trials_with_missing = pd.Series(dtype=int),
            missing_trial_numbers = pd.Series(dtype=object),
        )
    else: # Count trials with missing in segments and store their trial numbers
        missing_segments[["count_trials_with_missing","missing_trial_numbers"]]= missing_segments.apply(
            lambda row: find_trials_with_missing(row["start_time"], row["end_time"], trial_df),
            axis=1,
            result_type = "expand")

    #----
    # Create overview of trials with missing data/missing trials

    # 400 trials during the experiment + 3 practice trials
    nr_practice_trials = 3
    nr_exp_trials = 400
    total_nr_trials = nr_practice_trials + nr_exp_trials

    # Create list of trials that should theoretically exist
    practice_trials = range(-nr_practice_trials, 0)
    exp_trials = range(1, nr_exp_trials + 1)
    all_trials = list(chain(practice_trials, exp_trials))

    # Count how many images (trials) were shown during interruption of the EEG stream and thereby have no/missing EEG data
    count_trials_with_missing_in_segments = missing_segments.count_trials_with_missing.sum()

    # Count trials that have a missing onset or offset event
    count_missing_onset_or_offset = len(trials_missing_onset_or_offset)

    # For example if the experiment was aborted early and not all images were shown
    count_additional_missing_trials = total_nr_trials - (len(trial_df) + count_missing_onset_or_offset)

    # Calculate total number of missing trials (absolute and relative)
    total_num_missing_trials = count_trials_with_missing_in_segments + count_missing_onset_or_offset + count_additional_missing_trials
    percentage_missing_trials = round(total_num_missing_trials/total_nr_trials * 100, 2)

    # Extract trial numbers of missing trials
    trials_with_missing_in_segments = list(chain.from_iterable(missing_segments["missing_trial_numbers"]))

    existing_trials = set(stim_shown_df['trial'].astype(int))
    additional_missing_trials = list(set(all_trials)-existing_trials)

    missing_trials_info = {
        "participant_id": int(subject_id),
        "has_nans": False if missing_segments.empty else True,
        "count_trials_with_missing_in_segments": int(count_trials_with_missing_in_segments), # has to be converted to Python int because the default JSON encoder can't handle np.int64
        "trials_with_missing_in_segments": trials_with_missing_in_segments,
        "count_missing_onset_or_offset": count_missing_onset_or_offset,
        "trials_missing_onset_or_offset": trials_missing_onset_or_offset,
        "count_additional_missing_trials": int(count_additional_missing_trials),
        "additional_missing_trials": additional_missing_trials,
        "total_num_missing_trials": int(total_num_missing_trials),
        "percentage_missing_trials": float(percentage_missing_trials)
    }

    print(f"For participant {padded_subject_id}, {missing_trials_info["total_num_missing_trials"]} out of {total_nr_trials} trials ({missing_trials_info["percentage_missing_trials"]}%) are missing.")

    #----
    # Save missing segments and missing trials info to file

    missing_segments.to_csv(os.path.join(output_path,f"sub-{padded_subject_id}_missing_data_segments.tsv"), sep="\t", index=False)

    with open(os.path.join(output_path,f'sub-{padded_subject_id}_missing_trials_info.json'), 'w') as f:
        f.write(json.dumps(missing_trials_info, indent=4))

    return missing_trials_info

def generate_missing_data_overview_plot(missing_trials_info_all_df, output_path):

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    missing_trials_info_all_df["max_num_trials"] = 403
    missing_trials_info_all_df["existing_num_trials"] = missing_trials_info_all_df["max_num_trials"]-missing_trials_info_all_df["total_num_missing_trials"]
    missing_trials_info_all_df.sort_values(by="percentage_missing_trials", inplace=True)

    plt.figure(figsize=(12, 5))

    plt.bar(missing_trials_info_all_df["participant_id"].astype(str),missing_trials_info_all_df["existing_num_trials"], width=0.5)
    plt.bar(missing_trials_info_all_df["participant_id"].astype(str),missing_trials_info_all_df["count_trials_with_missing_in_segments"],
            bottom=missing_trials_info_all_df["existing_num_trials"],
            width=0.5, label="Trials with missing EEG data", color="lightgray")
    plt.bar(missing_trials_info_all_df["participant_id"].astype(str),missing_trials_info_all_df["count_additional_missing_trials"],
            bottom=missing_trials_info_all_df["existing_num_trials"]+missing_trials_info_all_df["count_trials_with_missing_in_segments"],
            width=0.5, color="gray", label="Additional missing trials \n(e.g. early termination of the experiment)")
    plt.bar(missing_trials_info_all_df["participant_id"].astype(str),missing_trials_info_all_df["count_missing_onset_or_offset"],
            bottom=missing_trials_info_all_df["existing_num_trials"]+
            missing_trials_info_all_df["count_trials_with_missing_in_segments"]+
            missing_trials_info_all_df["count_additional_missing_trials"],
            width=0.5, color="black", label="Trials with missing onset or offset trigger")
    plt.axhline(y=0.65*403, linestyle="--", color="black", label="Participant inclusion threshold")

    plt.xlim(-0.5, len(missing_trials_info_all_df['participant_id']) - 0.5)
    plt.xlabel("Participant ID")
    plt.ylabel("Number of trials")
    barplot_legend = plt.legend(loc="lower left", framealpha=1)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.add_artist(barplot_legend)

    colors = {True: '#FF6B6B', False: '#8A8AFF'}
    for i, label in enumerate(ax.get_xticklabels()):
        participant_id = int(label.get_text())
        has_nans = missing_trials_info_all_df.loc[missing_trials_info_all_df['participant_id'] == participant_id, 'has_nans'].values[0]
        label.set_color(colors[has_nans])
        label.set_alpha(1)

    # Create proxy artists for the tick label legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Has nans', markerfacecolor='#FF6B6B', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='No nans', markerfacecolor='#8A8AFF', markersize=10)
    ]

    # Add the second legend for the tick label colors
    ax.legend(handles=legend_elements, loc='lower right', title='Participant nan Status', framealpha=1)

    plt.savefig(os.path.join(output_path, "Missing_data_overview.svg"))

def main():

    # Specify paths
    data_root_path = "/scratch/data/2024FreeViewingMSCOCO/"
    
    # Path where the missing trials info should be saved
    output_path = os.path.join(data_root_path, "derivatives", "missing_data/")

    # Test whether output folder already exists, otherwise create it
    os.makedirs(output_path, exist_ok=True)

    # Extract participant ids
    participants = pd.read_csv(os.path.join(data_root_path, "participants.tsv"), sep='\t')
    participant_list = participants.participant_id

    # Extract subject ids
    subject_ids = [int(participant.split('-')[1]) for participant in participant_list]

    missing_trials_info_list = []
    # Calculate number of missing trials for all subjects
    for id in subject_ids:
        padded_subject_id = f"{id:03}"
        output_path_subject = os.path.join(output_path, f"sub-{padded_subject_id}")
        # Test whether subject output folder already exists, otherwise create it
        os.makedirs(output_path_subject, exist_ok=True)

        missing_trials_info_list.append(calculate_missing_data(id, data_root_path, output_path_subject))
    
    missing_trials_info_all_df = pd.DataFrame(missing_trials_info_list)

    missing_trials_info_all_df.to_csv(output_path+"missing_trials_info_all_participants.tsv", sep="\t", index=False)

    # Generate a missing data overview plot
    generate_missing_data_overview_plot(missing_trials_info_all_df, output_path)

if __name__ == '__main__':
    main()