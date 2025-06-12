import mne
import numpy as np
import pandas as pd
# import sys
import json
import os.path

def calculate_missing_data(subject_id, data_root_path, output_path):
    #----
    # Load data

    #subject_id = int(sys.argv[1]) # Use command line argument
    padded_subject_id = f"{subject_id:03}"

    path = os.path.join(data_root_path, f"sub-{padded_subject_id}/ses-001/eeg/sub-{padded_subject_id}_ses-001_task-2024FreeViewingMSCOCO_eeg.set")
    raw = mne.io.read_raw_eeglab(path, preload=True)

    #----
    # Find missing data segments in the EEG data

    eeg_data = raw.get_data(units="uV")
    sample_nr = eeg_data[-1,:] # The last "channel" in the eeg_data is the sample number

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

    start_idx, end_idx = find_missing_data_segments(sample_nr)

    missing_segments = pd.DataFrame({
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_time': raw.times[start_idx],
        'end_time': raw.times[end_idx]})

    missing_segments['duration'] = missing_segments['end_time']-missing_segments['start_time']

    #----
    # Count number of trials (images shown) within the missing eeg data segments

    events = raw.annotations.to_data_frame() # load events/trigger messages

    events["onset_seconds"] = events.onset.apply(lambda x: x.timestamp())

    # Find all "Stimulus shown" events
    stim_shown_df = events[events["description"].str.contains("trigger=02")]

    # Function to count how many images were shown in the time window that has missing EEG data
    def count_missing_trials(start_time, end_time, stim_shown_events):
        return stim_shown_events.query("(onset_seconds >= @start_time) & (onset_seconds <= @end_time)").shape[0]

    missing_segments.loc[:, "count_missing_trials"] = missing_segments.apply(lambda row: count_missing_trials(row["start_time"], row["end_time"], stim_shown_df), axis=1)

    #----
    #Create overview of missing trials

    # 400 trials during the experiment + 3 practice trials
    total_nr_trials = 400 + 3

    # Count how many images (trials) were shown during interruption of the EEG stream and thereby have no EEG data
    total_missing_in_segments = missing_segments.count_missing_trials.sum()

    # For example if the experiment was aborted early and not all images were shown
    additional_missing_trials = total_nr_trials - stim_shown_df.shape[0]

    # Calculate total number of missing trials (absolute and relative)
    total_num_missing_trials = total_missing_in_segments + additional_missing_trials
    percentage_missing = round(total_num_missing_trials/total_nr_trials * 100, 2)

    missing_trials_info = {
        "participant_id": int(subject_id),
        #"missing_segments_df": missing_segments.to_dict(orient="records"),
        "total_missing_in_segments": int(total_missing_in_segments), # has to be converted to Python int because the default JSON encoder can't handle np.int64
        "additional_missing_trials": int(additional_missing_trials),
        "total_num_missing_trials": int(total_num_missing_trials),
        "percentage_missing": float(percentage_missing)
    }

    print(f"For participant {padded_subject_id}, {missing_trials_info["total_num_missing_trials"]} out of {total_nr_trials} trials ({missing_trials_info["percentage_missing"]}%) are missing.")

    #----
    # Save missing segments and missing trials info to file

    missing_segments.to_csv(output_path+f"sub-{padded_subject_id}_missing_data_segments", sep="\t", index=False)

    with open(output_path+f'sub-{padded_subject_id}_missing_trials_info.json', 'w') as f:
        f.write(json.dumps(missing_trials_info, indent=4))

    return missing_trials_info


def main():

    # Specify paths
    data_root_path = "/scratch/data/2024FreeViewingMSCOCO/"

    # Get the path to the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path where the missing trials info should be saved
    output_path = os.path.join(script_dir,"../missing_data/") 

    # Extract participant ids
    participants = pd.read_csv(os.path.join(data_root_path, "participants.tsv"), sep='\t')
    participant_list = participants.participant_id

    pilot_subjects = {"sub-770", "sub-889", "sub-890", "sub-999"}
    participant_list = [item for item in participant_list if item not in pilot_subjects]
    subject_ids = [int(participant.split('-')[1]) for participant in participant_list]

    #missing_trials_info_df = pd.DataFrame(columns=["total_missing_in_segments",
    #                                               "additional_missing_trials",
    #                                               "total_num_missing_trials",
    #                                               "percentage_missing"])

    missing_trials_info_list = []
    # calculate number of missing trials for all subjects
    for id in subject_ids:
        missing_trials_info_list.append(calculate_missing_data(id, data_root_path, output_path))
    
    missing_trials_info_all_df = pd.DataFrame(missing_trials_info_list)

    missing_trials_info_all_df.to_csv(output_path+"missing_trials_info_all_participants.tsv", sep="\t", index=False)

if __name__ == '__main__':
    main()