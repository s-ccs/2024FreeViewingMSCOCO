import mne
from mne_bids import BIDSPath
import pandas as pd
import os.path
import numpy as np
import re
import shutil

# Helper functions to extract trial info from description string
def extract_trial_info(description_string):
    parts = [part.strip() for part in description_string.split('|')]
    
    trial_info = {}
    for part in parts:
        key,value = part.split("=",1)
        if "trigger" in key:
            key = "event"
        
        trial_info[key] = value
    
    return trial_info

def append_trial_info(df):
    df = df.reset_index(drop=True) # Reset index to avoid alignment issues
    return df.assign(**pd.DataFrame(df["description"].apply(extract_trial_info).to_list()))

# Function to create an event df for one subject from the annotations of their eeg data
def create_events_dataframe(subject_eeg_path):

    # Load EEG data
    raw = mne.io.read_raw_fif(subject_eeg_path)

    # Extract events from annotations
    events_temp = raw.annotations.to_data_frame(time_format=None)

    events = (
        events_temp
        .query("not description.str.contains('ET')") # Remove ET events
        .query("not description.str.contains('@')") # Remove amplifier sync events
        .drop("ch_names", axis=1) # Remove ch_names column because it's only informative for ET events and otherwise []
        .pipe(append_trial_info) # Split the description string in separate columns
        .drop("description", axis=1) # Drop description column since it's no longer needed
        .astype({"event": "str", "block": "int", "trial": "int", "image": "str"}) # Adapt column data types
    )

    return events


def main():

    # Specify file paths
    data_root_path = "/scratch/data/2024FreeViewingMSCOCO/"
    eeg_input_path = os.path.join(data_root_path, "derivatives/mne-bids-pipeline")
    et_input_path = os.path.join(data_root_path, "derivatives/et_preprocessing")
    output_path = os.path.join(data_root_path, "derivatives/preprocessed")
    print(os.path.exists(output_path))

    # Specify session, run and task
    session = 1
    padded_session = f"{session:03}"
    task = "freeviewing"
    run = 1

    # Extract all participant numbers (including excluded subjects)
    participants = pd.read_csv(os.path.join(data_root_path, "participants.tsv"), sep='\t')
    participant_list = participants.participant_id
    subject_ids = list(map(lambda x: int(x.split("-")[1]), participant_list))

    # Create a list of all participants with preprocessed EEG data
    dir_list_eeg = os.listdir(eeg_input_path)
    subject_dir_list_eeg = list(filter(lambda x: re.match(r"sub-\d{3}", x), dir_list_eeg))
    subject_ids_eeg = list(map(lambda x: int(x.split("-")[1]), subject_dir_list_eeg))

    # Create a list of all participants with preprocessed ET data
    dir_list_et = os.listdir(et_input_path)
    subject_dir_list_et = list(filter(lambda x: re.match(r"sub-\d{3}", x), dir_list_et))
    subject_ids_et = list(map(lambda x: int(x.split("-")[1]), subject_dir_list_et))

    # Find all subjects that do not have EEG or ET data
    subjects_missing_eeg = list(set(subject_ids)-set(subject_ids_eeg))
    subjects_missing_et = list(set(subject_ids)-set(subject_ids_et))

    print(f"No preprocessed EEG data found for the following subjects: {subjects_missing_eeg}.")
    print(f"No preprocessed ET data found for the following subjects: {subjects_missing_et}.")

    for subject_id in [5]:#subject_ids:
        padded_subject_id = f"{subject_id:03}"

        subject_input_path = BIDSPath(
            subject = padded_subject_id,
            session = padded_session,
            task = task,
            run = run
        )

        subject_output_path = BIDSPath(
            subject = padded_subject_id,
            session = padded_session,
            task = task,
            run = run,
            root = output_path
        )

        if subject_id in subject_ids_eeg:

            # Specify subject eeg paths and create output path if it does not exist already
            subject_eeg_input_path = subject_input_path.copy().update(root = eeg_input_path, datatype = "eeg", processing = "clean", suffix = "raw", extension = ".fif", check = False) # check = False because "raw" is not an allowed suffix
            subject_eeg_output_path = subject_output_path.copy().update(datatype="eeg")
            subject_eeg_output_path.mkdir()

            # Create and save events df
            events_df = create_events_dataframe(subject_eeg_input_path)
            events_path = subject_eeg_output_path.copy().update(suffix="events", extension = "tsv")
            events_df.to_csv(events_path, sep="\t", index=False)
            print(f"Created and saved events.tsv file for subject {subject_id}.")

            # Copy preprocessed eeg data (if it exists) from the mne-bids-pipeline derivative to the preprocessed derivative
            if os.path.exists(subject_eeg_input_path):
                shutil.copy(subject_eeg_input_path, subject_eeg_output_path.copy().update(suffix="eeg", extension="fif"))
                print(f"Copying preprocessed EEG data for subject {subject_id} to `derivatives/preprocessed`.")
            else: 
                print(f"Preprocessed EEG data file ({subject_eeg_input_path}) could not be found. Skipping.")

        if subject_id in subject_ids_et:

            # Specify subject et paths and create output path if it does not exist already
            subject_et_input_path = subject_input_path.copy().update(root = et_input_path, run = None, datatype ="misc", suffix = "et_events", extension="tsv", check = False) # check = False because "misc" is no valid `datatype`
            subject_et_output_path = subject_output_path.copy().update(datatype="misc", run = None, check = False)
            subject_et_output_path.mkdir()

            # Copy preprocessed et data (if it exists) from the et_preprocessing derivative to the preprocessed derivative
            if os.path.exists(subject_et_input_path):
                shutil.copy(subject_et_input_path, subject_et_output_path.update(suffix="et_events", extension="tsv"))
                print(f"Copying preprocessed ET data for subject {subject_id} to `derivatives/preprocessed`.")
            else:
                print(f"Preprocessed ET data file ({subject_et_input_path}) could not be found. Skipping.")



if __name__ == '__main__':
    main()