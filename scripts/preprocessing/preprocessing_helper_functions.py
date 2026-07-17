import pandas as pd
import os.path

# Helper functions to extract trial info from description string
def extract_trial_info(description_string):

    if not "|" in description_string: # Exception for events like "BAD_break"
        return {"trial_type": description_string}
    
    parts = [part.strip() for part in description_string.split('|')]
    
    trial_info = {}
    for part in parts:
        key,value = part.split("=",1)
        if "trigger" in key:
            key = "trial_type"
        
        trial_info[key] = value
    
    return trial_info


# Function to extract the trial info (block, trial nr etc) for each trial and append it to the events data frame
def append_trial_info(df):
    df = df.reset_index(drop=True) # Reset index to avoid alignment issues
    return df.assign(**pd.DataFrame(df["description"].apply(extract_trial_info).to_list()))


# Function to create an event df for one subject from the annotations of their eeg data
def create_events_dataframe(raw):

    # Load EEG data
    #raw = mne.io.read_raw_fif(subject_eeg_path)

    # Extract events from annotations
    events_temp = raw.annotations.to_data_frame(time_format=None)

    events = (
        events_temp
        .query("not description.str.contains('ET')") # Remove ET events
        .query("not description.str.contains('@')") # Remove amplifier sync events
        .drop("ch_names", axis=1) # Remove ch_names column because it's only informative for ET events and otherwise []
        .pipe(append_trial_info) # Split the description string in separate columns
        .drop("description", axis=1) # Drop description column since it's no longer needed
        .astype({"trial_type": "str", "block": "Int64", "trial": "Int64", "image": "str"}) # Adapt column data types
    )

    return events

# Function to extract all participant numbers that exist in the participants.tsv file
def extract_subject_ids(data_root_path):
    participants = pd.read_csv(os.path.join(data_root_path, "participants.tsv"), sep='\t')
    participant_list = participants.participant_id
    subject_ids = list(map(lambda x: int(x.split("-")[1]), participant_list))

    return subject_ids