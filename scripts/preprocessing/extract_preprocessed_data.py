from mne_bids import BIDSPath
import argparse
import os.path
import re
import shutil

from preprocessing_helper_functions import *

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process subject IDs.")
    parser.add_argument(
        "--subject_ids",
        type=int,
        nargs="+",  # Accepts one or more values
        help="List of subject IDs (e.g., --subject_ids 7 12 30)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "If set, copy preprocessed files to output folder even if output files already exist for a subject. "
            "Otherwise, subjects with preprocessed data in the output folder are skipped."
        ),
    )
    args = parser.parse_args()

    # Specify file paths
    data_root_path = "/scratch/data/2024FreeViewingMSCOCO/"
    eeg_input_path = os.path.join(data_root_path, "derivatives/custom-preprocessing")
    events_input_path = os.path.join(data_root_path, "derivatives/et-preprocessing")
    output_path = os.path.join(data_root_path, "derivatives/preprocessed")
    
    # Check whether output folder exists, otherwise create it
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        print(f"Create output folder: {output_path}")

    # Specify session, run and task
    session = 1
    padded_session = f"{session:03}"
    task = "freeviewing"
    run = 1

    # Read subject_ids from command line or extract them from participants.tsv
    if args.subject_ids:
        subject_ids = args.subject_ids
        print(f"Using provided subject_ids: {subject_ids}")
    else:
        subject_ids = extract_subject_ids(data_root_path)
        print("Using subject_ids from participants.tsv file.")

    # Create a list of all participants with preprocessed EEG data
    dir_list_eeg = os.listdir(eeg_input_path)
    subject_dir_list_eeg = list(filter(lambda x: re.match(r"sub-\d{3}", x), dir_list_eeg))
    subject_ids_eeg = list(map(lambda x: int(x.split("-")[1]), subject_dir_list_eeg))

    # Create a list of all participants with preprocessed ET data/events df
    dir_list_events = os.listdir(events_input_path)
    subject_dir_list_events = list(filter(lambda x: re.match(r"sub-\d{3}", x), dir_list_events))
    subject_ids_events = list(map(lambda x: int(x.split("-")[1]), subject_dir_list_events))

    # Find all subjects that do not have EEG or ET data
    subjects_missing_eeg = list(set(subject_ids)-set(subject_ids_eeg))
    subjects_missing_events = list(set(subject_ids)-set(subject_ids_events))

    print(f"No preprocessed EEG data found for the following subjects: {subjects_missing_eeg}.")
    print(f"No preprocessed ET data found for the following subjects: {subjects_missing_events}.")

    for subject_id in subject_ids:
        padded_subject_id = f"{subject_id:03}"

        subject_input_path = BIDSPath(
            subject = padded_subject_id,
            session = padded_session,
            task = task,
            run = run,
            datatype="eeg",
        )

        subject_output_path = BIDSPath(
            subject = padded_subject_id,
            session = padded_session,
            task = task,
            run = run,
            root = output_path,
            datatype="eeg"
        )

        # Check whether subject-specific output folder exists, otherwise create it
        if (not os.path.isdir(subject_output_path)) and (subject_id in subject_ids_eeg) and (subject_id in subject_ids_events):
            subject_output_path.mkdir()
            print(f"Create subject output folder: {subject_output_path}")

        subject_eeg_output_path = subject_output_path.copy().update(suffix="eeg", extension="fif")
        subject_events_output_path = subject_output_path.copy().update(suffix="events", extension="tsv")

        if os.path.exists(subject_eeg_output_path) and os.path.exists(subject_events_output_path) and not args.overwrite:
            print(
                f"Skipping copying preprocessed data for sub-{padded_subject_id} — output files already exist: ",
                f"  {subject_eeg_output_path}",
                f"  {subject_events_output_path}",
                f"  Use --overwrite to reprocess.",
                sep="\n"
            )
            continue

        if subject_id in subject_ids_eeg:

            subject_eeg_input_path = subject_input_path.copy().update(root = eeg_input_path, datatype = "eeg", processing = "clean", suffix = "raw", extension = ".fif", check = False) # check = False because "raw" is not an allowed suffix

            if os.path.exists(subject_eeg_input_path):
                # Copy preprocessed eeg data (if it exists) from the input derivative to the preprocessed derivative
                shutil.copy(subject_eeg_input_path, subject_eeg_output_path)
                print(f"Copying preprocessed EEG data of subject {subject_id} to `derivatives/preprocessed`.")
            else: 
                print(f"Preprocessed EEG data file ({subject_eeg_input_path}) could not be found. Skipping.")

        if subject_id in subject_ids_events:

            # Specify subject events paths and create output path if it does not exist already
            subject_events_input_path = subject_input_path.copy().update(root = events_input_path, suffix = "events", extension="tsv", check = True)
            # Copy preprocessed et data (if it exists) from the et-preprocessing derivative to the preprocessed derivative
            if os.path.exists(subject_events_input_path):
                shutil.copy(subject_events_input_path, subject_events_output_path)
                print(f"Copying preprocessed EEG+ET events of subject {subject_id} to `derivatives/preprocessed`.")
            else:
                print(f"Preprocessed EEG+ET events file ({subject_events_input_path}) could not be found. Skipping.")



if __name__ == '__main__':
    main()