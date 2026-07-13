import os.path
import argparse
import mne
from mne_bids import BIDSPath
import pandas as pd

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
        "--interpolation_plot",
        action="store_true",
        help="Enable interpolation plot.",
    )
    args = parser.parse_args()

    # Specify file paths
    data_root_path = "/scratch/data/2024FreeViewingMSCOCO/"
    #input_path = os.path.join(data_root_path, "derivatives/mne-bids-pipeline")
    input_path = os.path.join(data_root_path, "derivatives/preprocessing-with-manual-preprocessing")
    output_path = os.path.join(data_root_path, "derivatives/custom-preprocessing")

    # Check whether output folder exists, otherwise create it
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        print(f"Create output folder: {output_path}")

    # Specify session, run and task
    session = 1
    padded_session = f"{session:03}"
    task = "freeviewing"
    run = 1

    # Specify if EEG and ET events should be combined in one events.tsv
    combine_eeg_et = True

    # Variables to save whether it has already been tested whether the events for EEG and ET exist/have been created
    checked_eeg_events = False
    checked_et_events = False

    # Read subject_ids from command line or extract them from participants.tsv
    if args.subject_ids:
        subject_ids = args.subject_ids
        print(f"Using provided subject_ids: {subject_ids}")
    else:
        subject_ids = extract_subject_ids(data_root_path)
        print("Using subject_ids from participants.tsv file.")

    
    for subject_id in subject_ids:
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

        # Specify subject eeg paths and create output path if it does not exist already
        subject_eeg_input_path = subject_input_path.copy().update(root= input_path, datatype = "eeg", processing = "clean", suffix = "raw", extension = ".fif", check = False) # check = False because "raw" is not an allowed suffix
        subject_eeg_output_path = subject_output_path.copy().update(datatype="eeg")
        subject_eeg_output_path.mkdir()

        if os.path.exists(subject_eeg_input_path):
            
            # Read EEG file
            raw_eeg = mne.io.read_raw_fif(subject_eeg_input_path, preload=True)

            # Create and save events df
            events_df = create_events_dataframe(raw_eeg)

            if combine_eeg_et:
                # Save EEG events (before combining with ET)
                events_path = subject_eeg_output_path.copy().update(suffix="events_without_et", extension = "tsv", check = False)
                events_df.to_csv(events_path, sep="\t", index=False)
                
                # Load ET events and combine with EEG events
                subject_et_input_path = subject_input_path.copy().update(root = input_path, run = None, datatype ="misc", suffix = "et_events", extension="tsv", check = False) # check = False because "misc" is no valid `datatype`

                et_events = pd.read_csv(subject_et_input_path, sep = "\t")
                events_combined = pd.concat([events_df, et_events], join="outer", ignore_index=True)
                assert len(events_combined) == (len(events_df) + len(et_events))
                events_combined.sort_values(by="onset", inplace=True)
                events_combined.to_csv(events_path.copy().update(suffix="events", check=True), sep="\t", index=False)
                print(f"Combined EEG and ET events and save as events.tsv.")
            else:
                # Save EEG events
                events_path = subject_eeg_output_path.copy().update(suffix="events", extension = "tsv", check = True)
                events_df.to_csv(events_path, sep="\t", index=False)
                print(f"Created and saved EEG events.tsv file for subject {subject_id}.")

            # Interpolate bad channels
            raw_eeg.interpolate_bads(reset_bads=False)

            # Save EEG data with interpolated channels
            raw_eeg.save(subject_eeg_output_path.copy().update(processing = "clean", suffix = "raw", extension = ".fif", check = False), overwrite=True)
            print("Saved EEG data with interpolated bad channels.")

            # Make a plot with interpolated channels marked in red
            if args.interpolation_plot:
                print("Creating plot to verify the channel interpolation.")
                raw_eeg_vis = raw_eeg.copy().set_annotations(None)
                interpolation_plot = raw_eeg_vis.plot(n_channels=128, show_scrollbars=False, show_scalebars=False, bad_color="red")
                interpolation_plot.savefig(subject_eeg_output_path.update(suffix="interpolation_plot", extension = "pdf", check = False))

        else:
            print(f"Preprocessed EEG data file ({subject_eeg_input_path}) could not be found. Skipping.")

if __name__ == '__main__':
    main()