import pandas as pd
import os
import re
import json

data_path = "/scratch/data/2024FreeViewingMSCOCO"

# Create a list of all participants in the data set
dir_list = os.listdir(data_path)
subject_dir_list = list(filter(lambda x: re.match(r"sub-\d{3}", x), dir_list))
subject_ids = list(map(lambda x: x.split("-")[1], subject_dir_list))

# Load the demographics data from all participant and aggregate them in one data frame
demographic_data = pd.DataFrame(columns=["id", "age", "gender", "handedness", "dom_eye", "no_preex_conditions", "visual_acuity_test", "remarks"])
for subject_id in subject_ids:
    participant_form = pd.read_csv(os.path.join(data_path, f"sub-{subject_id}/ses-001/misc/sub-{subject_id}_ses-001_participantform.tsv"), sep="\t")
    demographic_data = pd.concat([demographic_data, participant_form])

# Adapt data frame to match BIDS conventions
demographic_data.sort_values(by="id", inplace=True, ignore_index=True)
demographic_data.rename(columns={"id":"participant_id"}, inplace=True)
demographic_data.drop(columns=["no_preex_conditions","visual_acuity_test","remarks"], inplace=True)
demographic_data["participant_id"] = list(map(lambda i: f"sub-{i:03}", demographic_data.participant_id))

# Overwrite existing "dummy" participants file
demographic_data.to_csv(os.path.join(data_path, "participants.tsv"), sep="\t", index=False)

# Create demographics summary and save it as a JSON file
age_summary = demographic_data.age.agg(['mean', 'median', 'std', 'min', 'max'])
gender_summary = demographic_data["gender"].value_counts()
handedness_summary = demographic_data["handedness"].value_counts()
dom_eye_summary = demographic_data["dom_eye"].value_counts()

demographics_summary = {
    "n_participants": int(len(demographic_data)),
    "age":{"mean": float(age_summary["mean"]), "std": float(age_summary["std"]), "range": list([int(age_summary["min"]), int(age_summary["max"])])},
    "gender":{"female": str(gender_summary["f"]), "male": str(gender_summary["m"])},
    "handedness":{"right": str(handedness_summary["r"]), "left": str(handedness_summary["l"]), "ambidextrous": str(handedness_summary["a"])},
    "dominant_eye":{"right": str(dom_eye_summary["r"]), "left": str(dom_eye_summary["l"])}
}

with open(os.path.join(data_path, "demographics_summary.json"), "w") as f:
    json.dump(demographics_summary, f, indent=4)