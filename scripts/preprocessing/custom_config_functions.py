from mne.io.base import BaseRaw
from typing import Tuple 
import re

def add_start_and_stop_experiment(
        raw: BaseRaw,
        break_start_regex:str,
        break_end_regex:str
) -> Tuple[str, str]:
    #experiment_events_regex = re.compile(r".*-trigger=0.*")
    experiment_events_regex = r".*-trigger=0.*"
    experiment_annotations = [ann for ann in raw.annotations if re.search(experiment_events_regex, ann["description"])]
    first_exp_event = re.escape(experiment_annotations[0]["description"])
    last_exp_event = re.escape(experiment_annotations[-1]["description"])

    break_start_regex_new = break_start_regex + "|" + last_exp_event
    break_end_regex_new = break_end_regex + "|" + first_exp_event

    return break_start_regex_new, break_end_regex_new
