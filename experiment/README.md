# FreeViewingMSCOCO 

This is a free-viewing EEG-ET experiment written using [OpenSesame](https://osdoc.cogsci.nl/).

## Installation 👩‍💻

1. Install the required packages via conda. You can create a new environment using the `environment.yaml` file as follows:
```bash
conda env create -f environment.yaml
```

> [!NOTE]  
> Here we install OpenSesame via conda. If you want to install it via other download options, please refer to the [official website](https://osdoc.cogsci.nl/3.2/download#all-download-options).

2. EyeLink 1000 Plus Setup
    - Connect the EyeLink 1000 Plus to the host PC and follow the instructions in the manual to set up the device. [More on this later].
    - Make sure that the EyeLink 1000 Plus is connected to the host PC via the Ethernet cable/wifi.

## Experiment Flow 🌊

[WILL BE UPDATED WITH A FLOWCHART]

The structure of the experiment is as follows:

1. Welcome and Instructions
2. Initial Calibration
3. Start trial loop

    a. Press SPACE to start the trial

    b. Fixation cross

    c. Wait for center gaze fixation (if not, recalibrate and repeat the trial)

    d. Jitter (fixation cross still shows up on the screen)

    e. Image presentation
4. End trial loop


## Eye tracker setup 

1. **Eye tracker used :** EyeLink 1000 Plus (Version 5.50)
2. **Mount Type** : Desktop mount


## Experiment Details

### General Information
1. **Stimuli :** Images from the MSCOCO dataset
2. **Task :** Free-viewing task
3. **Electroencephalogram :** 32-channel EEG cap
4. **Eyetracker :** EyeLink 1000 Plus
5. **Screen to eye distance :** 700 mm
6. **Screen resolution :** 1920x1080 px

### Experimental Parameters
1. **No of trials :** 400
2. **Calibration breaks** : Every 50 trials
3. **1 Trial duration :** ~5-6 seconds
4. **Size of Images :** 947 x 710 pixels
5. **Calibration Type :** HV13 13 point


### Triggers used :

These are the triggers used in the experiment. The triggers are sent to the eyetracker for future analysis.

|                                  **Trigger Name**                                 | **Trigger Number** |                                 **Opensesame Location**                                 |
|:---------------------------------------------------------------------------------:|:------------------:|:---------------------------------------------------------------------------------------:|
| Fixation dot shown - Run & Fixation dot shown again due to recalibration - Prepare |          1         | wait_for_centre_gaze (Run) & wait_for_centre_gaze (Prepare) - after calibration step |
|                                Stimulus image shown                               |          2         |                               send_trigger_start_stimulus                               |
|                                Recalibration start                                |          3         |             wait_for_centre_gaze (Prepare) - at calibration step & breaks             |
|                                 Recalibration end                                 |          4         |                 wait_for_centre_gaze (Prepare) - after calibration step                 |
|                                    Break start                                    |          5         |                                 send_trigger-breakstart                                 |
|                                     Break end                                     |          6         |                                  send_trigger-breakend                                  |
|                                  End of practice                                  |          7         |                                End_of_practice sketchpad                                |
|                                 Stimulus event end                                |          8         |                                send_trigger_end_stimulus                                |
|                                 Manual pause start                                |          9         |                             send_trigger_manual_pause_start                             |



### Runif conditions used
Run if refers to the field in opensesame in the `sequence` item of the trial loop. It is used to send triggers only when the condition is met.

#### Main block sequence

| Opensesame Object |                                                        Runif                                                       |   |
|:-----------------:|:------------------------------------------------------------------------------------------------------------------:|---|
|   break_sequence  | =(count_block_sequence&gt;0) and (count_block_sequence!=total_trials) and ((count_block_sequence%block_size) == 0) |   |

#### fixation block sequence

|      **Opensesame Object**      |                                  **Runif**                                 |
|:-------------------------------:|:--------------------------------------------------------------------------:|
|        manual_pause_start       |                     [response_keyboard_response] = ‘p’                     |
| send_trigger_manual_pause_start |                     [response_keyboard_response] = ‘p’                     |
|         manual_pause_kbd        |                     [response_keyboard_response] = ‘p’                     |
|         manual_calibrate        | [response_keyboard_response] = ‘p’ and [response_keyboard_response] = ‘c’] |
|         manual_pause_end        |                     [response_keyboard_response] = ‘p’                     |
|       manual-pause_end_kbd      |                     [response_keyboard_response] = ‘p’                     |
|  send_trigger_manual_pause_end  |                     [response_keyboard_response] = ‘p’                     |

> [!CAUTION]
> Never name any variable in your inline script as `timeout` 🥲. It might break the functionality of your experiment. Follow this [discussion](https://forum.cogsci.nl/discussion/6393/sketchpad-does-not-wait-for-the-keypress) for more details!