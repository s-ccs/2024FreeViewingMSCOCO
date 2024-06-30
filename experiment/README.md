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

1. Welcome and Preliminary Instructions
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

|    **Trigger Name**   | **Trigger Number** |                               **Opensesame Location**                              |            **Runif**            |
|:---------------------:|:------------------:|:----------------------------------------------------------------------------------:|:-------------------------------:|
|  Fixation Point Shown |          1         | wait_for_centre_gaze (Run) wait_for_centre_gaze (Prepare) - after calibration step |                                 |
|  Stimulus Image Shown |          2         |                             send_trigger_start_stimulus                            |                                 |
|   Calibration Start   |          3         |                wait_for_centre_gaze (Prepare) - at calibration step                |                                 |
|    Calibration End    |          4         |               wait_for_centre_gaze (Prepare) - after calibration step              |                                 |
| Mandatory_Break Start |          5         |                               send_trigger-breakstart                              | =(count_block_sequence+1)%50==0 |
|  Mandatory_Break End  |          6         |                                send_trigger-breakend                               | =(count_block_sequence+1)%50==0 |
|    End of Practice    |          7         |                              End_of_practice sketchpad                             |                                 |
|   Stimulus event end  |          8         |                              send_trigger_end_stimulus                             |                                 |
|   Manual Pause Start  |          9         |                          `send_trigger_manual_pause_start`                         |  c for calibrate; r for resume  |
|    Manual Pause End   |         10         |                           `send_trigger_manual_pasue_end`                          |                                 |


**Run if refers to the field in opensesame in the `sequence` item of the trial loop. It is used to send triggers only when the condition is met.*

> [!CAUTION] Never name any variable in your inline script as `timeout` 🥲. It might break the functionality of your experiment. Follow this [discussion](https://forum.cogsci.nl/discussion/6393/sketchpad-does-not-wait-for-the-keypress) for more details!