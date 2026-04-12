# FreeViewingMSCOCO 

This is a free-viewing EEG-EyeTracking experiment. Participants are shown a series of images (selected from the MSCOCO dataset) - for each trial, first a plus symbol is displayed at the center of the screen; then, once the participant has fixated their gaze on this target, an image is shown to them for a fixed duration and the participant is free to explore the image. The experiment is implemented using [OpenSesame](https://osdoc.cogsci.nl/).

This README contains information about the experiment, the implementation in OpenSesame, and instructions on how to replicate it yourself based on the procedure followed for the original data-collection. 


## Experiment Details

### General Information
1. **Stimuli :** Selected images from the MSCOCO dataset
2. **Task :** Free-viewing task
3. **Electroencephalogram :** 128-channel EEG recording (cap: waveguard™ original, ANT Neuro, Berlin, Germany)
4. **Eyetracker :** EyeLink 1000 Plus (Version 5.50), Desktop mount
5. **Screen to eye distance :** 700 mm
6. **Screen resolution :** 1920x1080 px
7. **Exclusion Criteria :** A participant was excluded from the study if any of the following were present: any kind of colourblindness, photosensitive epilepsy, photosensitive migraine or any other neurological disorder (major depression, ADHD, autism, or similar). Participants were also required to have corrected to normal vision, and the eye tracker calibration parameters were required to be within a specific range (for each individual participant, a test calibration was performed to confirm this was possible - if it was not possible, no EEG/ET data was recorded for that participant ID). 

### Experimental Parameters
1. **Trial numbers :** 1-400 (+ practice trials with negative trial numbers)
2. **Calibration breaks :** Every 50 trials (configurable - see 'customisation' section)
3. **1 Trial duration :** ~5-6 seconds
4. **Size of Images :** 947 x 710 pixels
5. **Calibration Type :** HV13 13 point (sent to the eye tracker via pygaze in OpenSesame, along with the calibration area calculated determined according to the EyeLink guidelines based on the image & screen size and the screen-to-eye distance.)
6. **Number of Practice/Experimental Trials :** Configurable - see 'customisation' section of the experiment README.
7. **Triggers/Markers :** Contains a text description of the event along with a trigger number (see table below for details) and trial information (trial number, current stimulus image, etc), separated with a `|` symbol (pipe).


## Replication steps

### Setup

1. Install the required packages via conda. You can create a new environment using the `environment.yaml` file as follows:
```bash
conda env create -f environment.yaml
```

Some packages need to be installed using pip. For this, a `requirements.txt` is also included in this git repository.

> [!NOTE]  
> Here we install OpenSesame via conda. If you want to install it via other download options, please refer to the [official website](https://osdoc.cogsci.nl/3.2/download#all-download-options).

2. Set up the eye tracker and make sure the correct tracker type is selected in the OpenSesame experiment.

3. Customisation - you can customise the OpenSesame experiment, startup scripts, image file paths, etc. according to your lab setup and your preferences. For more details see the section on Customisation.

### Carrying out the experiment

1. For each participant, first run the two startup scripts in the `experiment` folder. When prompted, enter the subject and session number respectively. 
- `exp-startup` - This creates the BIDS-style folder `<project-root>/data/sub-xxx/ses-yyy/beh` and will copy the behavioural file templates (participant form, lab-notebook) to it. Here, xxx and yyy are the specified subject & session IDs respectively. The script will also automatically open these files in the local system text editor (gedit) and start LSLRecorder with the correct filepath and LSL stream names. 
- `startup-opensesame` - This starts the OpenSesame experiment using the given subject ID.

2. Once the experiment starts, it proceeds according to the steps outlined in the "Experiment Flow" section. 

> [!NOTE]  
> Researchers: To pause for calibration/other purpose at any point between trials (while the participant sees the “Press Space to start trial” screen), you can press ‘p’.   
> Then ‘c’ to go to calibration menu, or ‘r’ to resume.   
> After that the participant is asked to press space to continue to the image.   
> The OpenSesame console shows the triggers as they are being sent - you can use this to find out when the "Press Space" screen is being displayed.



## Customisation: 
- The python script `general_setup.py` at the start of the OpenSesame experiment contains several options to configure, e.g. the number of blocks, block size, etc.
- The images are stored in the `assets/images` folder, along with individual csv files containing details of the images that we want to show for the practice and the experimental runs. There is also one file containing the list of images for the pilot (450 images instead of the final 400) and one with just 23 images, to use while testing block sizes, breaks etc. 
- You may need to update the image file paths in the csv files, based on where the images are located in your setup.
- To change the images shown in the practice/experimental trials, update the file details in the respective csv file; then in OpenSesame, 
	- Change the csv file in the corresponding loop item - delete the old csv file from the file pool and import the new csv file into the file pool.
	- If you have changed the experimental images file to a different csv, set the new filename/path in the value `exp_imgdetails_file_name` in the general_setup python file. This ensures that the experiment shows the correct number of trials.
	- If you have changed the number of practice trials, update the value of n_practicetrials in the general_setup python file.
- The screen number & resolution in the OpenSesame experiment settings should match the subject's screen.
- Note regarding initial calibration in OpenSesame: In general_setup.py, we send settings to the eyetracker, like the calibration area and type. These settings are not applied before the calibration that automatically happens in OpenSesame's `new_pygaze_init`, therefore we do not use that initial calibration and instead manually trigger it at the end of general_setup.
- For connection to the EEG recording, we use Lab Streaming Layer with the stream name "experiment_markers" (set up in trigger_setup.py)
- Templates for the lab-notebook and participant-form files are stored in the assets folder (you can customise these templates to your liking). These will be copied to the `data/sub-xxx/ses-yyy/beh` folder upon running the exp-startup script for each participant. 
- `exp-startup` and `startup-opensesame` scripts: customise the file/folder paths for the data/participant files and the opensesame experiments respectively according to your setup.



## Experiment Implementation Details

### Experiment Flow

The structure of the OpenSesame experiment is as follows:

1. Welcome and Instructions
2. Initial Calibration
3. Start trial loop

    a. Press SPACE to start the trial (here the researcher can pause the experiment if required, before the participant presses SPACE)

    b. Fixation cross

    c. Wait for center gaze fixation (if not, recalibrate and repeat the trial)

    d. Jitter (fixation cross still shows up on the screen)

    e. Image presentation
4. End trial loop


### Triggers used :

These are the triggers used in the experiment. The triggers are sent to the eyetracker for future analysis.

> [!NOTE]  
> Some trigger numbers have more than one trigger name/text associated with them. E.g. trigger number 1 corresponds to the fixation cross being shown, however this could happen either in the normal course of the experiment, or again after a recalibration was necessary because the fixation was not detected the first time. For these trigger numbers, the table has a separate row for each instance of trigger text, with the trigger number being the same.

|                                  **Trigger Name/Text**                            | **Trigger Number** |                                 **Opensesame Location**                                 |
|:---------------------------------------------------------------------------------:|:------------------:|:---------------------------------------------------------------------------------------:|
| Python packages version info: (+contains actual version info of packages)         |          00         | general_setup (directly pushing to outlet instead of calling send_trigger because it's simpler) |
| 				Fixation dot shown 				    |          01         | wait_for_centre_gaze (Run) |
| 		   Fixation dot shown again due to recalibration		    |          01         | wait_for_centre_gaze (Prepare) - after recalibration 				|
|                                Stimulus image shown                               |          02         |                               send_trigger_start_stimulus                               |
|		 Recalibration start - wait_for_center_gaze timed out               |          03         |             wait_for_centre_gaze (Prepare) - at calibration step       |
|           		Recalibration start  - send_trigger_breakend                |          03         |             send_trigger_breakend - at the end of the break            |
|              		 Recalibration start  - manual calibrate                    |          03         |             manual_calibrate - when researcher pauses the experiment to calibrate between trials           |
| 	 	Recalibration end  - wait_for_center_gaze timed out        	    |          04         |                 wait_for_centre_gaze (Prepare) - after calibration step                 |
|         	 	Recalibration end  - send_trigger_breakend                  |          04         |                 send_trigger_breakend - at the end of the break                 	 |
|                       Recalibration end  - manual calibrate       	            |          04         |                 manual_calibrate - when researcher pauses the experiment to calibrate between trials                |
|                                    Break start                                    |          05         |                                 send_trigger_breakstart                                 |
|                                     Break end                                     |          06         |                                  send_trigger_breakend                                  |
|                               End of practice trials                              |          07         |                                send_trigger_end_practice                                |
|                                 Stimulus end                             	    |          08         |                                send_trigger_end_stimulus                                |
|                                 Manual pause start                                |          09         |                             send_trigger_manual_pause_start                             |
|                                 Manual pause end                                  |          10        |                             send_trigger_manual_pause_end	                           |



### Controlling the experiment flow - `runif` conditions used
`Run if` refers to the field in OpenSesame in the `sequence` item of the trial loop. It is used to execute certain items (e.g. breaks or sending triggers) only when the condition is met.

#### Main block sequence

| OpenSesame Object |                                                        Runif                                                       |   |
|:-----------------:|:------------------------------------------------------------------------------------------------------------------:|---|
|   break_sequence  | =(count_block_sequence&gt;0) and (count_block_sequence!=total_trials) and ((count_block_sequence%block_size) == 0) |   |

#### Fixation block sequence

|      **OpenSesame Object**      |                                  **Runif**                                 |
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
