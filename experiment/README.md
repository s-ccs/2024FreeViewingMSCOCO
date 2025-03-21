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

3. Customisation: 
	- The python script `general_setup.py` at the start of the OpenSesame experiment contains several options to configure, e.g. the number of blocks, block size, etc.
	- The images are stored in the assets/images folder, along with individual csv files containing details of the of images that we want to show for the practice and experimental runs. We also have one file containing the list of images for the pilot (450 images instead of the final 400) and one with just 23 images, to use while testing block sizes, breaks etc. 
	- To change the images shown in the practice/experimental trials, update the file details in the respective csv file; then in OpenSesame, 
		- Change the csv file in the corresponding loop item - delete the old csv file from the file pool and import the new csv file into the file pool.
		- If you have changed the experimental images file to a different csv, set the new filename/path in the value `exp_imgdetails_file_name` in the general_setup python file.
		- If you have changed the number of practice trials, update the value of n_practicetrials in the general_setup python file.
	- The screen number & resolution in the OpenSesame experiment settings should match the subject's screen.
	- In general_setup.py, we send settings to the eyetracker like calibration area and type. These settings are not applied before the calibration that automatically happens in new_pygaze_init, therefore we do not use that initial calibration and instead manually trigger it at the end of general_setup.
	- For connection to the EEG recording, we use Lab Streaming Layer with the stream name "experiment_markers" (set up in trigger_setup.py)
	- Templates for the lab-notebook and participant-form files are in the assets folder. These will be copied to the `data/sub-xxx/ses-yyy/beh` folder upon running the exp-startup script. 

## Experiment Flow 🌊

[WILL BE UPDATED WITH A FLOWCHART]

**Creating behavioural files (lab notebook, participant form)** - For each participant, before the experiment, from the 'scripts' folder run the script `exp-startup`. When prompted, enter the subject and session number respectively. The script will create the folder `<project-root>/data/sub-xxx/ses-yyy/beh` and will copy the behavioural file templates (participant form, lab-notebook) to it. (xxx and yyy are the specified subject & session IDs respectively.)

```bash
./exp-startup
```

The structure of the OpenSesame experiment is as follows:

1. Welcome and Instructions
2. Initial Calibration
3. Start trial loop

    a. Press SPACE to start the trial

    b. Fixation cross

    c. Wait for center gaze fixation (if not, recalibrate and repeat the trial)

    d. Jitter (fixation cross still shows up on the screen)

    e. Image presentation
4. End trial loop

> [!NOTE]  
> Researchers: To pause for calibration/other purpose at any point between trials (while the participant sees the “Press Space to start trial” screen), you can press ‘p’.   
> Then ‘c’ to go to calibration menu, or ‘r’ to resume.   
> After that the participant is asked to press space to continue to the image.   
> The OpenSesame console shows the triggers being sent - you can use this to tell when the "Press Space" screen is being displayed.



## Experiment Details

### General Information
1. **Stimuli :** Images from the MSCOCO dataset
2. **Task :** Free-viewing task
3. **Electroencephalogram :** 128-channel EEG cap
4. **Eyetracker :** EyeLink 1000 Plus (Version 5.50), Desktop mount
5. **Screen to eye distance :** 700 mm
6. **Screen resolution :** 1920x1080 px
7. **Exclusion Criteria :** If any of the following are present: any kind of colourblindness, photosensitive epilepsy, photosensitive migraine or any other neurological disorder (major depression, ADHD, autism, or similar). The participant should have corrected to normal vision.

### Experimental Parameters
1. **Trial numbers :** 1-400 (+ practice trials with negative trial numbers)
2. **Calibration breaks :** Every 50 trials (configurable - see 'customisation' section above)
3. **1 Trial duration :** ~5-6 seconds
4. **Size of Images :** 947 x 710 pixels
5. **Calibration Type :** HV13 13 point (sent to the eye tracker via pygaze in OpenSesame, along with the calibration area calculated based on the image & screen size and screen-to-eye distance.)
6. **Number of Practice/Experimental Trials :** Configurable - see 'customisation' section above

### Triggers used :

These are the triggers used in the experiment. The triggers are sent to the eyetracker for future analysis.

> [!NOTE]  
> Some trigger numbers have more than one trigger name/text associated with them. E.g. trigger number 1 corresponds to the fixation cross being shown, however this could happen either in the normal course of the experiment, or again after a recalibration was necessary because the fixation was not detected the first time. For these trigger numbers, the table has a separate row for each instance of trigger text, with the trigger number being the same.

|                                  **Trigger Name/Text**                            | **Trigger Number** |                                 **Opensesame Location**                                 |
|:---------------------------------------------------------------------------------:|:------------------:|:---------------------------------------------------------------------------------------:|
| Python packages version info: (+contains actual version info of packages)         |          0         | general_setup (directly pushing to outlet instead of calling send_trigger because it's simpler) |
| 				Fixation dot shown 				    |          1         | wait_for_centre_gaze (Run) |
| 		   Fixation dot shown again due to recalibration		    |          1         | wait_for_centre_gaze (Prepare) - after recalibration 				|
|                                Stimulus image shown                               |          2         |                               send_trigger_start_stimulus                               |
|		 Recalibration start - wait_for_center_gaze timed out               |          3         |             wait_for_centre_gaze (Prepare) - at calibration step       |
|           		Recalibration start  - send_trigger_breakend                |          3         |             send_trigger_breakend - at the end of the break            |
|              		 Recalibration start  - manual calibrate                    |          3         |             manual_calibrate - when researcher pauses the experiment to calibrate between trials           |
| 	 	Recalibration end  - wait_for_center_gaze timed out        	    |          4         |                 wait_for_centre_gaze (Prepare) - after calibration step                 |
|         	 	Recalibration end  - send_trigger_breakend                  |          4         |                 send_trigger_breakend - at the end of the break                 	 |
|                       Recalibration end  - manual calibrate       	            |          4         |                 manual_calibrate - when researcher pauses the experiment to calibrate between trials                |
|                                    Break start                                    |          5         |                                 send_trigger_breakstart                                 |
|                                     Break end                                     |          6         |                                  send_trigger_breakend                                  |
|                               End of practice trials                              |          7         |                                send_trigger_end_practice                                |
|                                 Stimulus end                             	    |          8         |                                send_trigger_end_stimulus                                |
|                                 Manual pause start                                |          9         |                             send_trigger_manual_pause_start                             |
|                                 Manual pause end                                  |          10        |                             send_trigger_manual_pause_end	                           |



### Runif conditions used
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
