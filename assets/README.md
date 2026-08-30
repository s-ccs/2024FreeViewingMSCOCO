# Assets for the 2024FreeViewingMSCOCO experiment

## `images` directory 

This directory stores images and CSV files needed in order to run the OpenSesame experiment:
- All the image files selected out of the MSCOCO dataset. 
- The CSV files contain file paths (and other details) of the images used for the experiment. OpenSesame loads images based on the CSV file selected for this purpose.  
    - `experiment_images_info.csv`, `practise_images_info.csv`: details of the images of the main experiment and the pre-experiment practise trials respectively. 
    - `experiment_images_info_test23imgs.csv`: Used while developing and testing the experiment. 
        23 images is a small enough number to allow creating multiple blocks of e.g. 5 trials each, and is not large enough that testing the experiment from start to finish takes unreasonably long. 23 being a prime number, one can also be sure of having uneven blocks (also a test case that was checked when developing the experiment), irrespective of block size.
    - `experiment_images_info_pilot.csv`, `practise_images_info_five_images.csv`: files used in some early pilot runs.

## Additional files:

- EEG cap size chart: for the lab personnel to choose the appropriate size of EEG cap for the participant.
- Mapping file: location for mapping the participant names to the subject-ID assigned to them on the day of recording. 
- template-labnotebook and template-participantform: copied to each individual subject folder via the `exp-startup` script.