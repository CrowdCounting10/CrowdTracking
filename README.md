# CrowdTracking
 Crowd Counting and Crowd Tracking

# CrowdCounting on VisDrone2020
In this repository it has been implemented an artificial intelligence model that allows to obtain the density heat-map corresponding to real scenes shot by drones, exploiting the [MobileCount](https://github.com/SelinaFelton/MobileCount) model and then extract from this a prediction of the number of people in the scene and their direction during the movement in a video sequence.

## Running the code

### Requirements

Python 3.8

Install requirement.txt

In order to train model use:
- main.py

In order to test model and obtain time performances use: 
 - test_gpu.py

For train and test all the parameters given in config.py and the the chosen dataset py file, will be used.

In order to run model use:
- run.py

For run mode, you must also specify:
<ul>
<li>--path: path to the video or image file, or the folder containing the image</li>
<li>--callbacks: list of callback function to be executed after the forward of each element</li>
</ul>


