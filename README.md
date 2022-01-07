# Crowd Flow Detection from Drones with Fully Convolutional Networks and Clustering
Crowd analysis from drones has attracted increasing attention in recent times, thanks to the ease of deployment and affordable cost of these devices. In this repository, we contribute by sharing a crowd flow detection method for video sequences shot by a drone. The method is mainly based on a fully convolutional network model for crowd density estimation, namely [MobileCount](https://github.com/SelinaFelton/MobileCount), which aims to provide a good compromise between effectiveness and efficiency, and clustering algorithms aimed at detecting the centroids of high-density areas in density maps. The method was tested on the [VisDrone Crowd Counting](http://aiskyeye.com/challenge/crowd-counting/) dataset (characterized not by still images but by video sequences) providing promising results. This direction may open up new ways of analyzing high-level crowd behavior from drones.

## Running the code

Main requirement: Python 3.8

Install requirements.txt

To train the model, use:
- main.py

To test the model and get the GPU performance time, use: 
 - test_gpu.py

All parameters provided in config.py and the chosen dataset py file will be used for training and testing.

To run the model, use:
- run.py

For the run mode, you must also specify:
<ul>
<li>--path: path to the video or image file, or the folder containing them</li>
<li>--callbacks: list of callback function to be executed after the forward of each element</li>
</ul>
