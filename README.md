# Behavioral Cloning Project

In this project, I implemented a self-driving car which can drive itself on a track of the simulator. The self-driving car used deep neural network as a computational framework to "leran" the driving behavioral of human. That is, I have to "drive" the simulated car on the simulator to collect traing data. Then, the self-driving robot learn my behavior. Although Udacity has provided a set of training data, I cannot cause the driving robot stay within yellow lane until I began to collect my own training data. At last, I use my own data completely.

The submitted files are as below:

* model.py - The script used to create and train the model.
* drive.py - The script to drive the car. I make some modifications: resize and normalize images, reduce the throttle, etc. 
* model.json - The model architecture.
* model.h5 - The model weights.
* README.md - explains how the training data were collected, the structure of my network, and training approach.

## How the training data were collected?
I collected my own data and used the collected data completely for training my deep neural network.
Training data is collected as the following method:
* (1) Firset round, let the car runs on the center of the road, and set steerings as 0;
* (2) Second round, let the car runs on the right hand side of the road, and set steerings as -0.5;
* (3) Third round, let the car runs on the left hand side of the road, and set steerings as +0.5.
* Note: Please see my driving_log.csv file.

## Network Architecture
* input shape is: 66x208x3 (HxWxD)
* 1st, conv layer: 24 filters with shape 5x5x3 (HxWxD), stride 2, same padding;
* 2nd, conv layer: 36 filters with shape 5x5x24 (HxWxD), stride 2, same padding;
* 3rd, conv layer: 48 filters with shape 5x5x36 (HxWxD), stride 2, valid padding;
* 4th, conv layer: 64 filters with shape 3x3x48 (HxWxD), stride 1, valid padding;
* 5th, conv layer: 64 filters with shape 3x3x64 (HxWxD), stride 1, valid padding;
* 6th, flatten: 


## Training approach
