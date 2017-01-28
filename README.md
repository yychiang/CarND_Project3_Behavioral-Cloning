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

## Training approach
