# Behavioral Cloning Project

In this project, I implemented a self-driving car which can drive itself on a track of the simulator. The self-driving car used deep neural network as a computational framework to "learn" the driving behavioral of human. That is, I have to "drive" the simulated car on the simulator to collect training data. Then, the self-driving robot learn my behavior. Although Udacity has provided a set of training data, I cannot cause the driving robot stay within yellow lane until I began to collect my own training data. At last, I use my own data completely.

The submitted files are as below:

* model.py - The script used to create and train the model.
* drive.py - The script to drive the car. I make some modifications: resize and normalize images, reduce the throttle, etc. 
* model.json - The model architecture.
* model.h5 - The model weights.
* README.md - explains how the training data were collected, the structure of my network, and training approach.

## How the training data were collected?
I collected my own data and used the collected data completely for training my deep neural network.
Training data is collected as the following method:
* (1) First round, let the car runs on the center of the road, and set steering angles as 0;
* (2) Second round, let the car runs on the right hand side of the road, and set steering angles as -0.5;
* (3) Third round, let the car runs on the left hand side of the road, and set steering angles as +0.5.
* Note: Please see my driving_log.csv file.

## Network Architecture
* input planes: 3@66x208
* 1st, conv layer: 5x5 kernel, normalized input planes 3@66x208, stride 2, same padding;
* 2nd, conv layer: 5x5 kernel, convolutional feature map 24@33x104, stride 2, same padding;
* 3rd, conv layer: 5x5 kernel, convolutional feature map 36@17x52, stride 2, valid padding;
* 4th, conv layer: 3x3 kernel, convolutional feature map 48@7x24, stride 1, valid padding;
* 5th, conv layer: 3x3 kernel, convolutional feature map 64@5x22, stride 1, valid padding;
* 6th, flatten: the output of 5th conv layer is (3,20,64), which is flattened to 3840 neurons.
* 7th, fully-connected layer, 100 neurons;
* 8th, fully-connected layer, 50 neurons;
* 9th, fully-connected layer, 10 neurons;
* 10th, fully-connected layer, 1 neuron.

## Training approach
* 'Adam' optimization is adopted.

## Dropout layers:
I have tried to add dropout layers after conv layers, but I feel this problem cannot be improved via dropout layers.

## Acknowledgement

This work is inspired by my mentor (https://medium.com/@deniserjames) and the following papers:
* https://medium.com/@deniserjames/denise-james-bsee-msee-5beb448cf184#.85ep0mupg
* https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.el6uog78o
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

I cannot finish this work without your help.
Thank you all!

