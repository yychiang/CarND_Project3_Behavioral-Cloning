# Behavioral Cloning Project

In this project, I implemented a self-driving car which can drive itself on a track of the simulator. The self-driving car used deep neural network as a computational framework to "learn" the driving behavioral of human. That is, I have to "drive" the simulated car on the simulator to collect training data. Then, the self-driving robot learn my behavior. Although Udacity has provided a set of training data, I cannot cause the driving robot stay within yellow lane until I began to collect my own training data. At last, I use my own data completely.

The submitted files are as below:

* model.py - The script used to create and train the model.
* drive.py - The script to drive the car. I make some modifications: resize and normalize images, reduce the throttle, etc. 
* model.json - The model architecture.
* model.h5 - The model weights.
* report.pdf - explains how the training data were collected, the structure of my network, and training approach.
* README.md - lists the files included in this project.



## Acknowledgement

This work is inspired by my mentor (https://medium.com/@deniserjames) and the following papers:
* https://medium.com/@deniserjames/denise-james-bsee-msee-5beb448cf184#.85ep0mupg
* https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.el6uog78o
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

I cannot finish this work without your help.
Thank you all!

