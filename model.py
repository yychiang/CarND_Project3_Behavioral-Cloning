# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# import the necessary packages ===============================================

import pandas as pd
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, ELU
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint

import matplotlib.image as mpimg
import cv2
import numpy as np
import random
import json
from scipy.misc.pilutil import imresize

# Loading Data ================================================================

csvdata = pd.read_csv('driving_log.csv')
# csvdata specifies the path of the captured images along with the steering, throttle, brake, and speed, etc.
# center: the images captured by the center camera;
# left: the images captured by the left camera;
# right: the images captured by the right camera;


# We just focus on the relationship between input images and steering;
# X_data: the paths of the input images;
# y_data: the steerings.

X_data=np.copy(csvdata['center']+csvdata['left']+csvdata['right'])
y_data=np.copy(csvdata['steering'])



def imgRead(imgpath):
	# the following process is consistent with "drive.py"
	# (1) read image
	# (2) crop image
	# (3) resize image
	# (4) normalize image
    img = mpimg.imread(imgpath, 1)
    cropimg = img[32:135, :]
    resimg = imresize(cropimg, .65, interp='bilinear', mode=None)
    image = -0.5 + resimg/255
    return image

def imgFlip(im):
    # for data augmentation
    flip = np.fliplr(im)
    return flip

# Collect data ================================================================
def collectData(x,y,number=5000):
    # We use this function to collect reasonable training data 
    # from original data set. That is, a steering angle that is big enough
    # will be collected. "Big enough" means its 2-norm is bigger than 0.001
    # X,y: original data
    # number: output size
    images = []
    steering = []
    for xi, yi in zip(x, y):
        probability = random.random()
        if (probability > 0.5 or abs(yi) > 0.001):
            images.append(xi)
            steering.append(yi)

    images=np.copy(images[0:number])
    steering=np.copy(steering[0:number])
    return images, steering

X_train,y_train=collectData(X_data,y_data,20000) # X_train are still the paths of captured images


# Training parameters =========================================================
nb_epoch = 10
nb_rows = 66
nb_columns = 208
nb_channels = 3

#Image Generator ==============================================================
def imgen(X,Y):
    
    counter  = 0 
    while counter < len(X):
        
        if counter==len(X):
            counter  = 0
            X, Y = shuffle(X, Y, random_state=0)
        for counter in range(len(X)): 
            y = Y[counter]
           
            if y <  0.0:
                randnum = random.random()
                if randnum > 0.5:
                    imagepath = X[counter].split(' ')[2] #2
                    image = imgRead(imagepath)
                    y = 3*y
                    if y<-1:
                        y=-1
                else:
                    imagepath = X[counter].split(' ')[0]
                    image = imgRead(imagepath)             
            elif y > 0.0:
                randnum = random.random()
                if randnum > 0.5:
                    imagepath = X[counter].split(' ')[1] #1
                    image = imgRead(imagepath)
                    y = 3*y 
                    if y > 1:
                        y=1 
                else:
                    imagepath = X[counter].split(' ')[0] 
                    image = imgRead(imagepath)                
            else:
                imagepath = X[counter].split(' ')[0]
                image = imgRead(imagepath)
                  
            y = np.array([[y]])
                
            if np.random.choice([True, False]):
                image = imgFlip(image)
                y = -y
            
            image = image.reshape(1, nb_rows, nb_columns, nb_channels)
            yield image, y



# NVIDIA Model ====== =========================================================
def dNNModel(): # NVIDIA Model
    input_shape = (nb_rows, nb_columns, nb_channels)
    model = Sequential()
    model.add(Convolution2D(24,5,5, input_shape=input_shape, subsample = (2,2),border_mode='same', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36,5,5, subsample = (2,2),border_mode='same', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48,5,5, subsample = (2,2),border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, subsample = (1,1),border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, subsample = (1,1),border_mode='valid',init='he_normal'))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100,name='hidden1', init='he_normal'))
    model.add(ELU())
    model.add(Dense(50,name='hidden2', init='he_normal'))
    model.add(ELU())
    model.add(Dense(10,name='hidden3',init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, name='output', init='he_normal'))
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')
    return model


model = dNNModel()



# checkpoint ==================================================================
checkpoint = ModelCheckpoint("model-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]


samples_per_epoch = len(X_train) * 1 
history = model.fit_generator(imgen(X_train, y_train), samples_per_epoch = samples_per_epoch, nb_epoch = nb_epoch,verbose=1, max_q_size = 32, callbacks=callbacks_list, validation_data=None, class_weight=None,pickle_safe=False)


model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
                
print("Trained model has been saved.")















