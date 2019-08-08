#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:58:49 2019

@author: ankit
"""
import pandas as pd
import numpy as np    # for mathematical operations
import glob2
from collections import OrderedDict
import re, csv

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


'''#Extracting frames from video as images
vidcap = cv2.VideoCapture('test.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1
print(count)
'''

#pre-processing the data
images_train = np.array(natural_sort(glob2.glob('data/training/*.jpg')))
images_test = np.array(natural_sort(glob2.glob('data/test/*.jpg')))
y_train = pd.read_csv('train.txt', delimiter='\n', names = ['labels'],dtype=float)

train_dataset = pd.DataFrame(OrderedDict({'image_names':pd.Series(images_train).str.slice(14)}))
test_dataset = pd.DataFrame(OrderedDict({'image_names':pd.Series(images_test).str.slice(10)}))
train_dataset['labels'] = y_train['labels']

print(len(train_dataset),len(test_dataset))
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Initialising the CNN
classifier = Sequential()

chanDim = -1

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (480, 640, 3), activation = 'relu', padding = 'same'))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a second convolutional layer
classifier.add(Convolution2D(64, (4, 4), activation = 'relu', padding = 'same'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu', padding = 'same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu', padding = 'same'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#classifier.add(Dense(units = 900, activation = 'linear'))
classifier.add(Dense(units = 100, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 50, activation = 'linear'))
classifier.add(Dense(units = 1))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Part 2 - Fitting the CNN to the images

from keras_preprocessing.image import ImageDataGenerator
from PIL import Image

train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.15)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_labels = np.array(train_dataset["labels"])
training_set = train_datagen.flow_from_dataframe(dataframe = train_dataset,
                                                 directory = './data/training/',
                                                 x_col = "image_names",
                                                 y_col = 'labels',
                                                 subset = "training",
                                                 class_mode = "other",
                                                 target_size = (480,640),
                                                 batch_size = 32, shuffle = True)

validation_set = train_datagen.flow_from_dataframe(dataframe = train_dataset,
                                                 directory = './data/training/',
                                                 x_col = "image_names",
                                                 y_col = 'labels',
                                                 subset = "validation",
                                                 class_mode = "other",
                                                 target_size = (480,640),
                                                 batch_size = 32, shuffle = True)

early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
model_checkpoint = ModelCheckpoint('speed_car.h5',verbose=1, save_best_only= True)


classifier.fit_generator(generator = training_set,
                         steps_per_epoch = training_set.n//training_set.batch_size,
                         epochs =30, callbacks = [early_stop, model_checkpoint],
                         validation_data = validation_set,
                         validation_steps = validation_set.n//validation_set.batch_size)

classifier.evaluate_generator(generator=validation_set)


test_set = test_datagen.flow_from_dataframe(dataframe = test_dataset,
                                           directory = './data/test/',
                                           x_col = 'image_names',
                                           target_size = (480, 640),
                                           batch_size = 1,
                                           class_mode = None, shuffle = False)

test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1)
#predicted_class_indices=np.argmax(pred,axis=1)

print(pred)
#writing to a file

