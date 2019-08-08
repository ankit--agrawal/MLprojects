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
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
for i in range(50):
    train_dataset = train_dataset.sample(frac=1)

#train_dataset = train_dataset.iloc[::2,:] #selecting every 2nd frame and it's label

# Importing the Keras libraries and packages
from keras import regularizers
from keras.models import Sequential, model_from_json, Model
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop, Adagrad, Adadelta, Adam, SGD
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

def create_model():
    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Convolution2D(64, (3, 3),strides=(2,2),input_shape = (480, 480, 3), activation = 'relu', padding='same'))
    classifier.add(BatchNormalization())
    
    # Adding a second convolutional layer
    classifier.add(Convolution2D(32, (3, 3), activation = 'relu',strides=(2,2)))
    classifier.add(BatchNormalization())
    
    # Adding a second convolutional layer
    classifier.add(Convolution2D(32, (3, 3), activation = 'relu',strides=(2,2)))
    classifier.add(BatchNormalization())

    classifier.add(Convolution2D(16, (3, 3), activation = 'relu',strides=(2,2)))
    classifier.add(BatchNormalization())

    # Step 4 - Full connection
    classifier.add(Flatten())
    #Adding full connection layer increases the parameter size.
    classifier.add(Dense(700, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(150, activation='relu'))
    #classifier.add(Dropout(0.5))
    classifier.add(Dense(1, activation='linear'))
    #Adding full connection layer increases the parameter size.
    
    return classifier

# Part 2 - Fitting the CNN to the images

from keras_preprocessing.image import ImageDataGenerator
from PIL import Image

train_datagen = ImageDataGenerator(rescale = 1./255., featurewise_std_normalization=True, validation_split = 0.20)
test_datagen = ImageDataGenerator(rescale = 1./255.)

training_set = train_datagen.flow_from_dataframe(dataframe = train_dataset,
                                                 directory = './data/training/',
                                                 x_col = 'image_names',
                                                 y_col = 'labels',
                                                 subset = "training",
                                                 class_mode = "other",
                                                 target_size = (480,480),
                                                 batch_size = 30, shuffle = True)

validation_set = train_datagen.flow_from_dataframe(dataframe = train_dataset,
                                                 directory = './data/training/',
                                                 x_col = 'image_names',
                                                 y_col = 'labels',
                                                 subset = "validation",
                                                 class_mode = "other",
                                                 target_size = (480,480),
                                                 batch_size = 30, shuffle = True)


test_set = test_datagen.flow_from_dataframe(dataframe = test_dataset,
                                           directory = './data/test/',
                                           x_col = 'image_names',
                                           target_size = (480, 480),
                                           batch_size = 30,
                                           class_mode = None, shuffle = False)

classifier = create_model()
try:
    print('---loading saved weights---')
    classifier.load_weights('speed_car.h5')
except ValueError:
    print('---new model detected---')

early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
model_checkpoint = ModelCheckpoint('speed_car.h5',verbose=1, save_best_only= True, save_weights_only=True)

rms = SGD(lr=0.00008)
classifier.compile(optimizer = rms, loss = 'mse')
print(classifier.summary())


STEP_SIZE_TRAIN=training_set.n//training_set.batch_size 
STEP_SIZE_VAL=validation_set.n//validation_set.batch_size
STEP_SIZE_TEST=test_set.n//test_set.batch_size

classifier.fit_generator(generator = training_set,
                         steps_per_epoch = STEP_SIZE_TRAIN,
                         epochs =10, callbacks = [early_stop, model_checkpoint],
                         validation_data = validation_set,
                         validation_steps = STEP_SIZE_VAL)

classifier.evaluate_generator(generator=validation_set, steps=STEP_SIZE_VAL)

test_set.reset()
loaded_model = create_model()
loaded_model.load_weights('speed_car.h5')

pred=loaded_model.predict_generator(test_set,verbose=1, steps=STEP_SIZE_TEST)
print(len(pred))
#predicted_class=np.argmax(pred,ax)

pred = pd.DataFrame(pred, columns=['predictions'])
#print(pred)
#writing to a file
pred.to_csv('output.csv', index=False)
