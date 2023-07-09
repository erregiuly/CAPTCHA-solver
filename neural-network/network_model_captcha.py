# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:00:58 2023

@author: Giulia
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K

class NetworkModelCaptcha:
    @staticmethod
    def build(width, height, depth, out_size):

        model = Sequential()
        inputShape = (height, width, depth)

        #Se immagine ha questo data format height & depth sono invertiti
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #applico trasformazioni per provare a far funzionare il modello
        model.add(Conv2D(32, (3, 3), padding='valid', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.25))

        # Fully connected layer
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(out_size))
        model.add(Activation('softmax'))
        
        return model