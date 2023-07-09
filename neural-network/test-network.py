# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:00:58 2023

@author: Giulia
"""
import numpy as np
import cv2
import CaptchaUtils
from keras.models import load_model

def caricaImmagine():
    seg_path = "../dataset/singolaImmagine/" #trova path con immagine da identificare

    data = []
    images,labels=CaptchaUtils.segmenta(seg_path)
    for image in images:
            resized = cv2.resize(image, (30, 30))
            data.append(resized)
    return data, labels

if __name__ == "__main__":
    print("Caricamento modello")
    model = load_model('../modelli/modello.hdf5')

    print("Caricamento immagine")
    data,_= caricaImmagine()
    output=''
    for i in range(len(data)):
        sample = data[i]
        sample = sample.astype("float32") / 255.0 #normalizza dividendo per 255 (astype trasforma in double)
        sample = np.expand_dims(sample, axis=0) #aggiunge dimensione all'array (modello prende 4 valori in input)

        out = model.predict(sample)
        decoded = CaptchaUtils.decode(out)
        output+=decoded
    print("Captcha rilevato: "+output)