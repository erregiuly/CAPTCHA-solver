# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:00:58 2023

@author: Giulia
"""

import cv2 #libreria per gestione di immagini
import os


from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from network_model_captcha import NetworkModelCaptcha
import CaptchaUtils

# carico il dataset
def load_dataset(): 
    seg_path = "../dataset/divisione/" #cartella con esempi di ogni singola lettera

    #setta come array vuoti
    data = []
    labels = []
    
    #Per ognuno dei possibili caratteri, cerco le immagini nella relativa cartella
    for char in CaptchaUtils.caratteri:
        print(f"Caricamento immagini relative a '{char}'")

        #trova il nome della cartella relativa al carattere
        path = seg_path + char + "/"
        if not os.path.isdir(path): # Se la cartella di una specifica lettera non c'è,skippo
           continue
        files = os.listdir(path) #files=array con stringhe che rappresentano cartelle e nomi file

        for file in files: #scorre cartelle per ogni carattere
            image = cv2.imread(path + file)
            resized = cv2.resize(image, (30, 30))
            label = CaptchaUtils.encode(char)   #utilizzo encode per passare un numero che rappresenta char al modello
            data.append(resized)    #a data aggiunge singoli esempi di ogni lettera
            labels.append(label)    #label contiene lettera associata tramite valore a 1
    return data, labels



if __name__ == "__main__": 
    
    epoche = 1024
    learning_rate = 1e-3
    batch_size = 128
    validation_split=0.66
    nomeModello='modello'


    print("Caricamento dataset...")
    data, labels = load_dataset()
    
    # Normalizzazione e separazione dati in train e validation
    n_data, n_labels = CaptchaUtils.normalize_samples(data, labels)
    (train_x, validation_x, train_y, validation_y) = train_test_split(n_data, n_labels, test_size=0.3, random_state=42)
    #train_test_split divide immagini in test e validation secondo ratio dato (metodo di keras) -> restituisce 4 variabili: allenamento di x e y, validation di x e y
    #x=immagini y=lettere; si allena su train, testa su validation

    print("Inizio allenamento!")
    
    #determina path dove salvare i modelli
    mod_path = "../modelli/"

    #cambia path corrente in path dei modelli
    if not os.path.isdir(mod_path):
        os.makedirs(mod_path)
    sgd = SGD(learning_rate=learning_rate)  #setta learning_rate a valore assegnato alla variabile

    # Inizializza il modello (usa metodo build della classe del modello base)
    model = NetworkModelCaptcha.build(30, 30, 3, len(CaptchaUtils.caratteri))   #len è lunghezza max output
    model.compile(loss='categorical_crossentropy', 
                optimizer=sgd, 
                metrics=['accuracy'])

    #Inizio training
    model.fit(train_x, train_y, 
                validation_data=(validation_x, validation_y), 
                batch_size=batch_size, 
                epochs=epoche, 
                verbose=1)

    model.save(mod_path+ nomeModello+'.hdf5')

    print("Fine allenamento!")

