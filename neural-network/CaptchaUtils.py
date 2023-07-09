# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:00:58 2023

@author: Giulia
"""
import cv2
import string
import numpy as np
from os import listdir, makedirs
from os.path import isdir,splitext
from collections import defaultdict

caratteri = string.ascii_lowercase + string.digits

#divide captcha generati nelle singole lettere
def segmenta(imgPath):
    
    imgs=[]
    labels=[]
    files = listdir(imgPath)
    for file in files:
      image = cv2.imread(imgPath + file, 0)
      letters = splitext(file)[0]   #fa split del nome del file nelle sue lettere

      ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV) #threshold filtra immagine in base a matrice di valori; solo i valori compresi in threshold vengono restituiti
      #THRESH_BINARY_INV inverte colori e fa in modo che tutti vadano a 0 o 255 dopo averli filtrati per threshold -> rimuovo rumore
      output = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S) #connectedComponentsWithStats restituisce tutti i componenti connessi nell'immagine (lettere)
      #output = array di tutte le lettere trovate come immagini
      
      num_labels = output[0] #numero lettere trovate
      stats = output[2] #informazioni sulle immagini trovate

      objects = []

      for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA] #prende grandezza immagini

        # rimozione valori rumorosi
        if a > 50:
            x = stats[i, cv2.CC_STAT_LEFT]  #stat individuano primo pixel in ogni immagine(usate per cropping)
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            objects.append((x, y, w, h))

      objects.sort(key=lambda t: t[0])

      num_detected = min(len(objects), 6) #minimo tra numero lettere trovate e numero di lettere dei captcha

      for i in range(num_detected):
        o = objects[i]
        x = o[0]
        y = o[1]
        w = o[2]
        h = o[3]
        img = image[y:y+h, x:x+w] #dà nuove dimensioni dell'immagine come range (taglia matrice)
        rgb=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) #aggiunge canale colore mancante
        labels.append(letters[i]) #valore restituito non preso nel testing
        imgs.append(rgb) #mette immagine corretta in imgs
    return imgs,labels  

def decode(array):  #passa da array a carattere -> usato per ritornare a testo alla fine della predict del pgm
    index = np.argmax(array)
    return caratteri[index]

def encode(char): #crea array di label settate a 0 (in riferimento a caratteri) e aumenta a 1 per il char passato
    arr = np.zeros((len(caratteri),), dtype="uint8")
    index = caratteri.index(char)
    arr[index] = 1
    return arr

def normalize_samples(data, labels):    #trasforma (colore) immagini in valore da 0 a 1 per aumentare precisione
    #Normalizziamo per una precisione migliore
    n_data = np.array(data, dtype="float") / 255.0
    n_labels = np.array(labels)
    return n_data, n_labels


# Questo si avvia per segmentare le immagini originali
if __name__ == "__main__":
      datasetPath="../dataset/captcha/"
      segmentazionePath="../dataset/divisione/"
      counts = defaultdict(int) #coppie chiave valore con lettere come chiavi e tot lettere lette come valore -> serve per assegnare numeri come nomi immagini
      imgs,labels=segmenta(datasetPath)
      for i in range(0,len(imgs)):
       if not isdir(segmentazionePath + labels[i]): #controlla se directory esiste, altrimenti la crea
                makedirs(segmentazionePath + labels[i])
       cv2.imwrite(segmentazionePath+ labels[i] + "/"+str(counts[labels[i]]).zfill(5)+".png", imgs[i])  #salva img in path con il nome corretto (fa padding)
       counts[labels[i]] += 1   #aumenta numero usato come nome immagine di uno
