# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:04:49 2020

@author: lambozsolty
"""


from PIL import Image
import glob
import math
import numpy as np
from sklearn.model_selection import train_test_split

auto_kepek = []
repulo_kepek = []
tanitasi_arany = 0.7
lr = 0.0005

def betolt_kepek():
    for kep in glob.glob("cars/*.jpg"):
        im = Image.open(kep).convert('LA')
        im = im.resize((64, 64))
        im = np.asarray(im)
        im = im.flatten()        
        auto_kepek.append(im)
        
    for kep in glob.glob("airplanes/*.jpg"):
        im = Image.open(kep).convert('LA')
        im = im.resize((64, 64))
        im = np.asarray(im)
        im = im.flatten()
        repulo_kepek.append(im)
        
    autoknr = len(auto_kepek)
    repuloknr = len(repulo_kepek)
    
    kepek = np.concatenate((auto_kepek, repulo_kepek), axis = 0)
    
    p = np.concatenate((np.zeros(autoknr), np.ones(repuloknr)), axis = 0)
    p = p.reshape(autoknr + repuloknr, 1)
            
    Xtrain, Xtest, dtrain, dtest = train_test_split(kepek, p, test_size = tanitasi_arany, random_state = 12)
    
    return Xtrain, Xtest, dtrain, dtest

def offlineLearning(Xtrain, dtrain)

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def gradiens(x):
    return sigmoid(x) * (1 - sigmoid(x))


Xtrain, Xtest, dtrain, dtest = betolt_kepek()


betolt_kepek()
