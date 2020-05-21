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

def offlineLearning(Xtrain, dtrain):
    n = len(Xtrain[0])
    w = np.random.rand(n, 1)
    epoch = 0
    
    while True:
        v = Xtrain * w
        y = logsig(v)
        e = y - dtrain
        g = np.transpose(Xtrain) * np.multiply(e, gradiens(v))
        w = w - lr * g
        E = np.sum(np.power(e, 2))
        
        if stop(E, epoch):
            break
        
        epoch += 1
        
    return w, E
        
def stop(E, epoch):
    if epoch > 10000:
        return True
    
    if len(E) < 10:
        return False
    
    end = len(E) - 1
    
    if E[end - 9] < E[end] or E[end - 9] - E[end] < 0.001 :
        return True
    
    return False

def logsig(v):
    for i in range(0, len(v)):
        v[i][0] = gradiens(v[i][0])
        
    return v
    
def sigmoid(x):
    return 1/(1 + math.exp(-x))

def gradiens(x):
    return sigmoid(x) * (1 - sigmoid(x))


Xtrain, Xtest, dtrain, dtest = betolt_kepek()
offlineLearning(Xtrain, dtrain)
