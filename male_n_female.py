# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:06:20 2019

@author: Apoorva
"""

from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np
from keras.utils import to_categorical 
X=pd.read_csv("male.csv")
y=pd.read_csv("male_target.csv")
X1=pd.read_csv("female.csv")
y1=pd.read_csv("female_target.csv")
frames=[X,X1]
mergedFrames=pd.DataFrame()
mergedFrames=pd.concat(frames, sort=False,axis=0)
frames1=[y,y1]
mergedFrames1=pd.DataFrame()
mergedFrames1=pd.concat(frames1, sort=False,axis=0)
y=[]
for i in mergedFrames1['m']:
    if(i=='m'):
        y.append(1)
    else:
        y.append(0)
        

model=Perceptron(alpha=0.0001, class_weight=None, eta0=1.0,fit_intercept=True, max_iter=1000,penalty=None, random_state=0, shuffle=True, tol=0.001,verbose=0, warm_start=False)
model.fit(mergedFrames[:len(y)],y)
k=model.predict(mergedFrames[len(y):])
k
model.score(mergedFrames[:len(y)],y)