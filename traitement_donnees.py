import os #path handling
import numpy as np #import numpy drives sklearn to use numpy arrays instead of python lists
import pandas as pd #CSV and dataframe handling
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier #kNN classifier
from sklearn.model_selection import train_test_split #Data set splitting functions
from sklearn.metrics import confusion_matrix #Confusion matrix**
import random
random.seed(10)

DF_to_use = fullDF[['SexId','Age','Fare','Pclass','SibSp','NameLen','Title','Survived']]

dataDF = DF_to_use[['SexId','Age','Fare','Pclass','SibSp','NameLen','Title']]
classDF = DF_to_use['Survived']

S = 0
nb = 100
for k in range(nb): # On moyenne sur 20 tirages aléatoires
    
    random.seed(k) # On fait varier la racine des fonctions random
    trainData,testData,trainY,testY = train_test_split(dataDF,classDF,test_size=0.1) # Séparation en données d'entrainement et données de test
    
    
    kNN = KNeighborsClassifier(n_neighbors=8,algorithm='kd_tree',metric='minkowski',p=2,n_jobs=-1)
    kNN.fit(trainData,trainY) # Entrainement du réseau de neurone
    trainPredictionsk = kNN.predict(trainData)
    trainCMk = confusion_matrix(y_pred=trainPredictionsk,y_true=trainY)
    testpredict = kNN.predict(testData)
    testCM = confusion_matrix(y_pred=testpredict,y_true=testY)
    perfkNN = (testCM[0,0]+testCM[1,1])/sum(sum(testCM)) # On calcule la performance comme le rapport de la trace sur 
    # la somme de tous les coeffcieitns de la matrice de confusion
    S += perfkNN/nb
print(S)    

