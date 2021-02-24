# TEST TRAITEMENT DONNEES
import os #path handling
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier #kNN classifier
from sklearn.model_selection import train_test_split #Data set splitting functions
from sklearn.metrics import confusion_matrix #Confusion matrix**
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import random
import copy
import csv
import pandas as pd
random.seed(10)

#Pion Moi : 11
#Pion adverse : 12
#Tour Moi : 51
# Tour adverse : 52
# Cavalier moi : 31
# Cavalier adverse : 32
# Fou moi : 33
# Fou adverse : 34
# Dame moi : 91
# Dame adverse : 92
# Roi moi : 101
# Roi adverse : 102

def initialisation(color):
    # Création du plateau initial 
    e = np.zeros((8,8))
    for k in range(8):
        e[1][k] = 12
        e[6][k] = 11
    e[0][0] = 52
    e[0][7] = 52
    e[7][0] = 51
    e[7][7] = 51
    e[0][1] = 32
    e[0][6] = 32
    e[0][2] = 34
    e[0][5] = 34
    e[7][1] = 31
    e[7][6] = 31
    e[7][2] = 33
    e[7][5] = 33
    if color == 'White':
        e[7][3] = 91
        e[0][3] = 92
        e[7][4] = 101
        e[0][4] = 102
    else:
        e[7][3] = 101
        e[0][3] = 102
        e[7][4] = 91
        e[0][4] = 92
    return e

def displace(pos0,pos1,M):
    # Déplace une pièce d'une position x,y vers une position a,b
    new_M = copy.copy(M)
    stockage = M[pos0[0]][pos0[1]]
    new_M[pos1[0]][pos1[1]] = stockage
    new_M[pos0[0]][pos0[1]] = 0
    return new_M

# Quelques coups arbitraires pour tester
M_0 = initialisation('White')
M_1 = displace([6,1],[5,1],M_0)
M_2 = displace([1,1],[2,1],M_1)
M_2_bis = displace([1,1],[3,1],M_1)
M_3 = displace([6,3],[4,3],M_2)
M_3_bis = displace([6,4],[4,4],M_2_bis)
Matrix = []
Matrix_Y = []

n_positions = 3000


Mx = np.zeros((n_positions,64)) # Entrées (les positions aplaties sur chaque ligne)
My = np.zeros((n_positions,64)) # Sorties

ident = 0 # On représente chaque coup possible par un entier compris en 0 et 64*64 (représentant l'ID d'un coup 
# défini par une position d'origine et d'arrivée)

Ids = np.zeros((64,64)) # Ligne case de départ, colonne case d'arrivée
for i in range(64):
    for j in range(64):
        Ids[i][j] = ident
        ident += 1
        
for k in range(1000):
    Matrix += [M_0]
    Matrix += [M_1]
    Matrix_Y += [M_1]
    Matrix_Y += [M_3]
for k in range(500):
    Matrix += [M_0]
    Matrix += [M_1]
    Matrix_Y += [M_1]
    Matrix_Y += [M_3_bis]

def postolist(L):
    # Fonction qui donne la position dans la liste à partir des coordonnées sur le plateau
    return int(7*L[0]+L[1])

def detect_coup(M0,M1):
    # Fonction qui donne l'identifiant du coup joué à partir des 2 positions.
    M = M1-M0
    for i in range(8):
        for j in range(8):
            if M[i,j]<0:
                depart = [i,j]
            if M[i,j]>0:
                arrivee = [i,j]
    return Ids[postolist(depart)][postolist(arrivee)]


for m in range(len(Matrix)):
    x = []
    y = []
    c = 0
    for i in range(8):
        for j in range(8):
            Mx[m][c] = Matrix[m][i][j]
            My[m][c] = Matrix_Y[m][i][j]
            c += 1
            
List_coups = np.zeros((n_positions,1))
for k in range(len(Matrix)):
    List_coups[k,0] = detect_coup(Matrix[k],Matrix_Y[k])

DFx = pd.DataFrame(Mx)
DFy = pd.DataFrame(List_coups)


trainData,testData,trainY,testY = train_test_split(Mx,List_coups,test_size=0.1) # Séparation en données d'entrainement et données de test
kNN = KNeighborsClassifier(n_neighbors=8,algorithm='kd_tree',metric='minkowski',p=2,n_jobs=-1)
classifier = MultiOutputClassifier(kNN, n_jobs=-1)
classifier.fit(trainData,trainY) # Entrainement du réseau de neurone
trainPredictionsk = classifier.predict(trainData) # Entrainement à la prédiction de la sortie 
trainCMk = confusion_matrix(y_pred=trainPredictionsk,y_true=trainY)
testpredict = classifier.predict(testData)
testCM = confusion_matrix(y_pred=testpredict,y_true=testY)