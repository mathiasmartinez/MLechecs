import numpy as np
import copy

# Pour la suite :
    # - Je situe toujours mes pièces en bas de la matrice
    # - J'exclue les parties ou il ya plusieurs dames (impossible pour le ML a mon avis)
    
    
f = open('parties.txt','r')
c=0
parties_w = [] # Parties ou je joue les blancs
parties_b = []
parties  = []
for ligne in f :
    if ligne[8:15]=='jlm2017':
        color = ligne[1:6]
    if ligne[0]=='1' and ligne[1]=='.':
        
        #parties += [[color,str.split(ligne)]]
        if color=='White': # Séparation des parties ou je joueles blancs
            parties_w += [str.split(ligne)]
        else:
            parties_b += [str.split(ligne)]
            


letters = ['a','b','c','d','e','f','g','h']
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


def bish(s):
    # Cases possibles d'un fou
    l=s[0]
    n=int(s[1])
    for k in range(8):
        if l==letters[k]:
            pos = k
    p = []
    x = copy.copy(pos)
    y = copy.copy(n)
    while x>=1 and y>=1:
        x = x -1
        y=y-1
        p+=[letters[x]+str(y)]
    x = copy.copy(pos)
    y = copy.copy(n)
    while x<=6 and y>=1:
        x = x+1
        y= y-1
        p+=[letters[x]+str(y)]
        
    x = copy.copy(pos)
    y = copy.copy(n)
    while x>=1 and y<=6:
        x = x-1
        y= y+1
        p+=[letters[x]+str(y)]
    x = copy.copy(pos)
    y = copy.copy(n)
    while x<=6 and y<=6:
        x = x+1
        y=  y+1
        p+=[letters[x]+str(y)]
    return  p
    
def tow(s):
    # Cases possibles d'une tour
    l=s[0]
    n=int(s[1])
    p = []
    for k in range(8):
        if k != n:
            p += [l + str(k)]
    for j in letters:
        if j!=l:
            p+= [ j + str(n)]
    return p

def king(s):
    # Cases possibles du roi
    l=s[0]
    n=int(s[1])
    for k in range(8):
        if l==letters[k]:
            pos = k
    p = []
    if n<=6:
        p+= [l+str(n+1)]
    if n>=1:
        p+= [l+str(n-1)]
    if pos<=6:
        p+= [letters[pos+1]+str(n)]
    if pos>=1 :
        p+= [letters[pos-1]+str(n)]
    if pos<=6 and n<=6:
        p+= [letters[pos+1]+str(n+1)]
    if pos>=1 and n>=1:
        p+= [letters[pos+1]+str(n-1)]
    if pos>=1 and n<=6:
        p+= [letters[pos-1]+str(n+1)]
    if pos>=1 and n>=1:
        p+= [letters[pos-1]+str(n-1)]
    return p
    

def queen(s):
    return [tow(s),bish(s)]

def knight(s):
    l=s[0]
    n=int(s[1])
    for k in range(8):
        if l==letters[k]:
            pos = k
    p= []
    x = pos + 2
    y = n + 1
    if x<=7 and y<=7 and x>=0 and y>=0:
        p+= [letters[x]+str(y)]
    x = pos + 2
    y = n - 1
    if x<=7 and y<=7 and x>=0 and y>=0:
        p+= [letters[x]+str(y)]
    x = pos+1
    y = n + 2
    if x<=7 and y<=7 and x>=0 and y>=0:
        p+= [letters[x]+str(y)]
    x = pos + 1
    y = n - 2 
    if x<=7 and y<=7 and x>=0 and y>=0:
        p+= [letters[x]+str(y)]
    x = pos - 2 
    y = n + 1
    if x<=7 and y<=7 and x>=0 and y>=0:
        p+= [letters[x]+str(y)]
    x = pos - 2
    y = n - 1
    if x<=7 and y<=7 and x>=0 and y>=0:
        p+= [letters[x]+str(y)]
    x = pos - 1
    y = n + 2
    if x<=7 and y<=7 and x>=0 and y>=0:
        p+= [letters[x]+str(y)]
    x = pos - 1
    y = n - 2
    if x<=7 and y<=7 and x>=0 and y>=0:
        p+= [letters[x]+str(y)]
    return p
    
def placer(L):
    # Fonction permettant de placer une case sous forme de string dans la matrice de la partie
    M = np.zeros((8,8))
    for s in L:
        l=s[0]
        n=int(s[1])
        for k in range(8):
            if l==letters[k]:
                pos = k
        M[n,pos] = 1
    return M
        

def postostr(x,y):
    # Conversion de la position en case string
    return letters[x] + str(y)
    
def strtopos(s):
    
    # Conversion d'une case string en position dans la matrice
    l=s[-2:-1]
    n=int(s[-1:])
    
    for k in range(8):
        if l==letters[k]:
            pos_search = k
    return [pos_search,8-n]

def g_roque_me(M,color):
    # Déplacement des poièces pour un grand roque de moi
    new_M = copy.copy(M)
    if color == 'White':
        new_M[7,4] = 0
        new_M[7,2] = 101
        new_M[7,0] = 0
        new_M[7,3] = 51
    if color == 'Black':
        new_M[7,3] = 0
        new_M[7,5] = 101
        new_M[7,7] = 0
        new_M[7,4] = 51
    return new_M

def g_roque_adv(M,color):
    new_M = copy.copy(M)
    if color == 'White': # Ma couleur à moi donc ca inverse
        new_M[0,4] = 0
        new_M[0,2] = 101
        new_M[0,0] = 0
        new_M[0,3] = 51
    if color == 'Black':
        new_M[0,3] = 0
        new_M[0,5] = 101
        new_M[0,7] = 0
        new_M[0,4] = 51
    return new_M

def p_roque_me(M,color):
    new_M = copy.copy(M)
    if color == 'White':
        new_M[7,4] = 0
        new_M[7,6] = 101
        new_M[7,7] = 0
        new_M[7,5] = 51
    if color == 'Black':
        new_M[7,3] = 0
        new_M[7,1] = 101
        new_M[7,0] = 0
        new_M[7,2] = 51
    return new_M

def p_roque_adv(M,color):
    new_M = copy.copy(M)
    if color == 'White': # Ma couleur à moi donc ca inverse
        new_M[0,4] = 0
        new_M[0,2] = 101
        new_M[0,0] = 0
        new_M[0,3] = 51
    if color == 'Black':
        new_M[0,3] = 0
        new_M[0,5] = 101
        new_M[0,7] = 0
        new_M[0,4] = 51
    return new_M

    
def displace(pos0,pos1,M,player):
    # Déplace une pièce d'une position x,y vers une position a,b
    new_M = copy.copy(M)
    stockage = M[pos0[0]][pos0[1]]
    new_M[pos1[0]][pos1[1]] = stockage
    new_M[pos0[0]][pos0[1]] = 0
    print(new_M)
    return new_M

def v_knight(s,M,prop):
    # Vérifie si un cavalier peut se déplacer vers la case s et renvoie la position de ce cavalier
    if prop=='me':
        value = 31
    else:
        value = 32
    for i in range(8):
        for j in range(8):
            if M[i][j]== value:
                for place in knight(postostr(i,j)):
                    if place==s:
                        return [i,j]
   
def v_bish(s,M,prop):
    # Vérifie si un fou peut se déplacer vers la case s et renvoie la position de ce fou
    if prop=='me':
        value = 33
    else:
        value = 34
    
    for i in range(8):
        for j in range(8):
            if M[i][j]== value:
                for place in bish(postostr(i,j)):
                    if place==s:
                        return [i,j]
def v_queen(s,M,prop):
    # Vérifie si une dame peut se déplacer vers la case s et renvoie la position de cette dame
    if prop=='me':
        value = 91
    else:
        value = 92
    for i in range(8):
        for j in range(8):
            if M[i][j]== value:
                for place in queen(postostr(i,j)):
                    if place==s:
                        return [i,j]              
   
def v_tow(s,M,prop):
# Vérifie si une tour peut se déplacer vers la case s et renvoie la position de cette tour
    if prop=='me':
        value = 51
    else:
        value = 52
    for i in range(8):
        for j in range(8):
            if M[i][j]== value:
                for place in tow(postostr(i,j)):
                    if place==s:
                        return [i,j]
  
def v_king(s,M,prop):
# Vérifie si un roi peut se déplacer vers la case s et renvoie la position de ce roi
    if prop=='me':
        value = 101
    else:
        value = 102
    for i in range(8):
        for j in range(8):
            if M[i][j]== value:
                for place in king(postostr(i,j)):
                    if place==s:
                        return [i,j]

def pawn_me(s,M):
    # Cases possible permettant à un pion à moi d'arriver en s
    l = s[0]
    n = int(s[1])
    p = [] # Liste des positions possibles
    #if M[strtopos(l+str(n+1))[1]][strtopos(l+str(n+1))[0]]==0: # On vérifie que la case est vide
    p += [l+str(int(n)-1)]
    if n==4 and M[strtopos(l+str(n+1))[1]][strtopos(l+str(n+1))[0]]==0:
        p+= [l+str(int(n)-2)] # Le pion n'a pas encore bougé, il peut avancer de 2 cases
    if strtopos(s)[0]>=1 and M[strtopos(s)[1]+1][strtopos(s)[0]-1]%2 == 0 and M[strtopos(s)[1]+1][strtopos(s)[0]-1] !=0:
        p+= [postostr(strtopos(s)[0]-1,strtopos(s)[1]-1)] # Les axes sont inversés quand on passe en string
    if strtopos(s)[0] <=6 and M[strtopos(s)[1]+1][strtopos(s)[0]+1]%2 == 0 and M[strtopos(s)[1]+1][strtopos(s)[0]+1] !=0:
        p+= [postostr(strtopos(s)[0]+1,strtopos(s)[1]-1)]
    return p
    
def pawn_adv(s,M):
    # Cases possible permettant à un pion adverse d'arriver en s
    l = s[0]
    n = int(s[1])
    p=[]
    #if M[strtopos(l+str(n-1))[1]][strtopos(l+str(n-1))[0]]==0: # On vérifie que la case est vide
    p += [l+str(int(n)+1)]
    if n==5 and M[strtopos(l+str(n-1))[1]][strtopos(l+str(n-1))[1]]==0:
        p+= [l+str(n+2)]
    if strtopos(s)[0]>=1 and M[strtopos(s)[1]-1][strtopos(s)[0]-1] % 2 == 0 and M[strtopos(s)[1]-1][strtopos(s)[0]-1] !=0:
        p+= [postostr(strtopos(s)[0]-1,strtopos(s)[1]+1)]
    if strtopos(s)[0]<=6 and M[strtopos(s)[1]-1][strtopos(s)[0]+1] % 2 == 0 and M[strtopos(s)[1]-1][strtopos(s)[0]+1] !=0:
        p+= [postostr(strtopos(s)[0]+1,strtopos(s)[1]+1)]
    return p

def detect_move(s,M,player,color):
    
    s = s.replace("#", "")
    s = s.replace("+", "")
    if len(s)==2 and player=='me':
        for pos in pawn_me(s,M):
            if M[strtopos(pos)[1]][strtopos(pos)[0]]==11:
                return displace(strtopos(pos),strtopos(s),M,player)
    if len(s)==2 and player=='adv':
        for pos in pawn_adv(s,M):
            if M[strtopos(pos)[1]][strtopos(pos)[0]]==12:
                return displace(strtopos(pos),strtopos(s),M,player)
    if s[0]=='N':
        return displace(v_knight(s[-2:],M,player),strtopos(s),M,player) 
    if s[0]=='B':
        
        return displace(v_bish(s[-2:],M,player),strtopos(s),M,player)
    if s[0]=='Q':
        return displace(v_queen(s[-2:],M,player),strtopos(s),M,player)
    if s[0]=='R':
        return displace(v_tow(s[-2:],M,player),strtopos(s),M,player) 
    if s[0]=='K':
        return displace(v_king(s[-2:],M,player),strtopos(s),M,player)
    if s=='O-O' and player=='me':
        return g_roque_me(M,color)
    if s=='o-o' and player=='me':
        return p_roque_me(M,color)
    if s=='O-O' and player=='adv':
        return g_roque_adv(M,color)
    if s=='o-o' and player=='me':
        return g_roque_adv(M,color)
    
    
liste_M = []
color = 'White'
compt_glob = 0 # permet de recuperer l'état t-1
for partie in parties_w:
    
    Mat = initialisation(color)
    liste_M += [Mat]
    compt_loc = 0
    for k in range(int(len(partie)/3)):
        partie.pop(2*k) # On supprime les numeros des coups
    #print(partie)
    for swag in partie:
        
        if compt_loc%2==0:
            
            liste_M += [detect_move(swag,liste_M[compt_glob],'me','White')]
        else:
            
            liste_M += [detect_move(swag,liste_M[compt_glob],'adv','White')]
                                
        compt_glob += 1
        compt_loc += 1
 

# TEST TRAITEMENT DONNEES
import os #path handling
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier #kNN classifier
from sklearn.model_selection import train_test_split #Data set splitting functions
from sklearn.metrics import confusion_matrix #Confusion matrix**
import random
random.seed(10)

M_0 = initialisation('White')
M_1 = displace(6,1,5,1,M_0)


Matrix_datas = []
Matrix_Y = []

# trainData,testData,trainY,testY = train_test_split(Matrix_datas,Matrix_Y,test_size=0.1) # Séparation en données d'entrainement et données de test

# kNN = KNeighborsClassifier(n_neighbors=8,algorithm='kd_tree',metric='minkowski',p=2,n_jobs=-1)
# kNN.fit(trainData,trainY) # Entrainement du réseau de neurone
# trainPredictionsk = kNN.predict(trainData)
# trainCMk = confusion_matrix(y_pred=trainPredictionsk,y_true=trainY)
# testpredict = kNN.predict(testData)
# testCM = confusion_matrix(y_pred=testpredict,y_true=testY)
    