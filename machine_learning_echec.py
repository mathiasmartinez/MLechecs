import numpy as np
import copy
f = open('parties.txt','r')
c=0
parties_w = []
parties_b = []
parties  = []
for ligne in f :
    if ligne[8:15]=='jlm2017':
        color = ligne[1:6]
    if ligne[0]=='1' and ligne[1]=='.':
        
        #parties += [[color,str.split(ligne)]]
        if color=='White':
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
def initialisation(color):
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
    print(e)
    
def pawn_w(s):
    l = s[0]
    n = int(s[1])
    if n==6:
        return [l+str(int(n)+1),l+str(int(n)+2)]
    else :
        return [l+str(int(n)+1)]

def pawn_b(s):
    l = s[0]
    n = int(s[1])
    if n==6:
        return [l+str(int(n)-1) , l+str(n-2)]
    else :
        return [l+str(n-1)]

def bish(s):
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
    return letters[x] + str(y)
    
def strtopos(s):
    l=s[0]
    n=int(s[1])
    for k in range(8):
        if l==letters[k]:
            pos = k
    return [pos,n]

def g_roque_me(M,color):
    if color == 'White':
        M[7,4] = 0
        M[7,2] = 101
        M[7,0] = 0
        M[7,3] = 51
    if color == 'Black':
        M[7,3] = 0
        M[7,5] = 101
        M[7,7] = 0
        M[7,4] = 51
    return 0

def g_roque_adv(M,color):
    if color == 'White': # Ma couleur à moi donc ca inverse
        M[0,4] = 0
        M[0,2] = 101
        M[0,0] = 0
        M[0,3] = 51
    if color == 'Blcak':
        M[0,3] = 0
        M[0,5] = 101
        M[0,7] = 0
        M[0,4] = 51
    return 0

def p_roque_me(M,color):
    if color == 'White':
        M[7,4] = 0
        M[7,6] = 101
        M[7,7] = 0
        M[7,5] = 51
    if color == 'Black':
        M[7,3] = 0
        M[7,1] = 101
        M[7,0] = 0
        M[7,2] = 51
    return 0

def p_roque_adv(M,color):
    if color == 'White': # Ma couleur à moi donc ca inverse
        M[0,4] = 0
        M[0,2] = 101
        M[0,0] = 0
        M[0,3] = 51
    if color == 'Blcak':
        M[0,3] = 0
        M[0,5] = 101
        M[0,7] = 0
        M[0,4] = 51
    return 0

def detect_pawn(s):
    if len(s)==2:
        
liste_M = []
color = 'White'
for partie in parties_w:
    Mat = initialisation(color)
    for k in range(int(len(partie)/3)):
        partie.pop(2*k)
    compt = 0
    for c in partie:
        
        if len(c)==2:
            if compt%2==0:
                for m in Mat:
                    if m==11 and pawn_w()==
            
    