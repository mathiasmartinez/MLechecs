# Ici, le but est d'afficher une position
# Principalement pour pouvoir faire des tests
# Mais je vais juste pour l'instant renvoyer une position au format FEN, qu'on peut faire afficher dans lichess.org

from partie import *

cp = {"k" : -6, "q" : -5, "b" : -4, "n" : -3, "r" : -2, "p" :-1, "P" : 1, "R" : 2, "N" : 3, "B" : 4, "Q" : 5, "K" : 6} # Dictionnaire qui stocke les correspondances entre les pièces dans nos echiquier et en code FEN

pc = dict()
for i in cp.keys():
	pc[cp[i]] = i

pc[0] = "1"


def FEN_vers_ech(fen):
	
	global pc, cp
	
	res = []
	
	for c in fen:
		if c != "/":
			if ord(c) <= 57 and ord(c) >= 48:
				res += [0]*(ord(c) - 48)
			else:
				res.append(cp[c])
		else:
			res[-1], res[-8] = res[-8], res[-1]
			res[-2], res[-7] = res[-7], res[-2]
			res[-3], res[-6] = res[-6], res[-3]
			res[-4], res[-5] = res[-5], res[-4]
	
	return [res[-i-1] for i in range(len(res))]

def ech_vers_FEN(ech):
	
	global pc, cp
	res = ""
	
	for i in range(64):
		if i % 8 == 0 and i > 0:
			res += "/"
		res += pc[ech[i]]
	
	return res

def affiche_mal(ech):
	for i in range(7, -1, -1):
		for j in range(8):
			c = ech[8*i + j]
			if c >= 0:
				print(c, end = "  ")
			else:
				print(c, end = " ")
		print("\n")


#--------------------------------------------------------------------------------------------------

while 1:
	ech = FEN_vers_ech(input("FEN : "))
	coup = int(input("coup 1 : ")), int(input("coup 2 : "))
	print(legal_sans_echec(ech, coup, 1))
