def echiquier():
	return [2, 3, 4, 5, 6, 4, 3, 2] + [1]*8 + [0]*32 + [-1]*8 + [-2, -3, -4, -5, -6, -4, -3, -2]

class Partie:
	"""Cette classe représente dynamiquement une partie, en stockant toutes les positions successives, ce qui je l'admet, n'est pas optimal.
	"""
	
	def __init__(self, coups):
		
		(a, b, c ) = calcul(coups)
		self.listepos = a
		self.roque = b							# le premier élément dit si les blancs ont déjà roquer, le deuxième si les noirs ont déja roquer
		self.coup = c	 						# Stocke le dernier coup joué si c'est une poussé de deux d'un pion ( i si c'est le pion blanc de la colonne i qui a été poussé, -i si c'est le noir ) et stocke 0 sinon.
	

def legal_sans_echec(echiquier, coup, roque = (0, 0), coupprec = 0, couleur):
	""" Dit si un coup est légal, mais sans prendre en compte le fait qu'on puisse se mettre échec en le jouant.
	couleur vaut 1 si c'est au blanc de jouer, -1 si c'est au noir."""
	
	# Bon là il va falloir bourriner les cas. Donc bourrinons.
	
	case_depart = coup[0]
	case_arrivee = coup[1]
	piece_bougee = echiquier[depart]
	
	# D'abord est ce que on bouge bien une pièce
	if piece_bougee == 0:
		return False
	
	# Ensuite, on regarde si on pousse une pièce de la bonne couleur
	if piece_bougee//abs(piece_bougee) != couleur:
		return False
		
	# Ensuite on vérifie si on arrive pas sur une pièce du joueur qui joue.
	if echiquier[case_arrivee] != 0 and couleur == echiquier[case_arrivee] // abs(echiquier[case_arrivee]):
		return False
	
	# Ensuite traitons le cas où 
