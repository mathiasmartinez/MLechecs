def echiquier_debut():
	return [2, 3, 4, 5, 6, 4, 3, 2] + [1]*8 + [0]*32 + [-1]*8 + [-2, -3, -4, -5, -6, -4, -3, -2]

class Partie:
	"""Cette classe représente dynamiquement une partie, en stockant toutes les positions successives, ce qui je l'admet, n'est pas optimal.
	"""
	
	def __init__(self, coups):
		
		(a, b, c ) = calcul(coups)
		self.listepos = a
		self.roque = b							# le premier élément dit si les blancs ont déjà roquer, le deuxième si les noirs ont déja roquer
		self.coup = c	 						# Stocke le dernier coup joué si c'est une poussé de deux d'un pion ( i-1 si c'est le pion blanc de la colonne i qui a été poussé, -i si c'est le noir ) et stocke un autre nombre sinon.



#--------------------------------------------------------------------------------------------------



def legal_sans_echec(echiquier, coup, couleur, roque = (0, 0, 0, 0), coupprec = 100000):
	""" Dit si un coup est légal, mais sans prendre en compte le fait qu'on puisse se mettre échec en le jouant.
	couleur vaut 1 si c'est au blanc de jouer, -1 si c'est au noir.
	Le coup n'est par contre pas effectuer, l'échiquier n'est pas modifier."""
	
	# Bon là il va falloir bourriner les cas. Donc bourrinons.
	
	case_depart = coup[0]
	case_arrivee = coup[1]
	piece_bougee = echiquier[case_depart]
	piece_mangee = echiquier[case_arrivee]
	
	print(case_depart)
	print(case_arrivee)
	print(piece_bougee)
	print(piece_mangee)
	
	# D'abord est ce que on bouge bien une pièce
	if piece_bougee == 0:
		return False
	
	# Ensuite, on regarde si on pousse une pièce de la bonne couleur
	if piece_bougee//abs(piece_bougee) != couleur:
		return False
		
	# Ensuite on vérifie si on arrive pas sur une pièce du joueur qui joue.
	if piece_mangee != 0 and couleur == piece_mangee // abs(piece_mangee):
		return False
	
	# Ensuite, on vérifie qu'on a bien fait un mouvement
	if case_depart == case_arrivee:
		return False
	
	print("1")
	
	# Ensuite traitons le cas où c'est un pion blanc
	if piece_bougee == 1:
		
		# Cas où on pousse le pion
		if case_depart % 8 == case_arrivee % 8:
			if piece_mangee != 0:
				return False
			if case_depart//8 + 1 == case_arrivee//8:
				return True
			if case_depart//8 + 2 == case_arrivee//8:
				return case_depart == 1
			else:
				return False
		
		# Cas où on prend quelque chose avec le pion
		if abs (case_depart % 8 - case_arrivee % 8) == 1:
			if case_depart // 8 + 1 == case_arrivee // 8:
				if piece_mangee != 0:
					return True
		
				if piece_mangee == 0:
					return case_arrivee%8 == abs(coupprec) 	# Ici on test si on fait une prise en passant

		return False

	
	# Ensuite le cas où on joue un pion noir ( on va copié collé sec ).
	if piece_bougee == -1:
		
		# Cas où on pousse le pion
		if case_depart % 8 == case_arrivee % 8:
			if piece_mangee != 0:
				return False
			if case_depart//8 - 1 == case_arrivee//8:
				return True
			if case_depart//8 - 2 == case_arrivee//8:
				return case_depart == 6
			else:
				return False
		
		# Cas où on prend quelque chose avec le pion
		if abs (case_depart % 8 - case_arrivee % 8) == 1:
			if case_depart // 8 - 1 == case_arrivee // 8:
				if piece_mangee != 0:
					return True
		
				if piece_mangee == 0:
					return case_arrivee%8 == abs(coupprec) 	# Ici on test si on fait une prise en passant

		return False
	
	
	# Ensuite le cas où on joue une tour ( blanc ou noir )
	
	if abs(piece_bougee) == 2:
		
		# cas où la tour se déplace verticalement
		if case_depart%8 == case_arrivee%8:
			print(2)
			# On regarde si elle passe sur une case non vide
			for i in range(min(case_depart//8, case_arrivee//8)+1, max(case_depart//8, case_arrivee//8), 8):
				print(i)
				if echiquier[i] != 0:
					return False
			return True
		
		# Le cas où elle se déplace horizontalement
		if case_depart//8 == case_arrivee // 8:
		
			for i in range(min(case_depart%8, case_arrivee%8)+1, max(case_depart%8, case_arrivee%8)):
				if echiquier[i] != 0:
					return False
			return True
	
		return False
	
	
	# Ensuite le cas où la pièce est un cavalier
	
	if abs(piece_bougee) == 3:
		
		dx = case_depart%8 - case_arrivee%8
		dy = case_depart//8 - case_arrivee//8
		
		return ( (abs(dx), abs(dy)) in [(1, 2), (2, 1)] )
	
	
	# Ensuite, le cas où la pièce a bouger est un fou
	
	if abs(piece_bougee) == 4:
		
		dx = case_depart%8 - case_arrivee%8
		dy = case_depart//8 - case_arrivee//8

		if abs(dx) != abs(dy):
			return False
		
		# La je fais des disjonctions de cas pour échanger de la malice contre de la sueur

		if dx > 0 and dy > 0:
			for i in range(1, dx):
				if echiquier[case_arrivee + i + 8*i] != 0:
					return False
			return True
		
		
		if dx > 0 and dy < 0:
			for i in range(1, dx):
				if echiquier[case_arrivee + i - 8*i] != 0:
					return False
			return True
		
		if dx < 0 and dy > 0:
			for i in range(1, -dx):
				if echiquier[case_arrivee - i + 8*i] != 0:
					return False
			return True
		
		if dx < 0 and dy < 0:
			for i in range(1, -dx):
				if echiquier[case_arrivee - i - 8*i] != 0:
					return False
			return True
		
	
	# Ensuite traitons le cas où la pièce jouée est une dame
	
	if abs(piece_bougee) == 5:
		
		# Je vais juste copié collé les trucs du fou et de la tour
		
		# cas où la dame se déplace verticalement
		if case_depart%8 == case_arrivee%8:
			
			# On regarde si elle passe sur une case non vide
			for i in range(min(case_depart//8, case_arrivee//8)+1, max(case_depart//8, case_arrivee//8), 8):
				if echiquier[i] != 0:
					return False
			return True
		
		# Le cas où elle se déplace horizontalement
		if case_depart//8 == case_arrivee // 8:
		
			for i in range(min(case_depart%8, case_arrivee%8)+1, max(case_depart%8, case_arrivee%8)):
				if echiquier[i] != 0:
					return False
			return True
	
		# Maintenant, il ne reste que la possibilité d'un déplacement diagonal que nous allons tester.
		
		dx = case_depart%8 - case_arrivee%8
		dy = case_depart//8 - case_arrivee//8

		if abs(dx) != abs(dy):
			return False
		
		# La je fais des disjonctions de cas pour échanger de la malice contre de la sueur
		
		if dx > 0 and dy > 0:
			for i in range(1, dx):
				if echiquier[case_arrivee + i + 8*i] != 0:
					return False
			return True
		
		
		if dx > 0 and dy < 0:
			for i in range(1, dx):
				if echiquier[case_arrivee + i - 8*i] != 0:
					return False
			return True
		
		if dx < 0 and dy > 0:
			for i in range(1, -dx):
				if echiquier[case_arrivee - i + 8*i] != 0:
					return False
			return True
		
		if dx < 0 and dy > 0:
			for i in range(1, -dx):
				if echiquier[case_arrivee - i - 8*i] != 0:
					return False
			return True


	# Maintenant, le cas où on bouge le roi, sachant que le roque est symbolisé par un déplacement de deux cases du roi
	
	if abs(piece_bougee) == 6:
	
		dx = case_depart%8 - case_arrivee%8
		dy = case_depart//8 - case_arrivee//8
		
		if abs(dx) < 2 and abs(dy) < 2:
			return True
		
		# Maintenant il faut traiter le cas hyper chiant du roque
		return False




	
		
