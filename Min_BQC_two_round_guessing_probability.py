"""
"""

from sympy import *

# To Do:
#	> write header
#	> Comment properly

alpha1, alpha2, alpha3, pi = symbols('alpha1 alpha2 alpha3 pi')
alpha11, alpha21, alpha31 = symbols('alphatilde1 alphatilde2 alphatilde3')
alpha12, alpha22, alpha32 = symbols('alphahat1 alphahat2 alphahat3')
r1, r2, r3, c1 = symbols('r1 r2 r3 c1')


# The reported angles for g (equiv g1):
alpha11 = alpha1 + r1*pi
alpha21 = (-1)**(r1+c1)*alpha2 + r2*pi
alpha31 = alpha3 + (r3 + r1 + c1)*pi


# The reported angles for g' (equiv g2):
alpha12 = alpha1 + r1*pi
alpha22 = alpha2 + (r2 + r1 + c1)*pi
alpha32 = (-1)**(r1 + c1)*alpha3 + r3*pi



def g_image(a1,a2,a3,c):
	"""
	"""
	image_set = []
	for x3 in [0,1]:
		for x2 in [0,1]:
			for x1 in [0,1]:
				new_angles = (simplify(alpha11.subs([(r1,x1),(alpha1,a1)]) 
									   %(2*pi)), 
							  simplify(alpha21.subs([(r1,x1),(r2,x2),(c1,c), 
							  						(alpha2 a2)]) 
							  		   %(2*pi)), 
							  simplify(alpha31.subs([(r1,x1),(c1,c),(r3,x3), 
							  						(alpha3,a3)])
							  		   %(2*pi)))
				angle_is_new = True
				for angles in image_set:
					if (simplify((angles[0] - new_angles[0])%(2*pi)) == 0 
						and simplify((angles[1] - new_angles[1])%(2*pi)) == 0 
						and simplify((angles[2] - new_angles[2])%(2*pi)) == 0):
						angle_is_new = False
						break
				if angle_is_new:
					image_set.append(new_angles)
	return image_set


def g_prime_image(a1,a2,a3,c):
	"""
	"""
	image_set = []
	for x3 in [0,1]:
		for x2 in [0,1]:
			for x1 in [0,1]:
				new_angles = (simplify(alpha12.subs([(r1,x1),(alpha1,a1)])
									   %(2*pi)), 
							  simplify(alpha22.subs([(r1,x1),(r2,x2),(c1,c),
							  		   				(alpha2,a2)])
							  		   %(2*pi)), 
							  simplify(alpha32.subs([(r1,x1),(c1,c),(r3,x3),
							  						(alpha3,a3)])
							  		   %(2*pi)))
				angle_is_new = True
				for angles in image_set:
					if (simplify((angles[0] - new_angles[0])%(2*pi)) == 0 
						and simplify((angles[1] - new_angles[1])%(2*pi)) == 0 
						and simplify((angles[2] - new_angles[2])%(2*pi)) == 0):
						angle_is_new = False
						break
				if angle_is_new:
					image_set.append(new_angles)
	return image_set


def set_intersection(list1, list2):
	""" Computes the set intersection of list1 and list2 under the assumption 
	that each of list1 and list2 has no repeated elements.
	"""
	intersection = []
	for angles1 in list1:
		for angles2 in list2:
			if (simplify((angles1[0] - angles2[0])%(2*pi)) == 0 
				and simplify((angles1[1] - angles2[1])%(2*pi)) == 0 
				and simplify((angles1[2] - angles2[2])%(2*pi)) == 0):
				intersection.append(angles1)
				break

	return intersection

def set_union(list1, list2):
	""" Computes the set union of list1 and list2 under the assumption 
	that each of list1 and list2 has no repeated elements.
	"""
	union = list1.copy()
	for angles2 in list2:
		not_in_list1 = True
		for angles1 in list1:
			if (simplify((angles1[0] - angles2[0])%(2*pi)) == 0 
				and simplify((angles1[1] - angles2[1])%(2*pi)) == 0 
				and simplify((angles1[2] - angles2[2])%(2*pi)) == 0):
				not_in_list1 = False
				break
		if not_in_list1:
			union.append(angles2)

	return union

def set_union_without_intersection(list1, list2):
	""" Computes the set symmetric difference of list1 and list2 under the  
	assumption that each of list1 and list2 has no repeated elements.
	"""
	union_without_intersection = []
	union = set_union(list1, list2)
	intersection = set_intersection(list1, list2)
	for angles_union in union:
		in_intersection = False
		for angles_int in intersection:
			if (simplify((angles_union[0] - angles_int[0])%(2*pi)) == 0 
				and simplify((angles_union[1] - angles_int[1])%(2*pi)) == 0 
				and simplify((angles_union[2] - angles_int[2])%(2*pi)) == 0):
				in_intersection = True
				break
		if not in_intersection:
			union_without_intersection.append(angles_union)
	return union_without_intersection


# # # # # # # # # # # # Computing Guessing Probability # # # # # # # # # # # # 

total = 0

for d2 in [0,1]:
	for d1 in [0,1]:
		g_g_prime_int = set_intersection(g_image(alpha1, alpha2, alpha3, d1),
										 g_prime_image(alpha1, alpha2, alpha3, 
										 			   d1))
		g_g_prime_union_without_int = set_union_without_intersection(
											g_image(alpha1, alpha2, alpha3, d1), 
											g_prime_image(alpha1, alpha2, 
														  alpha3, d1))

		g_of_int = []
		g_prime_of_int = []
		for angles in g_g_prime_int:
			g_of_int = set_union(g_of_int, 
								 g_image(angles[0], angles[1], angles[2], d2))
			g_prime_of_int = set_union(g_prime_of_int, 
									   g_prime_image(angles[0], angles[1], 
									   				 angles[2], d2))

		g_of_union_without_int = []
		g_prime_of_union_without_int = []
		for angles in g_g_prime_union_without_int:
			g_of_union_without_int = set_union(g_of_union_without_int, 
											   g_image(angles[0], angles[1], 
											   		   angles[2], d2))
			g_prime_of_union_without_int = set_union(
												   g_prime_of_union_without_int, 
												   g_prime_image(angles[0], 
												   				 angles[1], 
												   				 angles[2], 
												   				 d2))

		both_rounds_both_gflows = set_intersection(g_of_int, g_prime_of_int)
		one_and_two_gflows = set_union(set_union_without_intersection(g_of_int, 
																g_prime_of_int), 
									   set_intersection(g_of_union_without_int, 
									   			  g_prime_of_union_without_int))
		both_rounds_one_gflow = set_union_without_intersection(
														 g_of_union_without_int, 
												   g_prime_of_union_without_int)

		total += (4*len(both_rounds_both_gflows) 
				  + 2*len(one_and_two_gflows) 
				  + len(both_rounds_one_gflow))
