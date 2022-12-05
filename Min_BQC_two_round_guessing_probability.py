"""
This script calculates the guessing probability of D_{client}^{(2)} (see 
Min_BQC_generate_comb_as_list.py and the manuscript for more details on 
notation) via symbolic means rather than the semi-definite programming approach.
This script is distinct from the other scripts in this repository and stands
alone.

This script contains five functions:

	> g_one_image 			computes the set of angles in the image (equiv. 
							pre-image) of the function defined for the gflow g1 
							given a specific angle and classical message, and as 
							the one-time-pads vary;
	> g_two_image 			computes the set of angles in the image (equiv. 
							pre-image) of the function defined for the gflow g2 
							given a specific angle and classical message, and as 
							the one-time-pads vary;
	> set_intersection 		calculates the intersection of two sets;
	> set_union 			calculates the union of two sets;
	> set_sym_diff			calculates the symmetric difference of two sets;

Computing Guessing Probability: 
Using the above function, the guessing probabiltiy for D_{client}^{(2)} is
calculated via the sizes of various sets given by the intersection, union and
symmetric differences of images and pre-images of g1 and g2. See below and also 
Proposition c.2 in Appendix C of the manuscript for more details.

This work was conducted within the Quantum Information and Computation group at 
the University of Innsbruck.
Contributors: Isaac D. Smith, Marius Krumm

This work is licensed under a Creative Commons by 4.0 license.
"""

from sympy import *


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



def g_one_image(a1,a2,a3,c):
	""" Calculates the set of angles the are reportable from the angles 
	(a1,a2,a3) by the gflow g1 given classical message c as the one-time-pads
	vary. This is equivalent to the set of angles that could report the angles
	(a1,a2,a3) given c as the one-time-pads vary.

	Inputs:
		a1,a2,a3 	angles, represented as a symbolic expression;
		c 			Int, either 0 or 1;

	Returns:
		List of tuples (b1,b2,b3) where b1,b2,b3 are angles.
	"""
	image_set = []
	for x3 in [0,1]:
		for x2 in [0,1]:
			for x1 in [0,1]:
				new_angles = (simplify(alpha11.subs([(r1,x1),(alpha1,a1)]) 
									   %(2*pi)), 
							  simplify(alpha21.subs([(r1,x1),(r2,x2),(c1,c), 
							  						(alpha2, a2)]) 
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


def g_two_image(a1,a2,a3,c):
	""" Calculates the set of angles the are reportable from the angles 
	(a1,a2,a3) by the gflow g2 given classical message c as the one-time-pads
	vary. This is equivalent to the set of angles that could report the angles
	(a1,a2,a3) given c as the one-time-pads vary.

	Inputs:
		a1,a2,a3 	angles, represented as a symbolic expression;
		c 			Int, either 0 or 1;

	Returns:
		List of tuples (b1,b2,b3) where b1,b2,b3 are angles.
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

def set_sym_diff(list1, list2):
	""" Computes the set symmetric difference of list1 and list2 under the  
	assumption that each of list1 and list2 has no repeated elements.
	"""
	sym_diff = []
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
			sym_diff.append(angles_union)
	return sym_diff


# # # # # # # # # # # # Computing Guessing Probability # # # # # # # # # # # # 

"""
The guessing probabiltiy fo D_{client}^{(2)} for the specific example given in 
the manuscript is calculate here via calculating the size of various sets 
pertaining to the intersections and symmetric differences of images and 
pre-images of sets generated by the gflows g1 and g2. This is the content of
the proof of Proposition C.2 in the manuscript, with further explanations given
there. In brief, we aim to calculate the size of each of

	S1: g1(B_int) int g2(B_int)

	S2: (g1(B_int) sym_diff g2(B_int)) union (g1(B_sym_diff) int g2(B_sym_diff))

	S3: g1(B_sym_diff) sym_diff g2(B_sym_diff)

where B_int is the set of angles in the intersection of the pre-images of g1 
and g2, B_sym_diff is the symmetric difference of the same two sets, and int, 
sym_diff and union denote the corresponding set operations. 

The quantity of interest derived from these set sizes is 

	total := sum_{d1, d2} 4*|S1| + 2*|S2| + |S3|

where the sum is over the classical messages c for the two rounds.
"""

total = 0

for d2 in [0,1]:
	for d1 in [0,1]:
		# Intersection of the pre-images of g1 and g2:
		B_int = set_intersection(g_one_image(alpha1, alpha2, alpha3, d1),
								 g_two_image(alpha1, alpha2, alpha3, d1))
		
		# Symmetric difference of the pre-images of g1 and g2:
		B_sym_diff = set_sym_diff(g_one_image(alpha1, alpha2, alpha3, d1), 
								  g_two_image(alpha1, alpha2, alpha3, d1))

		g_one_of_B_int = []			# Image of B_int under g1
		g_two_of_B_int = []			# Image of B_int under g2
		for angles in B_int:
			g_one_of_B_int = set_union(g_one_of_B_int, 
								 	   g_one_image(angles[0], angles[1], 
								 	   			   angles[2], d2))
			g_two_of_B_int = set_union(g_two_of_B_int, 
									   g_two_image(angles[0], angles[1], 
									   			   angles[2], d2))

		g_one_of_B_sym_diff = [] 	# Image of B_sym_diff under g1
		g_two_of_B_sym_diff = []	# Image of B_sym_diff under g2
		for angles in B_sym_diff:
			g_one_of_B_sym_diff = set_union(g_one_of_B_sym_diff, 
											g_one_image(angles[0], angles[1], 
											   		    angles[2], d2))
			g_two_of_B_sym_diff = set_union(g_two_of_B_sym_diff, 
											g_two_image(angles[0], angles[1], 
												   		angles[2], d2))

		S1 = set_intersection(g_one_of_B_int, g_two_of_B_int)
		S2 = set_union(set_sym_diff(g_one_of_B_int, g_two_of_B_int), 
					   set_intersection(g_one_of_B_sym_diff, 
									   	g_two_of_B_sym_diff))
		S3 = set_sym_diff(g_one_of_B_sym_diff, g_two_of_B_sym_diff)

		total += (4*len(S1) + 2*len(S2) + len(S3))
