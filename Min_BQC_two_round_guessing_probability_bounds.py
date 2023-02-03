"""
This script calculates bounds for the guessing probability of D_{client}^{(2)} 
(see Min_BQC_generate_comb_as_list.py and the manuscript for more details on 
notation) for a specific choice of angle set and by direct means rather than via 
the semi-definite programming approach. These bounds are derived from the 
general bounds in Prop 3.2 in the manuscript.

This script contains five functions:

	> generate_full_angle_set 	takes a list of angles as np.float as input and
								returns a list of angles that satisfies the
								required closure properties for the example;
	> possible_true_angles 		takes a given tuple of three angles as np.float
								as input and returns a list of tuples of angles
								that satisfy the closure properties for the 
								input tuple;
	> p 						computes the deterministic probability 
								distribution for a given tuple of reported 
								angles, a tuple of true angles, a choice of 
								gflow, and choices of one-time pads and 
								classical message;
	> upper_bound				computes the upper bound for the guessing 
								probability for D_{client}^{(2)} for a specific
								input angle set;
	> lower_bound				computes the lower bound for the guessing 
								probability for D_{client}^{(2)} for a specific
								input angle set.

This work was conducted within the Quantum Information and Computation group at 
the University of Innsbruck.
Contributors: Marius Krumm, Isaac Smith

This work is licensed under a Creative Commons by 4.0 license.
"""

import numpy as np
import itertools

# # # # # # # # # Helper Functions for Computing the Bounds # # # # # # # # # # 

def generate_full_angle_set(angle_set):
	""" The full angle set is generated based on a smaller set of generating
	angles. The full angle set A must satisfy the closure condition:

			A = {(-1)^{x}a + z*pi mod 2pi | a in A and x,z in {0,1}}
	
	Parameters:
		angle_set   a list of angles as numpy floats;

	Returns:
		completed   a list of angles as numpy floats.
	"""
	completed = angle_set.copy()
	for a in range(len(angle_set)):
		# Compute (-1)a mod 2pi
		aX = np.mod(-1.0*angle_set[a], 2.0*np.pi)
		in_set = False
		for b in range(len(completed)):
			if np.allclose(aX, completed[b]):
				in_set = True
				break
		if not in_set:
			completed.append(aX)

		# Compute a + pi mod 2pi
		aZ = np.mod(angle_set[a]+np.pi, 2.0*np.pi)
		in_set = False
		for b in range(len(completed)):
			if np.allclose(aZ, completed[b]):
				in_set = True
				break
		if not in_set:
			completed.append(aZ)

		# Compute (-1)a + pi mod 2pi
		aXZ = np.mod(-1.0*angle_set[a]+np.pi, 2.0*np.pi)
		in_set = False
		for b in range(len(completed)):
			if np.allclose(aXZ, completed[b]):
				in_set = True
				break
		if not in_set:
			completed.append(aXZ)
	return completed


def possible_true_angles(reported_angles):
	""" This function generates the subset of the full angle set generated by
	the angles reported in the first round of interaction. That is, the set

		Poss = {(a0 + z0pi, (-1)^{x1}a1 + z1pi, (-1)^{x2}a2 + z2pi) 
							|(a0,a1,a2) was reported, z0,z1,z2,x1,x2 in {0,1}}

	is generated. This set provides a smaller set to enumerate over in the 
	computation of the bounds below (see upper_bound() and lower_bound()).

	Parameters:
		reported_angles     list of angles as numpy floats;

	Returns
		possible_set        list of angles as numpy floats.
	"""
	possible_set = [reported_angles.copy()]

	for z0, z1, z2, x1, x2 in itertools.product(range(2), repeat=5): 
		poss_angles = []
		poss_angles.append(np.mod(reported_angles[0]+z0*np.pi, 2.0*np.pi))
		poss_angles.append(np.mod(((-1.0)**x1)*reported_angles[1]+z1*np.pi, 
								  2.0*np.pi))
		poss_angles.append(np.mod(((-1.0)**x2)*reported_angles[2]+z2*np.pi, 
								  2.0*np.pi))

		in_possible_set = False
		for a in possible_set:
			if np.allclose(poss_angles, a):
				in_possible_set = True

		if not in_possible_set:
			possible_set.append(poss_angles.copy())

	return possible_set


def p(aprimes, true_angles, r0, r1, r2, c, g):
	""" Computes the probability P(a'|a, r, c, g) where a' is the three tuple of
	reported angles, a is the three tuple of true angles, r is the tree tuple of 
	bits representing the one-time pads, c is a tuple of classical messages only
	the first of which is relevant for this example, and g is one of the two
	gflows relevant for this example. The probability distribution here is 
	deterministic - it has value 1 when each element of a' is reportable from
	the corresponding element of a given g, c and r, and 0 otherwise.

	Parameters:
		aprimes 		the reported angles as a length three list of numpy 
						floats;
		true_angles 	the true angles as a length three list of numpy floats;
		r0 				binary value, the first one-time pad value;
		r1 				binary value, the second one-time pad value;
		r2 				binary value, the third one-time pad value;
		c 				binary value, the first classical message value;
		g 				binary value, indicating the gflow - g1 or g2 in the
						manuscript.

	Returns:
		prob 			float, either 1.0 or 0.0.
	"""
	prob = 1.0

	# Check whether a'[0] is reportable from a[0] (this is the same for both g1  
	# and g2):
	if not np.allclose(np.mod(aprimes[0]+r0*np.pi, 2*np.pi), true_angles[0]):
		return 0.0
	if g == 0:
		# Check whether a'[1] is reportable from a[1] for g1:
		if not np.allclose(np.mod(((-1.0)**(c+r0))*aprimes[1]+r1*np.pi,2*np.pi),
						   true_angles[1]):
			return 0.0
		# Check whether a'[2] is reportable from a[2] for g1:
		if not np.allclose(np.mod(aprimes[2]+(r2+r0+c)*np.pi, 2*np.pi),
						   true_angles[2]):
			return 0.0
		
	elif g == 1:
		# Check whether a'[1] is reportable from a[1] for g2:
		if not np.allclose(np.mod(aprimes[1]+(r1+r0+c)*np.pi, 2*np.pi),
						   true_angles[1]):
			return 0.0
		# Check whether a'[2] is reportable from a[2] for g2:
		if not np.allclose(np.mod(((-1.0)**(c+r0))*aprimes[2]+r2*np.pi,2*np.pi),
						   true_angles[2]):
			return 0.0
	return prob


# # # # # # # # # # # # # # Computing the Bounds # # # # # # # # # # # # # # #

def upper_bound(full_angle_set):
	""" The upper bound for the guessing probability for D_{client}^{(2)} for 
	the specific example in question is given by

		sum_{a'1,a'2} 1/(|A|^3 2^8) max_{a, c'1, c'2} sum_{g1,g2,r1,r2} 
															 P(a'1|a,c'1,r1,g1)
															 *P(a'2|a,c'2,r2,g2)

	where A is the full set of allowed angles, a'1 (a'2) are the reported angles 
	in round 1 (round 2), the maximum is over a the true angles and c'1, c'2 
	the classical messages for rounds 1 and 2, g1 and g2 denote the choice of
	gflow in rounds 1 and 2, and similarly for r1 and r2 as one-time pads. 

	Parameters:
		full_angle_set 		list of angles as numpy floats

	Returns
		fullprob 			float, the upper bound for the guessing probability.
	"""
	# The normalisation 1/(|A|^3 2^8):
	norm = 1.0/((len(full_angle_set)**3)*(2.0**8))
	fullprob = 0.0
	# Sum over the reported angles a'1:
	for aprimeone0, aprimeone1, aprimeone2 in itertools.product(full_angle_set, 
																repeat=3):
		reportone = [aprimeone0, aprimeone1, aprimeone2]
		# For a'2, we do not need to sum over all A, only those that are in the
		# set generated by the a'1. That is, we are enumerating over the set 
		# of angles for which P(a'1|a,c'1,r1,g1)*P(a'2|a,c'2,r2,g2) isn't 
		# guaranteed a priori to be zero:
		candidate_set = possible_true_angles(reportone.copy())

		# Sum over the reported angles a'2 from the reduced set:
		for aprimetwo0, aprimetwo1, aprimetwo2 in candidate_set:
			reporttwo = [aprimetwo0, aprimetwo1, aprimetwo2]
			maximum = 0.0

			# Maximising over the true angles a. Again, we need only enumerate 
			# over a subset of the full angle set.
			for atrue1, atrue2, atrue3 in candidate_set:
				atrue = [atrue1,atrue2, atrue3]

				# Maximising also over the classical message c'1, c'2:
				for c1, c2 in itertools.product(range(2), repeat=2):
					prob = 0.0
					itrset = itertools.product(range(2), repeat=8)
					# Calculate sum_{g1,g2,r1,r2} P(a'1|a,c'1,r1,g1)
					#										*P(a'2|a,c'2,r2,g2):
					for gone,gtwo,rone0,rone1,rone2,rtwo0,rtwo1,rtwo2 in itrset:
						prob += (p(reportone,atrue,rone0,rone1,rone2,c1,gone)
								 *p(reporttwo,atrue,rtwo0,rtwo1,rtwo2,c2,gtwo))
						#print("prob is {}".format(prob))
					if prob > maximum:
						maximum = prob
						print("max is {}".format(maximum))
			fullprob += maximum
	print("fullprob is {}".format(fullprob))
	fullprob *= norm
	return fullprob

def lower_bound(full_angle_set):
	""" The upper bound for the guessing probability for D_{client}^{(2)} for 
	the specific example in question is given by

		sum_{a'1,a'2,c'1,c'2} 1/(|A|^3 2^10) max_{a} sum_{g1,g2,r1,r2} 
															 P(a'1|a,c'1,r1,g1)
															 *P(a'2|a,c'2,r2,g2)

	where A is the full set of allowed angles, a'1 (a'2) are the reported angles 
	in round 1 (round 2), c'1, c'2 are the classical messages for rounds 1 and
	2, the maximum is only over the true angles a, g1 and g2 denote the choice 
	of gflow in rounds 1 and 2, and similarly for r1 and r2 as one-time pads. 

	Parameters:
		full_angle_set 		list of angles as numpy floats

	Returns
		fullprob 			float, the lower bound for the guessing probability.
	"""
	# 1/(|A|^3 2^10)
	norm = 1.0/((len(full_angle_set)**3)*(2.0**10))
	fullprob = 0.0
	# Sum over the reported angles a'1:
	for aprimeone0, aprimeone1, aprimeone2 in itertools.product(full_angle_set, 
																repeat=3):
		reportone = [aprimeone0, aprimeone1, aprimeone2]
		# For a'2, we do not need to sum over all A, only those that are in the
		# set generated by the a'1. That is, we are enumerating over the set 
		# of angles for which P(a'1|a,c'1,r1,g1)*P(a'2|a,c'2,r2,g2) isn't 
		# guaranteed a priori to be zero:
		candidate_set = possible_true_angles(reportone.copy())

		# Sum over the reported angles a'2 from the reduced set:
		for aprimetwo0, aprimetwo1, aprimetwo2 in candidate_set:
			reporttwo = [aprimetwo0, aprimetwo1, aprimetwo2]

			# Sum also over the classical messages c'1, c'2:
			for c1, c2 in itertools.product(range(2), repeat=2):
				maximum = 0.0

				# Maximise over the true angles a. Again, we need only enumerate 
				# over a subset of the full angle set.
				for atrue1, atrue2, atrue3 in candidate_set:
					atrue = [atrue1,atrue2, atrue3]
					prob = 0.0
					itrset = itertools.product(range(2), repeat=8)
					# Calculate sum_{g1,g2,r1,r2} P(a'1|a,c'1,r1,g1)
					#										*P(a'2|a,c'2,r2,g2):
					for gone,gtwo,rone0,rone1,rone2,rtwo0,rtwo1,rtwo2 in itrset:
						prob += (p(reportone,atrue,rone0,rone1,rone2,c1,gone)
								 *p(reporttwo,atrue,rtwo0,rtwo1,rtwo2,c2,gtwo))
						#print("prob is {}".format(prob))
					if prob > maximum:
						maximum = prob
						print("max is {}".format(maximum))
				fullprob += maximum
	print("fullprob is {}".format(fullprob))
	fullprob *= norm
	return fullprob