"""

This script generates D_{client}^{(m)} for |A| angles and m rounds for the  
minimal example given by:
	
		G: 	V = {1, 2, 3}, E = {(1,2), (2,3), (1,3)}
		 	O = {2,3}  - Note that the input set I is irrelevant here

There are two gflows for this example, g and g_prime (corresponding to g1 amd g2 
in the text).

NOTE: this script creates the operators D_client as 1D objects which correspond
to the diagonal of the matrix version. 
"""

import cvxpy as cvx
import numpy as np
from comb_constraints import elementary_matrix


# To DO:
#	> review header

def g_sigma_bqc(angles, one_time_pads, total_num_angles):
	""" Computes sigma_{BQC}^{alpha, g, r} where g: 1 mapsto {2} is a gflow on
	the graph G defined by V = {1, 2, 3}, E = {(1,2), (1,3), (2,3)} with 
	total order on vertices. The angles (alpha_{1}, alpha_{2}, alpha_{3}) are
	adapted according to:

		alpha'_{1} = alpha_{1} + r_{1}pi mod 2pi
		alpha'_{2} = (-1)^{r_{1} oplus c'_{1}}alpha_{2} + r_{2}pi mod 2pi
		alpha'_{3} = alpha_{3} + (r_{3} oplus r_{1} oplus c'_{1})pi mod 2pi

	Inputs:
		angles 				Length three list of integers from 0 to 
							total_num_angles-1;
		one_time_pads		Length three list of binary;
		total_num_angles 	Int indicating the total number of allowed angles.

	Returns:
		np.array of dimension (total_num_angles**3)*(2**3) - the array is square
	"""
	if len(angles) != 3 or len(one_time_pads) != 3:
		raise ValueError("Function only defined for 3 angles/one time pads.")
	if not all(0 <= a < total_num_angles for a in angles):
		raise ValueError("Some angle label is invalid.")

	sigma_dim = (total_num_angles**3)*(2**3)
	sigma = np.zeros((1,sigma_dim), dtype=np.float32)

	for c1 in range(2):
		for c2 in range(2):
			for c3 in range(2):

				# Implementing the corrections based on gflow g:
				alpha_one_prime = (
					(angles[0] + 							# alpha_{1}
					one_time_pads[0]*total_num_angles/2)	# + r_{1}pi
					%total_num_angles						# mod 2pi
				)
				alpha_one_prime = int(alpha_one_prime)

				if (one_time_pads[0] + c1)%2 == 0:
					alpha_two_prime = (
						(angles[1] 								# alpha_{2}
						+ one_time_pads[1]*total_num_angles/2)	# + r_{2}pi
						%total_num_angles						# mod 2pi
						)
				else:
					alpha_two_prime = (
						((total_num_angles - angles[1] - 1)		# -alpha_{2}
						+ one_time_pads[1]*total_num_angles/2)	# + r_{2}pi
						%total_num_angles						# mod 2pi
						)

				alpha_two_prime = int(alpha_two_prime)

				alpha_three_prime = (
					(angles[2] + 							# alpha_{3}
					(one_time_pads[0] + one_time_pads[2] 	# + (r_{1} + r_{3}
					+ c1)*total_num_angles/2)				# + c_{1})pi
					%total_num_angles						# mod 2pi
				)
				alpha_three_prime = int(alpha_three_prime)

				a_one_prime = elementary_matrix(total_num_angles,
												[alpha_one_prime],
												as_list=True)
				a_two_prime = elementary_matrix(total_num_angles,
												[alpha_two_prime],
												as_list=True)
				a_three_prime = elementary_matrix(total_num_angles,
												  [alpha_three_prime],
												  as_list=True)
				c_1 = elementary_matrix(2, [c1], as_list=True)
				c_2 = elementary_matrix(2, [c2], as_list=True)
				c_3 = elementary_matrix(2, [c3], as_list=True)

				A1 = np.kron(a_one_prime, c_1)
				A2 = np.kron(np.kron(A1, a_two_prime), c_2)
				sigma += np.kron(np.kron(A2, a_three_prime), c_3)

	return sigma




def g_prime_sigma_bqc(angles, one_time_pads, total_num_angles):
	""" Computes sigma_{BQC}^{alpha, g', r} where g': 1 mapsto {3} is a gflow on
	the graph G defined by V = {1, 2, 3}, E = {(1,2), (1,3), (2,3)} with 
	total order on vertices. The angles (alpha_{1}, alpha_{2}, alpha_{3}) are
	adapted according to:

		alpha'_{1} = alpha_{1} + r_{1}pi mod 2pi
		alpha'_{2} = alpha_{2} + (r_{2} oplus r_{1} oplus c'_{1})pi mod 2pi
		alpha'_{3} = (-1)^{r_{1} oplus c'_{1}}alpha_{3} + r_{3}pi mod 2pi

	Inputs:
		angles 				Length three list of integers from 0 to 
							total_num_angles-1;
		one_time_pads		Length three list of binary;
		total_num_angles 	Int indicating the total number of allowed angles.

	Returns:
		np.array of dimension (total_num_angles**3)*(2**3) - the array is square
	"""
	if len(angles) != 3 or len(one_time_pads) != 3:
		raise ValueError("Function only defined for 3 angles/one time pads.")
	if not all(0 <= a < total_num_angles for a in angles):
		raise ValueError("Some angle label is invalid.")

	sigma_dim = (total_num_angles**3)*(2**3)
	sigma = np.zeros((1,sigma_dim), dtype=np.float32)

	for c1 in range(2):
		for c2 in range(2):
			for c3 in range(2):

				# Implementing the corrections based on gflow g':
				alpha_one_prime = (
					(angles[0] + 							# alpha_{1}
					one_time_pads[0]*total_num_angles/2)	# + r_{1}pi
					%total_num_angles						# mod 2pi
				)
				alpha_one_prime = int(alpha_one_prime)

				alpha_two_prime = (
					(angles[1] + 							# alpha_{2}
					(one_time_pads[0] + one_time_pads[1] 	# + (r_{1} + r_{2}
					+ c1)*total_num_angles/2)				# + c_{1})pi
					%total_num_angles						# mod 2pi
				)
				alpha_two_prime = int(alpha_two_prime)

				if (one_time_pads[0] + c1)%2 == 0:
					alpha_three_prime = (
						(angles[2] 								# alpha_{3}
						+ one_time_pads[2]*total_num_angles/2)	# + r_{3}pi
						%total_num_angles						# mod 2pi
						)
				else:
					alpha_three_prime = (
						((total_num_angles - angles[2] - 1)		# -alpha_{3}
						+ one_time_pads[2]*total_num_angles/2)	# + r_{3}pi
						%total_num_angles						# mod 2pi
						)

				alpha_three_prime = int(alpha_three_prime)

				
				a_one_prime = elementary_matrix(total_num_angles,
												[alpha_one_prime],
												as_list=True)
				a_two_prime = elementary_matrix(total_num_angles,
												[alpha_two_prime],
												as_list=True)
				a_three_prime = elementary_matrix(total_num_angles,
												  [alpha_three_prime],
												  as_list=True)

				c_1 = elementary_matrix(2, [c1], as_list=True)
				c_2 = elementary_matrix(2, [c2], as_list=True)
				c_3 = elementary_matrix(2, [c3], as_list=True)

				A1 = np.kron(a_one_prime, c_1)
				A2 = np.kron(np.kron(A1, a_two_prime), c_2)
				sigma += np.kron(np.kron(A2, a_three_prime), c_3)

	return sigma




def sum_over_gflows_and_OTPs(angles, total_num_angles):
	""" Computes the intermediate operator

		(1/(2*2**3))sum_{r=(r1,r2,r3)}sigma_{alpha,g,r}+sigma^{alpha,g',r}
	
	for use in computing D_{client}^{(m)}.

	Inputs:
		angles				Length three list of integers from 0 to 
							total_num_angles-1;
		total_num_angles 	Int indicating the total number of allowed angles.

	Returns:
		np.array of dimension (total_num_angles**3)*(2**3) - the array is square
	"""

	sigma_dim = (total_num_angles**3)*(2**3)
	sigma = np.zeros((1,sigma_dim), dtype=np.float32)

	for r1 in range(2):
		for r2 in range(2):
			for r3 in range(2):
				sigma += g_sigma_bqc(angles, [r1,r2,r3], total_num_angles)
				sigma += g_prime_sigma_bqc(angles, [r1,r2,r3], total_num_angles)

	return (1/(2**4))*sigma


def m_round_client(rounds, total_num_angles):
	""" Computes the m round operator for the client:

		D_{client}^{(m)} = sum_{alpha} P(alpha)ket{alpha}bra{alpha} otimes
							bigotimes_{j=1}^{m}((1/(2*2**3))sum_{r=(r1,r2,r3)}
							sigma_{alpha,g,r} + sigma^{alpha,g',r})
	
	The probability distribution is taken to be uniform.

	NOTE: The factor P(alpha)ket{alpha}bra{alpha} is implemented as the 
	right-hand-most factor of the tensor product in order to be compatible with
	the function qcomb_constraint() in qcomb_constraint.py and 
	conditional_min_entropy() in conditional_min_entropy.py.

	Inputs:
		rounds				Int indicating the number of rounds, i.e. the number
							of tensor factors of sum_over_gflows_and_OTPs;
		total_num_angles 	Int indicating the total number of allowed angles.

	Returns:
		np.array of dimension: 
			(total_num_angles**3)*((total_num_angles**3)*(8))**rounds
	"""

	P_alpha = 1/(total_num_angles**3)

	dim = (total_num_angles**3)*((total_num_angles**3)*(8))**rounds
	D_client = np.zeros((1,dim), dtype=np.float32)

	for a1 in range(total_num_angles):
		for a2 in range(total_num_angles):
			for a3 in range(total_num_angles):
				angles = [a1,a2,a3]
				round_comb = P_alpha*elementary_matrix(total_num_angles, 
													   angles,
													   as_list=True)

				for _ in range(rounds):
					round_comb = np.kron(sum_over_gflows_and_OTPs(angles, 
															  total_num_angles),
										 round_comb)
				D_client += round_comb
	return D_client
