"""
This script generates different quantum combs related to Grey Box MBQC for the 
graph given by
	
		G: 	V = {1,2,3,4}, E = {(1,2), (1,3), (1,4), (2,4), (3,4)}) 
		 	I = {1}, O = {3,4}

There are 15 gflows and the corresponding combs sigma1, ..., sigma15 are 
computed. Furthermore, a variety of D_MBQC comb are generated:
	
	D_{MBQC}^{g} = sum_{g} bigot_{1}^{m} sigma_{g}) ot P(g)|g><g|

	D_{MBQC}^{mp} = sum_{mp} (sum_{g~mp} 1/5 bigot_{1}^{m} sigma_{g}) 
																ot P(mp)|mp><mp| 
	
	D_{MBQC}^{two_mp} = sum_{mp=XY,XZ} (sum_{g~mp} 1/5 bigot_{1}^{m} sigma_{g}) 
																ot P(mp)|mp><mp|

	D_{MBQC}^{XY-gflows} = sum_{g~XY} bigot_{1}^{m} sigma_{g}) ot P(g)|g><g|

	D_{MBQC}^{XY,1<2} = sum_{g=1,2,4,5} bigot_{1}^{m} sigma_{g}) ot P(g)|g><g|

NOTE: the ordering above, i.e. with the state space corresponding to the random
variable last, is in order to be compatible with the function that computes
the guessing probability - see guessing_probability.py

This script primarily interacts with comb_constraints.py and 
observational_strategy.py.

This work was conducted within the Quantum Information and Computation group at 
the University of Innsbruck.
Contributors: Isaac D. Smith, Marius Krumm

This work is licensed under a Creative Commons by 4.0 license.
"""
import numpy as np
import cvxpy as cvx

from cvxpy.expressions import cvxtypes
from comb_constraints import qcomb_constraints, elementary_matrix


__all__ = ["grey_box_MBQC_meas_planes",
		   "grey_box_MBQC_two_meas_planes",
		   "grey_box_MBQC_gflows",
		   "grey_box_MBQC_XY_gflows",
		   "grey_box_MBQC_XY_1_2_partial_order",
		   "observational_meas_small"]


# # # # # # # # # # # # # # # # Global Arguments # # # # # # # # # # # # # # # #

#DTYPE = np.cfloat
DTYPE = np.float32

# # # # # # # # # # # # # # # # Helpful Matrices # # # # # # # # # # # # # # # #

X = np.matrix([[0,1], [1,0]])
Z = np.matrix([[1,0], [0,-1]])
id_2x2 = np.identity(2)
id_4x4 = np.identity(4)
id_8x8 = np.identity(8)
id_16x16 = np.identity(16)
SWAP13 = np.matrix(
	[[1,0,0,0,0,0,0,0],
	 [0,0,0,0,1,0,0,0],
	 [0,0,1,0,0,0,0,0],
	 [0,0,0,0,0,0,1,0],
	 [0,1,0,0,0,0,0,0],
	 [0,0,0,0,0,1,0,0],
	 [0,0,0,1,0,0,0,0],
	 [0,0,0,0,0,0,0,1]])
SWAP13_I_I_I = np.kron(SWAP13, id_8x8)


# # # # # # # # # # # # Generating the sigma_{MBQC}^g # # # # # # # # # # # # #

def generate_sigmas(dtype=DTYPE):
	""" Generates matrices sigma_{MBQC}^{g} for gflows g for the given graph 
	G = ({1, 2, 3, 4}, {(1,2), (1,3), (1,4), (2,4), (3,4)}) with I = {1} and 
	O = {3,4}.

	Returns:
		List of 15 np.arrays, i.e. [sigma_g1, ..., sigma_g15]
	"""

	# Gflows for (XY, XY):
	sigma_g1 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g2 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g3 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g4 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g5 = np.zeros((2**6, 2**6), dtype=dtype)

	# Gflows for (XY, XZ):
	sigma_g6 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g7 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g8 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g9 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g10 = np.zeros((2**6, 2**6), dtype=dtype)

	# Gflows for (XY, YZ):
	sigma_g11 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g12 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g13 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g14 = np.zeros((2**6, 2**6), dtype=dtype)
	sigma_g15 = np.zeros((2**6, 2**6), dtype=dtype)


	for i in range(2**10):
		binary = format(i, 'b')
		if len(binary) < 10:
			binary = (10 - len(binary))*'0' + binary
		a1 = int(binary[0])
		a2 = int(binary[2])
		a3 = int(binary[4])
		a4 = int(binary[6])
		b1 = int(binary[1])
		b2 = int(binary[3])
		b3 = int(binary[5])
		b4 = int(binary[7])
		c1 = int(binary[8])
		c2 = int(binary[9])
		
		# We generate the operators sigma by applying correction operators 
		# conditioned on c1 and c2 on computational basis states of the form
		#
		#	|a1><b1| ot |c1><c1| ot |a2><b2| ot |c2><c2| ot |a3><b3| ot |a4><b4|
		#
		# Consequently the correction operators will be of the form
		#
		#	I ot I_{c1} ot D'^{c1} ot I_{c2} ot D''^{c1,c2} ot D'''^{c1,c2}
		#
		# where the first identity is on the state space of |a1><b1| since it 
		# can receive no corrections, the operators on the state spaces 
		# associated with c1 and c2 recieve no corrections since they are input 
		# state spaces, and the operators D', D'', and D''' are products of X  
		# and Z operators with exponents given by some combination of c1 and c2 
		# determined by the specific gflow.

		comp_ket = [a1, c1, a2, c2, a3, a4]
		comp_bra = [b1, c1, b2, c2, b3, b4]
		pre_corr_state = elementary_matrix(2, comp_ket, comp_bra)
		
		# The following swapped state is required for three gflows (g3, g8, g12) 
		# which follow a different partial order.
		swapped_pre_corr_state = np.matmul(np.matmul(SWAP13_I_I_I, 
													 pre_corr_state),
										   SWAP13_I_I_I)

		graph_state_factor = (1/16)*(-1)**(a1*a2+a1*a3+a1*a4+a2*a4+a3*a4 
										   +b1*b2+b1*b3+b1*b4+b2*b4+b3*b4)

		X_c1 = np.linalg.matrix_power(X, c1)
		X_c2 = np.linalg.matrix_power(X, c2)
		X_c1c2 = np.linalg.matrix_power(X, (c1 + c2) % 2)
		Z_c1 = np.linalg.matrix_power(Z, c1)
		Z_c2 = np.linalg.matrix_power(Z, c2)
		Z_c1c2 = np.linalg.matrix_power(Z, (c1 + c2) % 2)

		# g1: 1 mapsto {2}, 2 mapsto {3,4} 
		# Corrections operator: 
		# 	I ot I_c1 ot (X^c1) ot I_c2 ot (X^c2 Z^c2) ot (X^c2 Z^c1+c2)
		corr_g1 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												  X_c1), 
										  id_2x2), 
								  np.matmul(X_c2, Z_c2)),
						  np.matmul(X_c2, Z_c1c2))
		corr_g1_trans = np.transpose(corr_g1)
		
		# g2: 1 mapsto {3}, 2 mapsto {3,4}:
		# Corrections op: 
		#	I ot I_c1 ot I ot I_c2 ot (X^c1+c2 Z^c2) ot (X^c2 Z^c1+c2)
		corr_g2 = np.kron(np.kron(id_16x16,
								  np.matmul(X_c1c2, Z_c2)),
						  np.matmul(X_c2, Z_c1c2))
		corr_g2_trans = np.transpose(corr_g2)

		# g3: 1 mapsto {3}, 2 mapsto {4} - NOTE: 2 < 1 in this case, which means 
		# that the correction operator is relative to the swapped computational
		# basis state (i.e. the ordering of the tensor factors is 2 ot 1 ot 3 ot
		# 4) and the labels c1 and c2 denote tensor factor position rather than 
		# label (i.e. c1 is for the result of measuring qubit 2 which is
		# position 1).
		# Corrections op: 
		# 	I ot I_c1 ot (Z^c1) ot I_c2 ot (Z^c1 X^c2) ot (X^c1 Z^c2)
		corr_g3 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												  Z_c1),
										  id_2x2),
								  np.matmul(Z_c1, X_c2)),
						  np.matmul(X_c1, Z_c2))
		corr_g3_trans = np.transpose(corr_g3)

		# g4: 1 mapsto {4}, 2 mapsto {3,4}
		# Corrections op: 
		# 	I ot I_c1 ot (Z^c1) ot I_c2 ot (X^c2 Z^c1+c2) ot (X^c1+c2 Z^c2)
		corr_g4 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												  Z_c1),
										  id_2x2),
								  np.matmul(X_c2, Z_c1c2)),
						  np.matmul(X_c1c2, Z_c2))
		corr_g4_trans = np.transpose(corr_g4)
		
		# g5: 1 mapsto {2,3,4}, 2 mapsto {3,4}
		# Corrections op: 
		# I ot I_c1 ot (X^c1 Z^c1) ot I_c2 ot (X^c1+c2 Z^c1+c2) ot(X^c1+c2 Z^c2)
		corr_g5 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												  np.matmul(X_c1, Z_c1)),
										  id_2x2),
								  np.matmul(X_c1c2, Z_c1c2)),
						  np.matmul(X_c1c2, Z_c2))
		corr_g5_trans = np.transpose(corr_g5)
		
		# g6: 1 mapsto {2}, 2 mapsto {2,4}
		# Corrections op: 
		# 	I ot I_c1 ot (X^c1) ot I_c2 ot (Z^c2) ot (X^c2, Z^c1+c2)
		corr_g6 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												  X_c1),
										  id_2x2),
								  Z_c2),
						  np.matmul(X_c2, Z_c1c2))
		corr_g6_trans = np.transpose(corr_g6)
		
		# g7: 1 mapsto {3}, 2 mapsto {2,4}
		# Corrections op: 
		# 	I ot I_c1 ot I ot I_c2 ot (X^c1 Z^c2) ot (X^c2 Z^c1+c2)
		corr_g7 = np.kron(np.kron(id_16x16, 
								  np.matmul(X_c1, Z_c2)),
						  np.matmul(X_c2, Z_c1c2))
		corr_g7_trans = np.transpose(corr_g7)

		# g8: 1 mapsto {3}, 2 mapsto {2,3,4} NOTE: 2 < 1 in this case, so the 
		# same convention regarding the swapped state and notation applies as in 
		# the case for g3 (see above).
		# Corrections op: 
		# 	I ot I_c1 ot (Z^c1) ot I_c2 ot (X^c1+c2 Z^c1) ot (X^c1 Z^c2)
		corr_g8 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												  Z_c1),
										  id_2x2),
								  np.matmul(X_c1c2, Z_c1)),
						  np.matmul(X_c1, Z_c2))
		corr_g8_trans = np.transpose(corr_g8)
		
		# g9: 1 mapsto {4}, 2 mapsto {2,4}
		# Corrections op: 
		# 	I ot I_c1 ot (Z^c1) ot I_c2 ot (Z^c1+c2) ot (X^c1+c2 Z^c2)
		corr_g9 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												  Z_c1),
										  id_2x2),
								  Z_c1c2),
						  np.matmul(X_c1c2, Z_c2))
		corr_g9_trans = np.transpose(corr_g9)
		
		# g10: 1 mapsto {2,3,4}, 2 mapsto {2,4}
		# Corrections op: 
		# 	I ot I_c1 ot (X^c1 Z^c1) ot I_c2 ot (X^c1 Z^c1+c2) ot (X^c1+c2 Z^c2)
		corr_g10 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												   np.matmul(X_c1, Z_c1)),
										   id_2x2),
								   np.matmul(X_c1, Z_c1c2)),
						   np.matmul(X_c1c2, Z_c2))
		corr_g10_trans = np.transpose(corr_g10)
		
		# g11: 1 mapsto {2}, 2 mapsto {2,3}
		# Corrections op: 
		# 	I ot I_c1 ot (X^c1) ot I_c2 ot (X^c2) ot (Z^c1)
		corr_g11 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												   X_c1),
										   id_2x2),
								   X_c2),
						   Z_c1)
		corr_g11_trans = np.transpose(corr_g11)

		# g12: 1 mapsto {3}, 2 mapsto {2} NOTE: 2 < 1 in this case, so the same
		# convention regarding the swapped state and notation applies as in the 
		# case for g3 (see above).
		# Corrections op: 
		# 	I ot I_c1 ot (Z^c1) ot I_c2 ot (X^c2) ot (Z^c1+c2)
		corr_g12 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												   Z_c1),
										   id_2x2),
								   X_c2),
						   Z_c1c2)
		corr_g12_trans = np.transpose(corr_g12)
		
		# g13: 1 mapsto {3}, 2 mapsto {2,3}
		# Correction op: 
		# 	I ot I_c1 ot I ot I_c2 ot (X^c1+c2) ot (Z^c1)
		corr_g13 = np.kron(np.kron(id_16x16,
								   X_c1c2),
						   Z_c1)
		corr_g13_trans = np.transpose(corr_g13)
		
		# g14: 1 mapsto {4}, 2 mapsto {2,3}
		# Correction op: 
		# 	I ot I_c1 ot (Z^c1) ot I_c2 ot (X^c2 Z^c1) ot (X^c1)
		corr_g14 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												   Z_c1),
										   id_2x2),
								   np.matmul(X_c2, Z_c1)),
						   X_c1)
		corr_g14_trans = np.transpose(corr_g14)
		
		# g15: 1 mapsto {2,3,4}, 2 mapsto {2,3}
		# Correction op: 
		# 	I ot I_c1 ot (X^c1 Z^c1) ot I_c2 ot (X^c1+c2 Z^c1) ot (X^c1)
		corr_g15 = np.kron(np.kron(np.kron(np.kron(id_4x4, 
												   np.matmul(X_c1, Z_c1)),
										   id_2x2),
								   np.matmul(X_c1c2, Z_c1)),
						   X_c1)
		corr_g15_trans = np.transpose(corr_g15)
		

		# The following applies the correction operators to the current comp 
		# basis state multiplied by the graph state phase factor for each of the
		# sigmas.
		sigma_g1 += graph_state_factor*np.matmul(np.matmul(corr_g1,
														   pre_corr_state),
												 corr_g1_trans)

		sigma_g2 += graph_state_factor*np.matmul(np.matmul(corr_g2,
														   pre_corr_state),
												 corr_g2_trans)

		sigma_g3 += graph_state_factor*np.matmul(np.matmul(corr_g3,
														swapped_pre_corr_state),
												 corr_g3_trans)

		sigma_g4 += graph_state_factor*np.matmul(np.matmul(corr_g4,
														   pre_corr_state),
												 corr_g4_trans)

		sigma_g5 += graph_state_factor*np.matmul(np.matmul(corr_g5,
														   pre_corr_state),
												 corr_g5_trans)

		sigma_g6 += graph_state_factor*np.matmul(np.matmul(corr_g6,
														   pre_corr_state),
												 corr_g6_trans)

		sigma_g7 += graph_state_factor*np.matmul(np.matmul(corr_g7,
														   pre_corr_state),
												 corr_g7_trans)
		
		sigma_g8 += graph_state_factor*np.matmul(np.matmul(corr_g8,
														swapped_pre_corr_state),
												 corr_g8_trans)

		sigma_g9 += graph_state_factor*np.matmul(np.matmul(corr_g9,
														   pre_corr_state),
												 corr_g9_trans)

		sigma_g10 += graph_state_factor*np.matmul(np.matmul(corr_g10,
															pre_corr_state),
												  corr_g10_trans)

		sigma_g11 += graph_state_factor*np.matmul(np.matmul(corr_g11,
															pre_corr_state),
												  corr_g11_trans)

		sigma_g12 += graph_state_factor*np.matmul(np.matmul(corr_g12,
														swapped_pre_corr_state),
												  corr_g12_trans)

		sigma_g13 += graph_state_factor*np.matmul(np.matmul(corr_g13,
															pre_corr_state),
												  corr_g13_trans)

		sigma_g14 += graph_state_factor*np.matmul(np.matmul(corr_g14,
															pre_corr_state),
												  corr_g14_trans)

		sigma_g15 += graph_state_factor*np.matmul(np.matmul(corr_g15,
															pre_corr_state),
												  corr_g15_trans)

	return [sigma_g1,sigma_g2,sigma_g3,sigma_g4,sigma_g5,
		    sigma_g6,sigma_g7,sigma_g8,sigma_g9,sigma_g10,
		    sigma_g11,sigma_g12,sigma_g13,sigma_g14,sigma_g15]


# # # # # # # # # # # # # # Problem Specific Combs # # # # # # # # # # # # # #

def grey_box_MBQC_meas_planes(rounds=1, dtype=DTYPE):
	""" Computes the operator

		D_MBQC = sum_{mp} (sum_{g~mp} (1/5) bigot_{j=1}^{m} sigma_{MBQC}^{g}) 
												ot P(mp)ket{mp}bra{mp}

	where mp stands for measurement plane, m is the number of rounds and 
	g~mp denotes the gflows associated to the given measurement plane.

	Inputs:
		rounds 		Int indicating the number of times sigma^{g} is tensored 
					with itself.

	Returns:
		np.array which is square of size 3*((1*2*2*2*2*4*1)**rounds)
	"""
	sigmas=generate_sigmas(dtype)
	rounds_dim = (1*2*2*2*2*4*1)**rounds
	sigma_MBQC_dim = 3*rounds_dim

	sigma_MBQC = np.zeros((sigma_MBQC_dim, sigma_MBQC_dim),dtype=dtype)
	P_mp = 1/3 # We assume uniform P(mp)
	for mp in range(3):
		intermediate_sigma = P_mp*elementary_matrix(3, [mp])
		mp_comb = np.zeros((rounds_dim, rounds_dim), dtype=dtype)
		for g in range(5):
			g_round_comb = (1/5)*np.identity(1)
			for _ in range(rounds):
				g_round_comb = np.kron(sigmas[mp*5 + g], g_round_comb)

			mp_comb += g_round_comb

		sigma_MBQC += np.kron(mp_comb, intermediate_sigma)

	return sigma_MBQC



def grey_box_MBQC_two_meas_planes(rounds=1, dtype=DTYPE):
	""" Computes the operator

		D_MBQC = sum_{mp=Xy,XZ}(sum_{g~mp}(1/5 bigot_{j=1}^{m} sigma_{MBQC}^{g}) 
												ot P(mp)ket{mp}bra{mp}

	where mp stands for measurement plane (restricted to just two choices), 
	m is the number of rounds and g~mp denotes the gflows associated to the
	given measurement plane.

	Inputs:
		rounds 		Int indicating the number of times sigma^{g} is tensored 
					with itself.

	Returns:
		np.array which is square of size 2*((1*2*2*2*2*4*1)**rounds)
	"""
	sigmas=generate_sigmas(dtype)
	rounds_dim = (1*2*2*2*2*4*1)**rounds
	sigma_MBQC_dim = 2*rounds_dim

	sigma_MBQC = np.zeros((sigma_MBQC_dim, sigma_MBQC_dim),dtype=dtype)
	P_mp = 1/2 # We assume uniform P(mp)
	for mp in range(2):
		intermediate_sigma = P_mp*elementary_matrix(2, [mp])
		mp_comb = np.zeros((rounds_dim, rounds_dim), dtype=dtype)
		for g in range(5):
			g_round_comb = (1/5)*np.identity(1)
			for _ in range(rounds):
				g_round_comb = np.kron(sigmas[mp*5 + g], g_round_comb)

			mp_comb += g_round_comb

		sigma_MBQC += np.kron(mp_comb, intermediate_sigma)

	return sigma_MBQC




def grey_box_MBQC_gflows(rounds=1, dtype=DTYPE):
	""" Computes the operator

		D_MBQC = sum_{g} bigot_{j=1}^{m} sigma_{MBQC}^{g}) ot P(g)ket{g}bra{g}

	where m is the number of rounds and the sum is over all gflows.

	Inputs:
		rounds 		Int indicating the number of times sigma^{g} is tensored 
					with itself.

	Returns:
		np.array which is square of size 15*((1*2*2*2*2*4*1)**rounds)
	"""
	sigmas=generate_sigmas(dtype)
	rounds_dim = (1*2*2*2*2*4*1)**rounds
	sigma_MBQC_dim = 15*rounds_dim

	sigma_MBQC = np.zeros((sigma_MBQC_dim, sigma_MBQC_dim),dtype=dtype)
	P_g = 1/15 # We assume uniform P(g)

	for g in range(15):
		intermediate_sigma = P_g*elementary_matrix(15, [g])
		g_round_comb = np.identity(1)
		for _ in range(rounds):
			g_round_comb = np.kron(g_round_comb, sigmas[g])
		sigma_MBQC += np.kron(g_round_comb, intermediate_sigma)

	return sigma_MBQC



def grey_box_MBQC_XY_gflows(rounds=1, dtype=DTYPE):
	""" Computes the operator

		D_MBQC = sum_{g~XY} bigot_{j=1}^{m} sigma_{MBQC}^{g}) 
														ot P(g)ket{g}bra{g}

	where m is the number of rounds and the sum is over all XY-plane gflows.

	Inputs:
		rounds 		Int indicating the number of times sigma^{g} is tensored 
					with itself.

	Returns:
		np.array which is square of size 5*((1*2*2*2*2*4*1)**rounds)
	"""
	sigmas = generate_sigmas(dtype)[0:5]
	print("Len of sigmas is {}".format(len(sigmas)))
	rounds_dim = (1*2*2*2*2*4*1)**rounds
	sigma_MBQC_dim = 5*rounds_dim

	sigma_MBQC = np.zeros((sigma_MBQC_dim, sigma_MBQC_dim),dtype=dtype)
	P_g = 1/5 # We assume uniform P(g)

	for g in range(5):
		intermediate_sigma = P_g*elementary_matrix(5, [g])
		g_round_comb = np.identity(1)
		for _ in range(rounds):
			g_round_comb = np.kron(g_round_comb, sigmas[g])
		sigma_MBQC += np.kron(g_round_comb, intermediate_sigma)

	return sigma_MBQC



def grey_box_MBQC_XY_1_2_partial_order(rounds=1, dtype=DTYPE):
	""" Computes the operator

		D_MBQC = sum_{g = 1,2,4,5} bigot_{j=1}^{m} sigma_{MBQC}^{g}) 
														ot P(g)ket{g}bra{g}

	where m is the number of rounds.

	Inputs:
		rounds 		Int indicating the number of times sigma^{g} is tensored 
					with itself.

	Returns:
		np.array which is square of size 4*((1*2*2*2*2*4*1)**rounds)
	"""
	sigmas = generate_sigmas(dtype)[0:5]
	sigmas.pop(2)
	print("length of sigmas is {}".format(len(sigmas)))
	rounds_dim = (1*2*2*2*2*4*1)**rounds
	sigma_MBQC_dim = 4*rounds_dim

	sigma_MBQC = np.zeros((sigma_MBQC_dim, sigma_MBQC_dim),dtype=dtype)
	P_g = 1/4 # We assume uniform P(g)

	for g in range(4):
		intermediate_sigma = P_g*elementary_matrix(4, [g])
		g_round_comb = np.identity(1)
		for _ in range(rounds):
			g_round_comb = np.kron(g_round_comb, sigmas[g])
		sigma_MBQC += np.kron(g_round_comb, intermediate_sigma)

	return sigma_MBQC



# # # # # # # # # # # # # # # Auxilliary Functions # # # # # # # # # # # # # # #

def observational_meas_small(theta1, phi1, theta2a, phi2a, theta2b = 0,  
						   	 phi2b = 0, adaptive = False):
	""" Generates a dual comb to the combs D above, representing projective 
	measurements on the first two qubits. The classical outcomes are ONLY sent 
	back into the gflow-comb.

	If adaptive == True:
		A 0-outcome on the first qubit will lead to measurement settings 
		theta2a, phi2a, on qubit 2;
		A 1-outcome will lead to measurement settings theta2b, phi2b.
	Else:
		The second qubit is always measured with theta2a, phi2a.

	Pure states are parametrized as: 
	cos(theta/2) |0> + e^(i phi) sin(theta/2) | 1 >

	For the system order, we make the following convention:
	qubit1, outcome1, qubit2, outcome2

	Returns the associated (probabilistic) comb for each outcome combination.
	"""

	# Create the pure states for the 0 outcomes:
	ket1 = np.array([[np.cos(theta1/2.0)],
					 [np.exp(1.0j * phi1 )*np.sin(theta1/2.0)]],
					dtype = np.cfloat)
	ket2a = np.array([[np.cos(theta2a/2.0)],
					  [np.exp(1.0j * phi2a )*np.sin(theta2a/2.0)]], 
					 dtype = np.cfloat)
	ket2b = np.array([[np.cos(theta2b/2.0)],
					  [np.exp(1.0j * phi2b )*np.sin(theta2b/2.0)]], 
					 dtype = np.cfloat)
	
	# Write the states as density matrices:
	pure1 = ket1 @ (ket1.copy().conj().transpose())
	pure2a = ket2a @ (ket2a.copy().conj().transpose())
	pure2b = ket2b @ (ket2b.copy().conj().transpose())
	
	# The projectors for outcomes 1:
	opposite1 = id_2x2 - pure1
	opposite2a = id_2x2 - pure2a
	opposite2b = id_2x2 - pure2b

	# Apply the partial transpositions:
	pure1T = pure1.transpose()
	pure2aT = pure2a.transpose()
	pure2bT = pure2b.transpose()
	opposite1T = opposite1.transpose()
	opposite2aT = opposite2a.transpose()
	opposite2bT = opposite2b.transpose()

	# This will make it easier to access the operators in loops:
	qubit1Meas = [pure1T, opposite1T]
	qubit2Meas = [pure2aT, opposite2aT , pure2bT, opposite2bT]

	# Represent the classical outcomes as operators:
	zeroOutcome = np.array([[1,0],[0,0]], dtype = np.cfloat)
	oneOutcome = np.array([[0,0],[0,1]], dtype = np.cfloat)
	outcomes = [zeroOutcome, oneOutcome]

	# Construct the comb, respecting the order convention:
	# qubit1, outcome1, qubit2, outcome2
	result = np.zeros(shape = (2,2,2**4, 2**4), dtype = np.cfloat)
	for out1 in range(2):
		for out2 in range(2):
			change = np.kron(qubit1Meas[out1] , outcomes[out1])
			if adaptive:
				change = np.kron(change, qubit2Meas[2*out1+out2])
			else:
				change = np.kron(change, qubit2Meas[out2])
			change = np.kron(change, outcomes[out2])
			result[out1,out2 , : , :] = change.copy()
	return result.copy()


grey_box_MBQC_gflows()