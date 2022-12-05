"""
This script generates the combs related to calibrating the measurement planes 
for a grey box MBQC device via learning the discrepancy of the angle between
the X-axis internal to the device and that of the measurement instruments. The
combs generated here are of the form:

	D_{calibr}^{(m)} = sum_{theta} 1/num_angles |theta><theta| 
													ot sigma_theta * rho_G_theta

where m is the number of rounds, theta is the angle discrepancy that can take
num_angles different values, sigma_theta is the operator enacting the 
corrections in the rotated X basis and rho_G_theta is the graph state in the 
same basis (c.f. Section 5.5 of the manuscript for more details). The example
considered here is the linear 3-cluster:

				G: 	V = {1,2,3}, E = {(1,2), (2,3)}) 
		 			I = {}, O = {3}

This script contains three functions:

	> rho_G_theta 						generates the rotated graph state, i.e.
										the graph state prepared on qubits in 
										the state R_{Z}(theta)|+>;
	> sigma_theta_link_rho_G_theta 		generates the operator arising from
										sigma_theta * rho_G_theta where * is the
										link product and sigma_theta enacts the
										rotated corrections;
	> D_calibr 							generates the comb D_{client}^{(m)} for
										the input number of angles and number of
										rounds;

This script primarily interacts with comb_constraints.py

This work was conducted within the Quantum Information and Computation group at 
the University of Innsbruck.
Contributors:  Marius Krumm, Isaac D. Smith

This work is licensed under a Creative Commons by 4.0 license.
"""
import cvxpy as cvx
import numpy as np
import itertools

from cvxpy.expressions import cvxtypes
from comb_constraints import elementary_matrix


# # # # # # # # # # # # # # # # Helpful Matrices # # # # # # # # # # # # # # # #

Z = np.matrix([[1,0], [0,-1]])
id2 = np.diag([1.0 + 0.0j, 1.0 +0.0j])

# # # # # # # # # # # # # # Problem Specific Combs # # # # # # # # # # # # # #

def rho_G_theta(theta = np.pi/2):
	""" Generates the rotated graph state, i.e.

		|G_theta> := Prod_{(i,j)} CZ_{i,j} R_{Z}(theta)^{ot n}|+>^{ot n}

	for the linear graph G on three qubits. 

	Input:
		theta 		float, angle between 0 and 2pi;

	Returns:
		np.array, corresponding to rho_G_theta = |G_theta><G_theta|.
	"""
	thetaPlus =  np.matrix([[0.5+0.0j,  0.5*np.exp(1.0j * theta)], 
							[0.5*np.exp(-1.0j * theta) , 0.5+0.0j]])

	uncorr = np.kron(np.kron(thetaPlus, thetaPlus), thetaPlus)
	CZ = np.diag([1.0 + 0.0j , 1.0 + 0.0j , 1.0 +0.0j, -1.0+0.0j])
	CZ12 = np.kron(CZ, id2)
	CZ23 = np.kron(id2, CZ)
	state = CZ23 @ CZ12 @ uncorr @ CZ12 @ CZ23
	
	# The transpose is applied in anticipation of the use of the link product
	transposed = state.getT()
	return transposed.copy()

def sigma_theta_link_rho_G_theta(graph_state, theta = np.pi/2):
	""" Generates the operator that results from applying the link product
	between sigma_theta and rho_G_theta, where rho_G_theta is the rotated
	graph state (as above) and where sigma_theta is the operator that applies 
	condition Z and rotated-X operators based on measurement outcomes (see
	manuscript, Section 5.5 for more details). The rotated-X operators are of
	the form
					R_{Z}(theta) X R_{Z}(theta)^{dagger}
	
	The output operator has dimensions [2,2,2,2,2] following the ordering, from
	left to right, of corrected graph state qubit 1, measurement outcome qubit
	1, corrected graph state qubit 2, measurement outcome 2 and 
	corrected graph state qubit 3 (there is no measurement outcome for the final
	qubit).

	Inputs:
		graph_state		np.array, representing the rotated graph state;
		theta 			float, an angle between 0 and 2pi;

	Returns
		np.array, representing sigma_theta * graph_state (* denotes the link
		product).
	"""
	X_theta = np.matrix([[ 0.0 + 0.0j, np.exp(1.0j * theta)], 
						 [np.exp(-1.0j * theta), 0.0 + 0.0j]])
	corr1 = np.kron(np.eye(2**5 , dtype = np.cfloat) , X_theta)
	corr1 = np.kron(corr1, id2) 
	corr1 = np.kron(corr1, Z)

	corr2 = np.kron(np.eye(2**7 , dtype = np.cfloat) , X_theta)

	Choi = np.zeros((2**8, 2**8), dtype = np.cfloat)

	for a1, a2, a3, b1, b2, b3, c1, c2 in itertools.product(range(2),repeat=8):
		rowIndices =[a1 , a2, a3 , a1, c1 , a2, c2, a3]
		columnIndices =[b1 , b2, b3 , b1, c1 , b2, c2, b3]
		elem = elementary_matrix(2,rowIndices,columnIndices,dtype = np.cfloat)

		corr1_c1 = np.linalg.matrix_power(corr1, c1)
		corr2_c2 = np.linalg.matrix_power(corr2, c2)

		inc = corr2_c2 @ corr1_c1 @ elem @ (corr1_c1.getH()) @ (corr2_c2.getH())
		
		Choi = Choi + inc.copy()
	
	extended_input = np.kron(graph_state, np.eye(2**5 , dtype = np.cfloat))

	prod = extended_input @ Choi
	prod_cvx = cvxtypes.constant()(prod)

	linked_cvx0 = cvx.partial_trace(prod_cvx , [2,2,2,2,2,2,2,2], 0)
	linked_cvx01 = cvx.partial_trace(linked_cvx0 , [2,2,2,2,2,2,2], 0)	
	linked_cvx012 = cvx.partial_trace(linked_cvx01 , [2,2,2,2,2,2], 0)
	linked = np.matrix(linked_cvx012.value)
	return linked.copy()


def D_calibr(rounds = 1, num_angles = 4):
	""" Generates the comb 

		D_{calibr}^{(m)} = sum_{theta} 1/num_angles |theta><theta| 
													ot sigma_theta * rho_G_theta

	for m the number of rounds and for an angle set of size num_angles (the 
	asterisk denotes the link product).

	Inputs:
		rounds 		Int, the number of rounds of interaction
		num_angles 	Int, the number of values that the angle discrepancy between
					X-axes can take;

	Returns:
		np.array, the comb D_{calibr}^{(m)}
	"""
	multi_comb = np.zeros(((2**(5*rounds))*num_angles, 
						   (2**(5*rounds))*num_angles), 
						  dtype = np.cfloat)

	for theta_index in range(num_angles):
		theta = theta_index * 2.0*np.pi/num_angles
		corrected = sigma_theta_link_rho_G_theta(rho_G_theta(theta), theta)

		comb1 = corrected.copy()
		multi_comb_theta = comb1.copy()

		for _ in range(rounds - 1):
			multi_comb_theta = np.kron(multi_comb_theta, comb1.copy())

		with_theta_index = (1.0/num_angles)*np.kron(multi_comb_theta, 
											elementary_matrix(num_angles, 
															  [theta_index], 
															  [theta_index],
															 dtype = np.cfloat))
		multi_comb = multi_comb + with_theta_index.copy()

	return multi_comb.copy()
