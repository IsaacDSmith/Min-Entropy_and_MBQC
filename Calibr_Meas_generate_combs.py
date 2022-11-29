"""
This script generates different quantum combs related to QRF grey box MBQC.
We assume that we know the Z-axis, but X-axis and Y-axis are unknown. 

We consider two different graph states (for now without input).	
	The linear 3-cluster:
		G: 	V = {1,2,3}, E = {(1,2), (2,3)}) 
		 	I = {}, O = {3}

	The triangle:
		G: 	V = {1,2,3}, E = {(1,2), (1,3), (2,3)}) 
		 	I = {}, O = {2,3}
"""

import cvxpy as cvx
from cvxpy.expressions import cvxtypes
import numpy as np
import itertools

from comb_constraints import elementary_matrix

# TO DO:
#	> Write header
#	> Integrate code better with the text of the paper


# # # # # # # # # # Helpful Matrices # # # # # # # # # # 

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
id2 = np.diag([1.0 + 0.0j , 1.0 +0.0j])


"""
Comb definition for a linear cluster state.
"""


"""
Graph state definition for a linear cluster state.
The global e^{i phi_0} drops out as soon as we write the density matrix.
"""
def graph_state_delta_linear(delta = np.pi/2):
	deltaPlus =  np.matrix([[0.5+0.0j,  0.5*np.exp(1.0j * delta)], 
							[0.5*np.exp(-1.0j * delta) , 0.5+0.0j]])

	uncorr = np.kron(np.kron(deltaPlus, deltaPlus), deltaPlus)
	CZ = np.diag([1.0 + 0.0j , 1.0 + 0.0j , 1.0 +0.0j, -1.0+0.0j])
	CZ12 = np.kron(CZ, id2)
	CZ23 = np.kron(id2, CZ)
	state = CZ23 @ CZ12 @ uncorr @ CZ12 @ CZ23
	
	# Actually, we only need the transpose of the state
	transposed = state.getT()
	return transposed.copy()


"""
The comb implementing corrections, contracted with the input state.
Convention on Tensor product order:
The first three qubits are the input state.
Then we use the following order:
output qubit 1, message 1, output qubit 2, message 2, output qubit 3 (no 
classical message)
"""

def correct(inputstate , delta = np.pi/2):
	X_delta = np.matrix([[ 0.0 + 0.0j, np.exp(1.0j * delta)], 
						 [np.exp(- 1.0j * delta), 0.0 + 0.0j]])
	corr1 = np.kron(np.eye(2**5 , dtype = np.cfloat) , X_delta)
	corr1 = np.kron( corr1, id2 ) 
	corr1 = np.kron( corr1, Z)

	corr2 = np.kron(np.eye(2**7 , dtype = np.cfloat) , X_delta)

	Choi = np.zeros( (2**8, 2**8), dtype = np.cfloat)

	for a1, a2, a3, b1, b2, b3, c1, c2 in itertools.product(range(2),repeat=8):
		rowIndices =[a1 , a2, a3 , a1, c1 , a2, c2, a3]
		columnIndices =[b1 , b2, b3 , b1, c1 , b2, c2, b3]
		elem = elementary_matrix(2,rowIndices,columnIndices,dtype = np.cfloat)

		corr1_c1 = np.linalg.matrix_power(corr1, c1)
		corr2_c2 = np.linalg.matrix_power(corr2, c2)

		inc = corr2_c2 @ corr1_c1 @ elem @ (corr1_c1.getH()) @ (corr2_c2.getH())
		
		Choi = Choi + inc.copy()
	
	extended_input = np.kron(inputstate, np.eye(2**5 , dtype = np.cfloat))

	prod = extended_input @ Choi
	prod_cvx = cvxtypes.constant()(prod)

	linked_cvx0 = cvx.partial_trace(prod_cvx , [2,2,2,2,2,2,2,2], 0)
	linked_cvx01 = cvx.partial_trace(linked_cvx0 , [2,2,2,2,2,2,2], 0)	
	linked_cvx012 = cvx.partial_trace(linked_cvx01 , [2,2,2,2,2,2], 0)
	linked = np.matrix(linked_cvx012.value)
	return linked.copy()



"""
The finished comb.
The 4level system for the delta-index is the right-most Hilbert space. 
"""

def what_is_X_linear_comb( rounds = 1, num_angles = 4):
	multi_comb = np.zeros(((2**(5*rounds))*num_angles, 
						   (2**(5*rounds))*num_angles), 
						  dtype = np.cfloat)

	for delta_index in range(num_angles):
		delta = delta_index * 2.0*np.pi/num_angles
		print("Delta is "+str(delta))
		corrected = correct(graph_state_delta_linear(delta) , delta)

		comb1 = corrected.copy()
		multi_comb_delta = comb1.copy()

		for _ in range(rounds - 1):
			multi_comb_delta = np.kron(multi_comb_delta, comb1.copy())

		with_delta_index = (1.0/num_angles)*np.kron(multi_comb_delta, 
											elementary_matrix(num_angles, 
															  [delta_index], 
															  [delta_index],
															 dtype = np.cfloat))
		multi_comb = multi_comb + with_delta_index.copy()

	

	return multi_comb.copy()


