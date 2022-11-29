"""

"""

import cvxpy as cvx
import numpy as np

from comb_constraints import qcomb_constraints, list_qcomb_constraints

# TO DO:
#	> write header


# # # # # # # # # # Quantum Comb Guessing Probability # # # # # # # # # # # #  

def guessing_probability(op, dims, solver=cvx.SCS):
	""" Computes the guessing probability of a quantum comb in the set of combs
	specified by the list dims which is of the form 

					[in, out, in, out, ..., in, var]

	with var (for variable) replacing the last output dimension since we 
	consider operators of the form

				op = sum_{v in var} sigma_{v} ot P(v)|v><v|

	where each sigma_{v} is a quantum comb.

	Inputs:
		op 		np.array: 2D of size prod(dims) by prod(dims);
		dims 	List of ints, following the alternating in-out style above;
		solver 	cvx solver with the Splitting Conic Solver as default.
	"""
	if len(op.shape) != 2 or op.shape[0] != op.shape[1]:
		raise ValueError("Operator is not square.")
	# The specified dimensions and total dimension of the operator must match.
	if op.shape[0] != np.prod(dims):
		raise ValueError("Total op dim and input-output dims don't match.")
	if len(dims)%2 != 0 or len(dims) < 4:
		raise ValueError("An even number (>=) of spaces need to be specified.")
	var_dim = dims.pop()
	last_in_dim = dims.pop()

	n = np.prod(dims)
	in_dims = [dims[2*i] for i in range(int(len(dims)/2))]
	prod_in_dims = np.prod(in_dims)
	
	# The variable the to be minimised over:
	X = cvx.Variable((n,n), complex=True)

	constraints = []

	# Generate the constraints for X to be an unnormalised comb (the default
	# for the qcomb_constraints function is to generate constraints for 
	# an unnormalised comb):
	constraints += qcomb_constraints(X, dims)

	# Add constraint: X ot Id >> op:
	constraints += [cvx.kron(X,np.identity(var_dim*last_in_dim)) >> op]

	# Minimise the trace over X, however since X is a complex valued 
	# variable, the real part must be taken to avoid an error with the trace
	# function (X has real diagonal anyway):
	prob = cvx.Problem(cvx.Minimize((1/prod_in_dims)*cvx.trace(cvx.real(X))), 
					   constraints)
	prob.solve(solver=solver,verbose=True)

	return [prob.value, X.value, prob.status, prob.size_metrics, 
			prob.solver_stats]




# # # # # # # # # # # Classical Comb Guessin Probability # # # # # # # # # # # 


def list_guessing_probability(op, dims, solver=cvx.SCS):
	"""Computes the guessing probability of a "classical" comb in the set of
	combs specified by the list dims which is of the form 

					[in, out, in, out, ..., in, var]

	with var (for variable) replacing the last output dimension since we 
	consider operators of the form

				op = sum_{v in var} sigma_{v} ot P(v)|v><v|

	where each sigma_{v} is diagonal. 

	Inputs:
		op 		np.array - 1D of size prod(dims);
		dims 	List of ints, following the alternating in-out style above;
		solver 	cvx solver with the Splitting Conic Solver as default.
	"""
	if op.shape != (1, np.prod(dims)):
		raise ValueError("Operator must be of shape (1, prod(dims)).")

	if len(dims)%2 != 0 or len(dims) < 4:
		raise ValueError("An even number (>=) of spaces need to be specified.")
	var_dim = dims.pop()
	last_in_dim = dims.pop()

	n = np.prod(dims)
	in_dims = [dims[2*i] for i in range(int(len(dims)/2))]

	X = cvx.Variable((1,n))

	constraints = []
	constraints += list_qcomb_constraints(X, dims)

	# Add constraint equivalent to X ot Id >> op, i.e.
	# ([[1,1,...,1]] ot X) - op >= [0,0,...,0]
	for x in range(op.shape[1]):
		constraints += [cvx.kron(X, np.ones((1,var_dim*last_in_dim),
										    dtype=np.float32))[0][x] 
						- op[0][x] >= 0]

	prob = cvx.Problem(cvx.Minimize((1/np.prod(in_dims))*cvx.sum(X)), 
					   constraints)
	prob.solve(solver=solver,verbose=True)

	return [prob.value, X.value, prob.status, prob.size_metrics, 
			prob.solver_stats]