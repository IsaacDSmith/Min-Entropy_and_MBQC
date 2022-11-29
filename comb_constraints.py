"""

"""


import cvxpy as cvx
import numpy as np
import scipy as sp

from cvxpy.expressions import cvxtypes
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackError

from list_partial_trace import list_partial_trace


# TO DO: 
#	> write header
#	> delete old elementary matrix function (at the end) after testing the new one


# # # # # # # # # # # # # # # Helpful Functions # # # # # # # # # # # # # # #

def elementary_matrix(base, row, col=[], dtype=np.float32, as_list=False):
	""" Computes an elementary matrix, i.e. a matrix with a single non-zero 
	entry, by computing a consecutive tensor product of elementary matrices of 
	dimension base*base. The number of products are given by the length of row
	and the entry index of the 1 entry in the kth tensor factor is given by
	row[k-1] and col[k-1]. If each tensor factor is diagonal, only row must be 
	input (col is kept as the empty list).

	If base = 2 and row and col are lists of binary digits, the leading entry is 
	the most signficant bit. 

	Inputs:
		base 		Int specifying the size of each tensor factor;
		row 		List of ints between 0 and base;
		col 		List of ints between 0 and base;
		dtype 		Specifies the dtype (np.float64 may cause problems due to 
					size issues later);
		as_list 	Bool which determines whether the diagonal of the matrix 
					should be returned as a list instead (useful for dim 
					reduction in the case of diagonal qcombs).

	Returns:
		np.array of dimension (base**len(row)) by (base**(len(row)))
	"""
	if len(col) > 0:
		if len(row) != len(col):
			raise ValueError("If col is not empty, then row and col must be" 
						 " equal length.")
		if not (all(0 <= r < base for r in row) 
				and all(0 <= c < base for c in col)):
			raise ValueError("Indices must be in [0,..., base-1].")
		col = col.copy()
	else:
		col = row.copy()
	row = row.copy()
	elem_mat = np.identity(1, dtype=dtype)

	while len(row) > 0:
		if as_list:
			A = np.zeros(base)
		else:
			A = np.zeros((base, base))
		row_index = row.pop(0)
		col_index = col.pop(0)
		if as_list:
				A[row_index] += 1.0
		else:
			A[row_index][col_index] += 1.0
		elem_mat = np.kron(elem_mat, A)

	return elem_mat.copy()




# # # # # # # # # # # # # Quantum Comb Constraints # # # # # # # # # # # # # 

def qcomb_constraints(expr, in_out_dims, normalised=False, tolerance=1e-8):
	""" 
	This function either returns a list of constraints that expr must 
	satisfy such that it is a quantum comb, in the case where expr is a
	cvx.Variable, or returns a boolean specifying whether the matrix expr is a 
	quantum comb.

	A quantum comb C is a PSD matrix that satisfies the recursive relations

		Tr_{A_{n}^{out}}[C_{n}] =  C_{n-1} otimes Id_{A_{n}^{in}}

	for all n in {1, ..., N} where the trace is the partial trace over the
	Hilbert space H_{A_{n}^{out}}, C_{n} is a suitable operator on the 
	Hilbert space bigotimes_{j = 1}^{n} H_{A_{j}^{in}} otimes H_{A_{j}^{out}},
	C_{N} = C and C_{0} = 1.

	When expr is a cvx Variable or if normalised=False, then the C_{0} 
	constraint is modified to C_{0} >= constant > 0. 

	expr 			either cvx.expressions.variable.Variable or anything that 
					can be cast to cvx.expressions.constant.Constant;
	in_out-dims		list of input Hilbert space and output Hilbert space 
					dimensions strictly following an alternating order, e.g.
					[in, out, in, ..., in, out];
	normalised		Boolean indicating whether to check if expr is a normalised
					quantum comb;
	tolerance		scalar value stipulating how much error is tolerated for 
					treating a strict inequality. Preset value is the same as
					for cvx.constraints.constraint.Constraint.
	"""
	if (len(expr.shape) != 2 or expr.shape[0] != expr.shape[1]):
		raise ValueError("Only square expressions are currently treated.")
	# The specified dimensions and total dimension of the operator must match.
	if expr.shape[0] != np.prod(in_out_dims):
		raise ValueError("Total dim and input-output dims don't match.")
	if not all(x >= 1 and isinstance(x, int) for x in in_out_dims):
		raise ValueError("Invalid in_out_dims were specified.")

	# If expr is a cvx.Variable, then comb constraints are generated.
	expr_is_variable = isinstance(expr, cvxtypes.variable())
	
	# If expr is not a cvx.Variable, it is cast to a cvx.Constant and the comb 
	# constraints are checked rather than generated.
	if not expr_is_variable and not isinstance(expr, cvxtypes.constant()):
		expr = cvxtypes.constant()(expr)

	# Expr must by PSD:
	if expr_is_variable:
		constraints = [expr >> 0]
	else:
		# The exception handling is to prevent termination of the checking of 
		# the comb constraints if the eigenvalue methods involved in checking 
		# PSD and hermiticity don't converge. (This is risky, but mostly we
		# will consider only matrices that are known to be PSD)
		try:
			is_hermitian = expr.is_hermitian()
			is_PSD = expr.is_psd()
			if not (is_hermitian and is_PSD):
				return False
		except (ArpackNoConvergence, ArpackError):
			print("An ArpackError occurred (such as non-convergence of"
				  " eigenvalue methods).")

	in_out_dims = in_out_dims.copy()

	# Recurrent constraints: Tr_{n, out}[C_{n}] = C_{n-1} \otimes I_{n, in} 
	while len(in_out_dims) > 0:
		# Tr_{n, out}[C_{n}]:
		if in_out_dims[-1] == 1: # No partial trace over out systems of dim 1
			pt_expr	= expr
		else:
			pt_expr = cvx.partial_trace(expr, in_out_dims, len(in_out_dims)-1)

		in_out_dims.pop()

		# C_{n-1} = (1/dim(n, in))Tr_{n, in}[Tr_{n, out}[C_{n}]]:
		if in_out_dims[-1] == 1: # No partial trace over in systems of dim 1
			expr = pt_expr
		else:	
			expr = (1/in_out_dims[-1])*cvx.partial_trace(pt_expr,
														in_out_dims,
														len(in_out_dims)-1)

		id_and_expr = cvx.kron(expr, np.identity(in_out_dims[-1]))
		in_out_dims.pop()
		
		if expr_is_variable:
			constraints += [pt_expr == id_and_expr]
		else:
			if not np.allclose(pt_expr.value, id_and_expr.value):
				return False

	# Final constraint, i.e. regarding C_{0}:
	if expr_is_variable: 
		# Append C_{0} >= const > 0:
		constraints += [cvx.real(expr) >= tolerance] 	# cvx.real() is required 
														# since expr is a 
														# complex variable.
		return constraints
	else:
		if normalised and not np.allclose(expr.value, 1.0):
			return False
		elif not normalised and expr.value <= 0.0:
			return False
		else:
			return True

	

# # # # # # # # # # # # # Classical Comb Constraints # # # # # # # # # # # # # 


def list_qcomb_constraints(list_expr, in_out_dims, normalised=False, 
						   tolerance=1e-8):
	""" 
	This function either returns a list of constraints that list_expr must 
	satisfy such that the corresponding matrix with list_expr as diagonal is a 
	"quantum" comb, in the case where list_expr is a cvx.Variable, or returns
	a boolean specifying whether the matrix coresponding to list_expr is an 
	instance of a comb.
	
	The general quantum comb constraints are detailed in the function 
	qcomb_contraints() above. The constraints modified for the diagonal case,
	the classical comb case are:

		> every element of list_expr must be >= 0 
		> sum_{j in OUT} list_expr @ (I ot ... ot |j>) 
			= (sum_{j in OUT, i in IN} list_expr @ (I ot ... |ij>)) ot [1,...,1]

	with the first constraint corresponding to the PSD constraint and the 
	second corresponding to the partial trace constraints (see 
	list_partial_trace for the 1D partial trace).

	When list_expr is a cvx.Variable or if normalised=False, then the C_{0} 
	constraint is modified to C_{0} >= constant > 0. 

	list_expr 		either cvx.expressions.variable.Variable or anything that 
					can be cast to cvx.expressions.constant.Constant;
	in_out-dims		list of input Hilbert space and output Hilbert space 
					dimensions strictly following an alternating order, e.g.
					[in, out, in, ..., in, out];
	normalised		Boolean indicating whether to check if expr is a normalised
					quantum comb;
	tolerance		scalar value stipulating how much error is tolerated for 
					treating a strict inequality. Preset value is the same as
					for cvx.constraints.constraint.Constraint.
	"""
	if list_expr.shape != (1, np.prod(in_out_dims)):
		raise ValueError("List_expr must be of shape (1, prod(in_out_dims)).")
	if not all(x >= 1 and isinstance(x, int) for x in in_out_dims):
		raise ValueError("Invalid in_out_dims were specified.")


	# If expr is a cvx.Variable, then comb constraints are generated.
	expr_is_variable = isinstance(list_expr, cvxtypes.variable())

	
	# If expr is not a cvx.Variable, it is cast to a cvx.Constant and the comb 
	# constraints are checked rather than generated.
	if not expr_is_variable and not isinstance(list_expr, cvxtypes.constant()):
		list_expr = cvxtypes.constant()(list_expr)

	# Expr must by PSD:
	if expr_is_variable:
		constraints = [list_expr[0][x] >= 0 for x in range(list_expr.shape[1])]
	else:
		if not all(x >= 0 for x in list_expr.value[0]):
			return False


	in_out_dims = in_out_dims.copy()

	# Recurrent constraints: Tr_{n, out}[C_{n}] = C_{n-1} \otimes I_{n, in} 
	while len(in_out_dims) > 0:
		# Tr_{n, out}[C_{n}]:
		if in_out_dims[-1] == 1: # No partial trace over out systems of dim 1
			pt_expr	= list_expr
		else:
			pt_expr = list_partial_trace(list_expr, in_out_dims, 
										 len(in_out_dims)-1)
		in_out_dims.pop()

		# C_{n-1} = (1/dim(n, in))Tr_{n, in}[Tr_{n, out}[C_{n}]]:
		if in_out_dims[-1] == 1: # No partial trace over in systems of dim 1
			list_expr = pt_expr
		else:	
			list_expr = (1/in_out_dims[-1])*list_partial_trace(pt_expr,
															   in_out_dims,
															 len(in_out_dims)-1)

		id_and_expr = cvx.kron(list_expr, np.ones((1,in_out_dims[-1])))
		in_out_dims.pop()
		
		if expr_is_variable:
			constraints += [pt_expr == id_and_expr]
		else:
			if not np.array_equal(pt_expr.value, id_and_expr.value):
				return False

	# Final constraint, i.e. regarding C_{0}:
	if expr_is_variable: 
		# Append C_{0} >= const > 0:
		constraints += [list_expr >= tolerance]
		return constraints
	else:
		if normalised and list_expr.value != 1:
			return False
		elif not normalised and list_expr.value <= 0:
			return False
		else:
			return True











"""

def elementary_matrix1(base, row, col=[], dtype=np.float32, as_list = False):
	Computes an elementary matrix, i.e. a matrix with a single non-zero 
	entry, by computing a consecutive tensor product of elementary matrices of 
	dimension base*base. The number of products are given by the length of row
	and the entry index of the 1 entry in the kth tensor factor is given by
	row[k-1] and col[k-1]. If each tensor factor is diagonal, only row must be 
	input (col is kept as the empty list).

	If base = 2 and row and col are lists of binary digits, the leading entry is 
	the most signficant bit. 

	Inputs:
		base 		Int specifying the size of each tensor factor;
		row 		List of ints between 0 and base;
		col 		List of ints between 0 and base;
		dtype 		Specifies the dtype (np.float64 may cause problems due to 
					size issues later);
		as_list 	Bool which determines whether the diagonal of the matrix 
					should be returned as a list instead (useful for dim 
					reduction in the case of diagonal qcombs).

	Returns:
		np.array of dimension (base**len(row)) by (base**(len(row))) if as_list
		is false, otherwise of dimension 1 by (base**(len(row))).
	
	if len(col) > 0:
		if len(row) != len(col):
			raise ValueError("If col is not empty, then row and col must be" 
						 " equal length.")
		if not (all(0 <= r < base for r in row) 
				and all(0 <= c < base for c in col)):
			raise ValueError("Indices must be in [0,..., base-1].")
		col = col.copy()
	else:
		col = row.copy()
	row = row.copy()

	is_binary = False
	if base == 2 and len(row) > 1:
		is_binary = True

	elem_mat = np.identity(1, dtype=dtype)
	
	while len(row) > 0:
		if as_list:
			A = np.zeros(base)
		else:
			A = np.zeros((base, base))

		# For the case where row and col specify binary numbers, i.e. with 
		# leading entry as most significant bit, the order of taking the
		# Kronecker product is reversed.
		if is_binary:
			row_index = row.pop()
			col_index = col.pop()
			if as_list:
				A[row_index] += 1
			else:
				A[row_index][col_index] += 1
			elem_mat = np.kron(A, elem_mat)
		else:
			row_index = row.pop(0)
			col_index = col.pop(0)
			if as_list:
				A[row_index] += 1
			else:
				A[row_index][col_index] += 1
			elem_mat = np.kron(elem_mat, A)

	return elem_mat

"""