"""
The script contains functions adapted from cvxpy.atoms.affine.partial_trace.py 
(see https://github.com/cvxpy/cvxpy) as at 14.10.2022. The aim is to be as  
close as possible to the original while implementing the partial trace over a 1D 
expression (to be interpreted as the diagonal of a matrix).

This script contains two functions:
	
	> list_partial_trace 	computes the partial trace of a diagonal matrix
							represented as a 1D expression containing only the
							diagonal elements;
	> _term					a helper function used in taking the partial trace;

The only logical changes to the original code exist in the _term function, 
namely by removing the operator 'a' from the code. In most cases, it has simply 
been commented out, however the return of the original function _term is  
a@expr@b so in this case it has been deleted. The remainder of the code is 
verbatim from the original except for minor changes: the name of the function is 
now list_partial_trace, and there are formatting changes to the comments.

This script primarily interacts with comb_constraints.py.


Copyright 2022 adapted from cvxpy.atoms.affine.partial_trace.py

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.atom import Atom

def _term(expr, j: int, dims: Tuple[int], axis: Optional[int] = 0):
	"""Helper function for partial trace.
	Parameters
	----------
	expr : :class:`~cvxpy.expressions.expression.Expression`
		The 2D expression to take the partial trace of.
	j : int
		Term in the partial trace sum.
	dims : tuple of ints.
		A tuple of integers encoding the dimensions of each subsystem.
	axis : int
		The index of the subsystem to be traced out
		from the tensor product that defines expr.
	"""
	# (I ⊗ <j| ⊗ I) x (I ⊗ |j> ⊗ I) for all j's
	# in the system we want to trace out.
	# This function returns the jth term in the sum, namely
	# (I ⊗ <j| ⊗ I) x (I ⊗ |j> ⊗ I).
	#a = sp.coo_matrix(([1.0], ([0], [0])))
	b = sp.coo_matrix(([1.0], ([0], [0])))
	for (i_axis, dim) in enumerate(dims):
		if i_axis == axis:
			v = sp.coo_matrix(([1], ([j], [0])), shape=(dim, 1))
			#a = sp.kron(a, v.T)
			b = sp.kron(b, v)
		else:
			eye_mat = sp.eye(dim)
			#a = sp.kron(a, eye_mat)
			b = sp.kron(b, eye_mat)
	return expr @ b


def list_partial_trace(expr, dims: Tuple[int], axis: Optional[int] = 0):
	"""
	Assumes :math:`\\texttt{expr} = X_1 \\otimes \\cdots \\otimes X_n` is a 1D 
	Kronecker product composed of :math:`n = \\texttt{len(dims)}` implicit 
	subsystems. Letting :math:`k = \\texttt{axis}`, the returned expression 
	represents the *partial trace* of :math:`\\texttt{expr}` along its 
	:math:`k^{\\text{th}}` implicit subsystem:
	.. math::
		\\text{tr}(X_k) (X_1 \\otimes \\cdots \\otimes X_{k-1} 
								\\otimes X_{k+1} \\otimes \\cdots \\otimes X_n).
	Parameters
	----------
	expr : :class:`~cvxpy.expressions.expression.Expression`
		The 1D expression to take the partial trace of.
	dims : tuple of ints.
		A tuple of integers encoding the dimensions of each subsystem.
	axis : int
		The index of the subsystem to be traced out
		from the tensor product that defines expr.
	"""
	expr = Atom.cast_to_const(expr)
	if expr.ndim != 2 or expr.shape[0] != 1:
		raise ValueError("Only supports matrices of shape (1, prod of dims).")
	if axis < 0 or axis >= len(dims):
		raise ValueError(
			f"Invalid axis argument, should be between 0 and {len(dims)},"
			" got {axis}."
		)
	if expr.shape[1] != np.prod(dims):
		raise ValueError("Dimension of system doesn't correspond to dimension "
						 "of subsystems.")
	return sum([_term(expr, j, dims, axis) for j in range(dims[axis])])