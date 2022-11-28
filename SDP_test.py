import sys
import cvxpy as cvx
from cvxpy.expressions import cvxtypes
import numpy as np
import scipy as sp
from qcomb_constraint import qcomb_constraints
import pickle
from datetime import date
import time
#from conditional_min_entropy import conditional_min_entropy


normed_record_dict = pickle.load(open("Min_BQC,2022-10-10,checking_qcomb_"
									  "constraints,is_normed_True"
									  "4_angles,1_rounds.p", 
									  "rb"))

unnormed_record_dict = pickle.load(open("Min_BQC,2022-10-10,checking_qcomb_"
										"constraints,is_normed_False"
										"4_angles,1_rounds.p", 
										"rb"))

print("=======================================================================")
print("The normed case:")
print("It is a comb: {}".format(normed_record_dict["is_comb"]))
print("It took {} to check".format(normed_record_dict["check_time"]))






print("=======================================================================")
print("The unnormed case:")
print("It is a comb: {}".format(unnormed_record_dict["is_comb"]))
print("It took {} to check".format(unnormed_record_dict["check_time"]))


"""

# The code below was used to create two files:
Min_BQC,2022-10-10,checking_qcomb_constraints,is_normed_True4_angles,1_rounds.p
Min_BQC,2022-10-10,checking_qcomb_constraints,is_normed_False4_angles,1_rounds.p

which contain the results of whether the comb stored in

Min_BQC,2022-10-09,4_angles,1_rounds.p

satisfies the qcomb contraints in the normalised and unnormalised cases. The 
results are analysed above.

normed = sys.argv[1]
if normed == "False":
	normed = False
elif normed == "True":
	normed = True
else:
	raise ValueError("Input valid bool")


record_dict = pickle.load(open("Min_BQC,2022-10-09,4_angles,1_rounds.p", "rb"))
print("Time to compile was {}".format(record_dict["time_to_compile"]))
print("Trace is {}".format(record_dict["trace_D_client"]))
D_client = record_dict["D_client"]
dims = [1, 4, 2, 4, 2, 4, 2, 64]
print("dims are {}".format(dims))
print("D_client shape is {}".format(D_client.shape))

t1 = time.time()
is_comb = qcomb_constraints(D_client, dims, normalised=normed)
t2 = time.time()


qcomb_constraints_check = {}
qcomb_constraints_check["is_comb"] = is_comb
qcomb_constraints_check["is_normed"] = normed
qcomb_constraints_check["check_time"] = t2 - t1

pickle.dump(qcomb_constraints_check, 
			open("Min_BQC,{},checking_qcomb_constraints,is_normed_{}"
				"4_angles,1_rounds.p".format(date.today(), normed),
				"wb"))



"""














#solver = cvx.SCS


#dims = [1,2,3,2,3,3]
#m = np.prod(dims)

#D = np.zeros((m,m))
#D1 = (1/m)*np.identity(m)
#D2 = (1/(2*2*3))*np.identity(m)
#D2[0][0] = 3
#D2 = cvxtypes.constant()(D2)

#D3 = np.identity()

#print(qcomb_constraints(D2, [1,2,3,2,3,3], normalised=True))

#for i in range(m):
#	D[i][i] = 1/(i+1)





#conditional_min_entropy(D1, [1,2,3,2,3,3], solver=solver)



"""
##### GRAVEYARD #####


# Test out partial trace:
A = np.matrix([[1,1],[1,1]])
B = np.matrix([[2,3],[4,5]])
D = np.matrix([[8, 8], [9, 9]])
C = np.kron(np.kron(A, B), D)




# Partial Trace investigations

F = np.kron(np.identity(2), np.identity(2))
print(F)


#left = np.matrix([[1, 0, 0, 0],[0, 1, 0, 0]])
#right = np.matrix([[1,0],[0,1],[0,0],[0,0]])
#print(left @ F @ right)

#print(_term(F, 0, [2, 2], 0) + _term(F, 1, [2,2], 0))

pt_F = partial_trace(F, [2,2], 0)
print(pt_F)






#print(C)
#print(C.shape)
#print(cvx.partial_trace(C, [2,2,2], 0))

#E = cvx.partial_trace(C, [2,2,2], 0)
#print(type(E))
#print(E)

#rho = cvx.Expression.cast_to_const(C)
#print(rho)
#print(type(rho))
#print(rho.T)
#print(rho.H)

#expr_A = Leaf([2,2], [[1,1],[1,1]])
#print(expr_A)



# Testing out problem sovling funcitonality


lmbd = cvx.Variable()
constraints = [lmbd*np.identity(2) >> A]

#constraints = [cvx.constraints.psd.PSD((lmbd*np.kron(B,D)) - np.kron(A,B))]
#constraints1 = Constraint(constraints)

#print(np.kron(A,B))
#print(np.kron(B,D))
#print(lmbd.value)

#prob = cvx.Problem(cvx.Minimize(lmbd), constraints)
#prob.solve(verbose=True)
#print(constraints1.is_affine())

#prob.solve(verbose=True)

#print("The optimal value is {}".format(prob.value))

#print("A solution is {}".format(X.value))
#print(prob.solver_stats().solve_time)




# Towards comb constraints

n = 16

X = cvx.Variable((n,n))
print(type(cvx.constraints.psd.PSD(X)))
constraints1 = [cvx.constraints.psd.PSD(X)]
constraints1 += [cvx.partial_trace(X, [2,2,2,2], 0) == np.identity(8)]

prob1 = cvx.Problem(cvx.Minimize(cvx.trace(X)), constraints1)

#prob1.solve(verbose=True)












# Partial trace code taken from https://github.com/cvxpy/cvxpy/issues/563
# as contributed by SteveDiamond on 14.01.22 which is the Python rewrite of
# the Julie code from https://github.com/jump-dev/Convex.jl/blob/4068f80ec581ae
# 77d8000d7f3fcb0387ce306577/src/atoms/affine/partialtrace.jl

def _term(rho, j, dims, axis=0):
	# (I ⊗ <j| ⊗ I) x (I ⊗ |j> ⊗ I) for all j's
	# in the system we want to trace out.
	# This function returns the jth term in the sum, namely
	# (I ⊗ <j| ⊗ I) x (I ⊗ |j> ⊗ I).
	a = np.matrix([1])
	b = np.matrix([1])
	for (i_axis, dim) in enumerate(dims):
		if i_axis == axis:
			v = np.zeros((dim, 1))
			v[j] = 1
			a = np.kron(a, v.T)
			b = np.kron(b, v)
		else:
			eye_mat = np.identity(dim)
			a = np.kron(a, eye_mat)
			b = np.kron(b, eye_mat)
	return cvx.matmul(cvx.matmul(a, rho), b)

def partial_trace(rho, dims, axis=0):
	rho = cvx.Expression.cast_to_const(rho)
	if rho.ndim < 2 or rho.shape[0] != rho.shape[1]:
		raise ValueError("Only supports square matrices.")
	if axis < 0 or axis >= len(dims):
		raise ValueError(f"Invalid axis argument, should be between 0 and {len(dims)}, got {axis}.")
	if rho.shape[0] != np.prod(dims):
		raise ValueError("Dimension of system doesn't correspond to dimension of subsystems.")
	return sum([_term(rho, j, dims, axis) for j in range(dims[axis])])


"""