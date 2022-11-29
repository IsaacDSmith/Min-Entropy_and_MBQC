"""
This script is the main file for infering the gflow of a MBQC device,
while only using projective measurements on the first two qubits. 

"""
import sys
import cvxpy as cvx
import numpy as np
import os
import itertools
from Grey_Box_generate_combs import *

# TO DO:
#   > rewrite header


def observational_strategy(op = grey_box_MBQC_gflows(rounds=1,dtype=cp.cfloat),
						   num_flows=15, max_resolution=7, adaptive=False,
						   incl_intermediate=False)
	""" Calculates the probability Tr[op E^T] where op is one of the operators
	(combs) defined in Grey_Box_generate_combs.py and where E is a dual comb 
	corresponding to an ``observational strategy'' i.e. projective measurement 
	on the first two qubits of op followed by a CPTP map on the output qubits.
	This map is optimised over using a CVXPY SDP solver.

	Parameters:
		op 					f
		num_flows			f
		max_resolution		f
		adaptive			Boolean
		incl_intermediate 	Boolean

	Returns:
		Dictionary of results, including best probability and corresponding 
		angles.

	"""
	increment = 2.0*np.pi/max_resolution

	# The following parameter determines the matrix size to save intermediate 
	# outputs.
	if adaptive:
		# 6 angles
		meshSize = max_resolution**6
	else: 
		# 4 angles
		meshSize = max_resolution**4


	# Save the intermediate outputs:
	if incl_intermediate:
		theta1_list = np.zeros(shape=(meshSize), dtype=np.float64)
		phi1_list = np.zeros(shape=(meshSize), dtype=np.float64)
		theta2a_list = np.zeros(shape=(meshSize), dtype=np.float64)
		phi2a_list = np.zeros(shape=(meshSize), dtype=np.float64)
		theta2b_list = np.zeros(shape=(meshSize), dtype=np.float64)
		phi2b_list = np.zeros(shape=(meshSize), dtype=np.float64)
		probs = np.zeros(shape=(meshSize), dtype=np.float64)


	# Record the best result so far:
	bestProb = 0.0 
	bestTheta1 = 0.0
	bestPhi1 = 0.0
	bestTheta2a = 0.0
	bestPhi2a = 0.0
	bestTheta2b = 0.0
	bestPhi2b = 0.0

	# The following will be used to address the right cell in the arrays  
	# containing intermediate results.
	index = 0

	# Create to value sets to iterate over.
	if adaptive:
		iterSet = itertools.product(range(max_resolution), repeat = 6)
	else:
		# If the second measurement is not dependent on the first outcome, we  
		# set the angles for the unused measurement to a fixed value and never
		# use them. This allows to keep the code shorter.
		iterSet = itertools.product(range(max_resolution),
									range(max_resolution),
									range(max_resolution),
									range(max_resolution),
									range(1),
									range(1))


	# Optimising over the observational strategies via an exhaustive search on a 
	# grid. disTheta1, disPhi1 are the angles for the 0-outcome of the   
	# measurement on the first qubit. disTheta2a, disPhi2a are the angles for 
	# the 0-outcome of the measurement on the second qubit. If adaptive = True, 
	# these will only be used if the first measurement had outcome 0, otherwise  
	# disTheta2b, disPhi2b will be used.

	for disTheta1,disPhi1,disTheta2a,disPhi2a,disTheta2b,disPhi2b in iterSet:
		
		if incl_intermediate:
			# Save current angle settings:
			theta1_list[index] = disTheta1*increment
			phi1_list[index] = disPhi1*increment
			theta2a_list[index] = disTheta2a*increment
			phi2a_list[index] = disPhi2a*increment
			
			if adaptive:
				theta2b_list[index] = disTheta2b*increment
				phi2b_list[index] = disPhi2b*increment

		# Create the combs for the projective measurements on the first two 
		# qubits. meas[o1,o2, :, :] is the comb for outcome o1 in the first 
		# measurement and outcome o2 in the second measurement. If adaptive = 
		# False, disTheta2b*increment and disPhi2b*increment are ignored.
		meas = observational_meas_small(disTheta1*increment, 
										disPhi1*increment, 
										disTheta2a*increment, 
										disPhi2a*increment, 
										disTheta2b*increment, 
										disPhi2b*increment, 
										adaptive)

		# An SDP variable is created for every outcome, representing the quantum
		# channel applied to the last two qubits of op. The last two 
		# digits in the name specify the outcome combination.
		restComb00=cvx.Variable((numFlows*(2**2),numFlows*(2**2)),complex=True)
		restComb01=cvx.Variable((numFlows*(2**2),numFlows*(2**2)),complex=True)
		restComb10=cvx.Variable((numFlows*(2**2),numFlows*(2**2)),complex=True)
		restComb11=cvx.Variable((numFlows*(2**2),numFlows*(2**2)),complex=True)

		# Positive semidefiniteness constraints:
		constraints = [restComb00 >> 0]
		constraints += [restComb01 >> 0]
		constraints += [restComb10 >> 0]
		constraints += [restComb11 >> 0]

		# Partial trace constraints:
		rest00OnlyIn = cvx.partial_trace(restComb00, [4,numFlows], 1)
		rest01OnlyIn = cvx.partial_trace(restComb01, [4,numFlows], 1)
		rest10OnlyIn = cvx.partial_trace(restComb10, [4,numFlows], 1)
		rest11OnlyIn = cvx.partial_trace(restComb11, [4,numFlows], 1)
		
		# If the output is discarded, the channel only acts as the trace.
		constraints += [rest00OnlyIn == np.identity(4, dtype = np.cfloat)]
		constraints += [rest01OnlyIn == np.identity(4, dtype = np.cfloat)]
		constraints += [rest10OnlyIn == np.identity(4, dtype = np.cfloat)]
		constraints += [rest11OnlyIn == np.identity(4, dtype = np.cfloat)]


		# Since the output of the final channel is a guess for the gflow,
		# we will decohere the output of the final channel to make it classical
		projs = []
		# Create the projectors on the computational basis of the output:
		for j in range(numFlows):
			lastFactor = np.zeros(shape=(numFlows,numFlows), dtype=np.cfloat)
			lastFactor[j,j] = 1.0 +0.0j
			proj = np.kron(np.identity(4, dtype=np.cfloat), lastFactor)
			projs.append(proj.copy())

		# Now, we iterate over the computational basis to decohere the output of 
		# the last channel
		decohered00 = projs[0] @ restComb00 @ projs[0]
		decohered01 = projs[0] @ restComb01 @ projs[0]
		decohered10 = projs[0] @ restComb10 @ projs[0]
		decohered11 = projs[0] @ restComb11 @ projs[0]

		for j in range(numFlows-1):
			decohered00 = decohered00 + (projs[j+1] @ restComb00 @ projs[j+1])
			decohered01 = decohered01 + (projs[j+1] @ restComb01 @ projs[j+1])
			decohered10 = decohered10 + (projs[j+1] @ restComb10 @ projs[j+1])
			decohered11 = decohered11 + (projs[j+1] @ restComb11 @ projs[j+1])

		constraints += [decohered00 == restComb00]
		constraints += [decohered01 == restComb01]
		constraints += [decohered10 == restComb10]
		constraints += [decohered11 == restComb11]

		# Combine the pieces to obtain the full strategy comb.
		# meas[o1,o2, : , :] represents the component for the first two qubits, 
		# for outcomes o1 and o2. Since the strategy comb will be contracted 
		# with op, the transpose is applied. Note, meas[o1,o2, : ,: ] is 
		# already transposed.

		strategy = cvx.kron(meas[0,0,:,:], 
							cvx.transpose(restComb00))
							+cvx.kron(meas[0,1,:,:], cvx.transpose(restComb01))
		strategy = strategy + cvx.kron(meas[1,0,:,:], 
									   cvx.transpose(restComb10))
									   +cvx.kron(meas[1,1,:,:],
												 cvx.transpose(restComb11))

		# Contract the combs to calculate the success probability:
		success_prob = cvx.trace(strategy @ op)

		# Calculate the best strategy for the current angle settings:
		problem = cvx.Problem(cvx.Maximize(cvx.real(success_prob)), 
							  constraints)
		problem.solve(solver=cvx.SCS,verbose=False)

		if incl_intermediate:
			probs[index] = cvx.real(success_prob).value

		# Save the best success probability and its settings:
		if cvx.real(success_prob).value > bestProb:
			bestProb = cvx.real(success_prob).value
			bestTheta1 = disTheta1*increment
			bestPhi1 = disPhi1*increment
			bestTheta2a = disTheta2a*increment
			bestPhi2a = disPhi2a*increment
			if adaptive:
				bestTheta2b = disTheta2b*increment
				bestPhi2b = disPhi2b*increment
			bestRest00 = restComb00.value
			bestRest01 = restComb01.value
			bestRest10 = restComb10.value
			bestRest11 = restComb11.value
		
		index += 1

	results = {}
	results["bestProb"] = bestProb
	results["bestTheta1"] = bestTheta1
	results["bestPhi1"] = bestPhi1
	results["bestTheta2a"] = bestTheta2a
	results["bestPhi2a"] = bestPhi2a
	if adaptive:
		results["bestTheta2b"] = bestTheta2b
		results["bestPhi2b"] = bestPhi2b
	if incl_intermediate:
		results["theta1_list"] = theta1_list
		results["phi1_list"] = phi1_list
		results["theta2a_list"] = theta2a_list
		results["phi2a_list"] = phi2a_list
		results["theta2b_list"] = theta2b_list
		results["phi2b_list"] = phi2b_list
		results["probs"] = probs

	return results