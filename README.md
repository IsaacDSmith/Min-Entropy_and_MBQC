# Min-Entropy and MBQC

This repository houses the code related to the manuscript [_"Classical-Quantum Combs, their Min-Entropy and their Measurement-Based Applications"_](https://arxiv.org/abs/2212.00553) situated in the research field of quantum information theory. An overview of this work is provided below, but in brief, the work solves a number of semi-definite programs (SDPs) related to [Measurement-Based Quantum Computation](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188). The code implementing these optimisations relies on [CVXPY](https://github.com/cvxpy).

## Abstract of the Work:

Learning a hidden property of a quantum system typically requires a series of interactions. In this work, we consider a formalisation of such multi-round learning processes that uses a generalisation of classical-quantum states called classical-quantum combs. Here, "classical" refers to a random variable encoding the hidden property to be learnt, and "quantum" refers to the quantum comb describing the behaviour of the system. By using the quantum combs formalism, the optimal strategy for learning the hidden property can be quantified via the comb min-entropy (Chiribella and Ebler, NJP, 2016). With such a tool on hand, we focus attention on an array of combs derived from measurement-based quantum computation (MBQC) and related applications. Specifically, we describe a known blind quantum computation (BQC) protocol using the combs formalism and thereby leverage the min-entropy to provide a proof of single-shot security for multiple rounds of the protocol, extending the existing result in the literature. Furthermore, we introduce novel connections between MBQC and quantum causal models and quantum causal inference, which allows for the use of the min-entropy to quantify the optimal strategy for causal discovery. We consider further operationally motivated examples, including one associated to learning a quantum reference frame.

## Overview of the Code:

The [comb min-entropy](https://arxiv.org/abs/1606.02394) is an extension of the [min-entropy for quantum states](https://arxiv.org/abs/0807.1338) and can be calculated via a semi-definite program. In this repository, the comb min-entropy for a given comb $D$ is calculated via

$$ -\log\left( \min_{\Gamma} \frac{1}{N}\text{Tr}[\Gamma] \right)$$

where the minimisation is performed over unnormalised combs $\Gamma$ on the same spaces as $D$ except for a designated output space that furthermore satisfy $I_{text{output}} \otimes \Gamma \geq D$. $N$ is a normalising constant related to (some of) the dimensions of $\Gamma$ and $\text{Tr}$ denotes the trace. The term inside the logarithm is called the _guessing probability_.

The two key scripts in this repository are:
- comb_constraints.py
- guessing_probability.py

The script **comb_constraints.py** contains functions that generate the linear constraints corresponding to $\Gamma$ being an unnormalised comb and *guessing_probability.py* computes the guessing probability by passing the required constraints (those of comb_constraints.py and also $I_{text{output}} \otimes \Gamma \geq D$) to a CVXPY solver. Due to the nature of some of the specific examples considered in this work (see below), both scripts contain the same functionality for 2-dimensional objects (matrices) and 1-dimensional objects which are to be understood as the diagonal elements of diagonal matrices. The comb constraints for the latter make use of **list_partial_trace.py**, which is a modification of [the CVXPY partial trace](https://github.com/cvxpy/cvxpy/blob/master/cvxpy/atoms/affine/partial_trace.py) to handle the 1-dimensional (list) version.

The remaining scripts are related to the specific examples considered in the manuscript, as follows:
- **Min_BQC_generate_comb_as_list.py** - generates the comb related to the blind quantum computing protocol considered in the paper (from [this paper](https://arxiv.org/abs/1608.04633)) for a minimal example. This comb is diagonal, so is treated as a 1-dimensional object. The guessing probability is calculated by passing this comb in the appropriate function of guessing_probability.py;
- **Min_BQC_two_rounds_guessing_probability.py** - calculates the guessing probability for two rounds of the above BQC protocol for the minimal example via symbolic methods. This script is distinct from the other scripts in this repository as it uses no semi-definite programming. Its existence was necessitated by size issues that occurred when attempting to use SDP methods for two round comb generated using the above script;
- **Grey_Box_generate_combs.py** - generates a range of combs related a "grey box" device that implements a measurement-based computation. The combs generated here include those related to quantum causal discovery mentioned in the abstract above;
- **observational_strategy.py** - In the manuscript, the guessing probability associated to the comb corresponding to the quantum causal discovery is compared to a similar quantity with a further restriction on the set of operators being optimised over. This script computes this quantity via slightly different methods to that used in the unrestricted guessing probability calculation;
- **Calibr_Meas_generate_combs.py** - generates the combs related to learning the quantum reference frame within the grey box MBQC device and that of the measurement devices outside the box.

Please note that the linked references above are provided as an aid to understanding the context of this repository and represent only a portion of the relevant material. See the manuscript for more complete referencing.
