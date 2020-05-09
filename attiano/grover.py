import sys
import inspect
import unittest
from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate
from itertools import combinations 
import numpy as np
import math
from operator import add

num_trials = 10



def create_minus_gate(n):
	N = 2**n
	
	minus = np.identity(N)

	for i in range(N):
		minus[i][i] = -1
	
	return minus

def create_zf(f,n):
	"""
	input: int n = (number of qubits on which zf acts), array<int> f = function being encoded/evaluated 
	output: zf = (-1)^f(x)|x>, the identity matrix with -1 on the rows that f(x) returns 1
	"""
	N = 2**n

	zf = np.identity(N)

	for i in range(N):
		f_val = int(f[i])
		if(f_val == 1):
			zf[i,i] = -1
		
	return zf
	
def create_z0(n):
	"""
	input: int n = (number of qubits on which zf acts), array<int> f = function being encoded/evaluated 
	output: z0 = -|x> if |x> = 0^n else |x> the identity matrix with -1 on first row
	"""
	N = 2**(n)

	z0 = np.identity(N)
	z0[0][0] = -1

	return z0

def all_f(n,a):
	""" 
	input: int n = number of qubits, int a = number of items that return 1 for function
	output: list of functions (arrays that can be index with 'x' for function value 'f(x)')

	"""  
	N = 2**n
	func_inputs = range(N)
	comb = combinations(func_inputs, a)
	func_list = []

	for combination in comb:
		out_list = np.zeros(N)
		for i in combination:
			out_list[i] = 1
		func_list.append(out_list)
		
	return func_list

def calc_lim(a,n):
	N = 2**n
	#theta = np.arcsin(a/N)
	theta = 1/np.sqrt(N)
	k = math.floor(abs(((np.pi)/(4*theta)) - 0.5))
	return int(k)

def grover(z0,zf,n,a):
	#f is the function we oracle call on
	#assum input is n length bit string, with each bit value as a different input
	
	
	#try every possible input - brute force, if f(x) = return 1, otherwise return 0
	z_0 = z0.get_constructor()
	z_f = zf.get_constructor()
	minus = create_minus_gate(1)
	minus_def = DefGate("MINUS", minus)
	minus_gate = minus_def.get_constructor()
	p = Program()
	p += zf
	p += z0
	p += minus_def

	#1. Hadamard each qubit
	for i in range(n):
		p += H(i)
	
	#setup inputs to multi-bit gates
	num_arr = list(range(n))
	
	#2. apply G = -(H^n)zo(H^n)zf k times
	k = calc_lim(a,n)

	for i in range(k):
		#Apply zf
		p += z_f(*num_arr)
		#Hadamard each qubit
		for i in range(n):
			p += H(i)
		#apply z0
		p += z_0(*num_arr)
		#-Hadamard each qubit
		for i in range(n):
			p += H(i)
			p += minus_gate(i)
	structure = "%dq-qvm" % (n)
	qc = get_qc(structure)
	#qvm.compiler.client.timeout = 1000000


	result = qc.run_and_measure(p, trials=num_trials)
	#measure all qubits and return them as a
	a = []
	for value in result.values():
		a.append(value[0])

	return a
		
	
if __name__== '__main__':
	num_inputs = len(sys.argv)
	inputs = sys.argv 
	#assume f and all inputs are given on CLI, even though this is unrealistic and poor practice
	print("inputs: ")
	print(sys.argv)
	print("output: ")
	n = 2 
	a = 1
	all = all_f(n,a)
	for func in all:
		print(func)
		z0 = create_z0(n)
		zf = create_zf(func,n)
		z_0 = DefGate("Z0", z0)
		z_f = DefGate("ZF", zf)
		result = grover(z_0,z_f,n,a)
		print(result)

		
