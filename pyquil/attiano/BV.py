#Bernstein-Vazirani Algorithm
import numpy as np
import inspect
import unittest
from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate

num_trials = 10

def convert_int_to_n_bit_string(integ, n):
#assume unsigned int value & representation
	attempt_input = [0] * n
	for i in range(n):
		attempt_input[i] = integ/(2^n-i)
		integ = integ%(2^n-i)
		
	return attempt_input

def convert_n_bit_string_to_int(x,n):
#assume unsigned int value & representation
		integ = 0
		for idx, val in enumerate(x):
			integ += val*2^(n-idx-1)
		return integ

def f_to_uf(f):
	n = len(inspect.signature(f).parameters)
	b = f([0]*n)
	uf = f
	return uf,n,b
				
def BV(U_f,n,b):
	UF1 = U_f.get_constructor()
	p = Program()
	p += U_f
	#make last qubit 1
	p += X(n)
	#Hadamard each qubit
	for i in range(n+1):
		p += H(i)
	num_arr = list(range(n+1))
	#apply U_f
	p += UF1(*num_arr)
	#Hadamard each qubit
	for i in range(n+1):
		p += H(i)
	qc = get_qc('9q-square-qvm')

	result = qc.run_and_measure(p, trials=num_trials)
	#measure all qubits and return them as a
	a = []
	for value in result.values():
		a.append(value[0])
	print(a)
	return a, b

class test_harness(unittest.TestCase):
	
	def test0(self):
		uf1 = np.array([[ 1, 0, 0, 0],
						[ 0, 1, 0, 0],
						[ 0, 0, 1, 0],
						[ 0, 0, 0, 1]])
		gate_def = DefGate("UF1", uf1)
		self.assertTrue( BV(gate_def,1,0) == [1]*num_trials )
	
if __name__ == '__main__':
	uf1 = np.array([[ 1, 0, 0, 0],
					[ 0, 1, 0, 0],
					[ 0, 0, 1, 0],
					[ 0, 0, 0, 1]])
	gate_def = DefGate("UF1", uf1)
	result = BV(gate_def,1,0)