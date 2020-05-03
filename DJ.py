#Deutsch-Jozsa Algorithm
import numpy as np
import math
import unittest
from itertools import permutations 
from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate

NUM_TRIALS = 10


class sizeError(Exception):
	def __init__(self, message):
		self.message = message

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

		
class create_f(object):
	def __init__(self):
		self.funcs = []
		self.uf = []
	
	def create(self, n):
		#make constant functions
		def f_const1(x):
			return 0
		def f_const2(x):
			return 1
		#all possible permutations of n balanced functions
		balanced = (n/2)*[0]
		balanced.extend((n/2)*[1])
		for permutation in permutations(balanced):
			
			def f_bal(x):
				#every possible balanced function.. how to build them systematically ?
				if( convert_n_bit_string_to_int(x) > convert_n_bit_string_to_int(permutation)):
					return 0
				else:
					return 1
			self.funcs.append(f_bal)			
		self.funcs.append(f_const1)
		self.funcs.append(f_const2)
		
			

	def f_to_uf(self):
	#empty implementation right now... is this needed?
		for func in self.func:
			return func
				
def DJ(gate_def,n):
	if( n <= 9 ) :
		qc = get_qc('9q-square-qvm')
	else:
		raise sizeError('N > MAX_SIZE = 9')
	UF1 = gate_def.get_constructor()
	p = Program()
	p += gate_def
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

	result = qc.run_and_measure(p, trials=NUM_TRIALS)
	result_sum = 0
	for i in range(n):
		result_sum+=sum(result[i])
	#result_sum will be 0 if over every trial we measured every qubit to be 0, otherwise we measure 1 from the qubit
	#if we measure all qubits to be 0 then the function is constant by the Deutsch-Jozsa proof, otherwise it is balanced
	#sum = 0 if constant o.w. sum != 0
	
	#return 1 if constant 0 if balanced

	return 1 if (result_sum == 0) else 0



class test_harness(unittest.TestCase):
		
	def test0(self):
		#constant
		UF1 = np.array([[ 1, 0, 0, 0],
						[ 0, 1, 0, 0],
						[ 0, 0, 1, 0],
						[ 0, 0, 0, 1]])
		gate_def = DefGate("UF1", UF1)
		self.assertTrue( DJ(gate_def,1) == 1 )
		
	def test1(self):
		#balanced
		UF1 = np.array([[ 1, 0, 0, 0],
						[ 0, 1, 0, 0],
						[ 0, 0, 0, 1],
						[ 0, 0, 1, 0]])
		gate_def = DefGate("UF1", UF1)
		self.assertTrue( DJ(gate_def,1) ==  0 )
		
	def test2(self):
		#balanced
		UF1 = np.array([[ 0, 1, 0, 0],
						[ 1, 0, 0, 0],
						[ 0, 0, 1, 0],
						[ 0, 0, 0, 1]])
		gate_def = DefGate("UF1", UF1)
		self.assertTrue( DJ(gate_def,1) == 0 )
		
	def test3(self):
		#constant
		UF1 = np.array([[ 0, 1, 0, 0],
						[ 1, 0, 0, 0],
						[ 0, 0, 0, 1],
						[ 0, 0, 1, 0]])
		gate_def = DefGate("UF1", UF1)
		self.assertTrue( DJ(gate_def,1) == 1 )
		
	def test4(self):
		#constant
		UF1 = np.array([[ 1, 0, 0, 0, 0, 0, 0, 0],
						[ 0, 1, 0, 0, 0, 0, 0, 0],
						[ 0, 0, 1, 0, 0, 0, 0, 0],
						[ 0, 0, 0, 1, 0, 0, 0, 0],
						[ 0, 0, 0, 0, 1, 0, 0, 0],
						[ 0, 0, 0, 0, 0, 1, 0, 0],
						[ 0, 0, 0, 0, 0, 0, 1, 0],
						[ 0, 0, 0, 0, 0, 0, 0, 1]])
		gate_def = DefGate("UF1", UF1)
		self.assertTrue( DJ(gate_def,2) == 1 )

	
if __name__ == '__main__':
    unittest.main()