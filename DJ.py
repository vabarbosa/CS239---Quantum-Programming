import numpy as np
import unittest
from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate


def f_to_uf(f):
	
	return uf,n
				
def DJ(U_f,n):
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
	qc = get_qc('9q-square-qvm')

	result = qc.run_and_measure(p, trials=10)

	print(result[0])

def test_harness(unittest.TestCase):
	uf1 = np.array([[ 1, 0, 0, 0],
					[ 0, 1, 0, 0],
					[ 0, 0, 0, 1],
					[ 0, 0, 1, 0]])
	gate_def = DefGate("UF1", uf1)
	DJ(gate_def,1)