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
import time
import matplotlib.pyplot as plt

def convert_n_bit_string_to_int(x,n):
	"""
	inputs: x - array of ints that represents an unsigned bit string
			n is the number of bits
	
	outputs: integer corresponding to x's unsigned value
	"""
		integ = 0
		for idx, val in enumerate(x):
			integ += val*2^(n-idx-1)
		return int(integ)


def check_correctness(func,result):
	"""
	inputs: func - np array of ints that represnts a function mapping with the index as input
			result - np array of ints that represents an unsigned bit string
	
	outputs: bool if the index that corresponds to result is 1, then f(result) = 1
	"""
	x = convert_n_bit_string_to_int(result)
	return (func[x] == 1)

def create_minus_gate(n):
	"""
	inputs: n (int) - number of qubits
	
	outputs: minus (np array of ints) - NxN matrix with -1 on the diagonal
	"""
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
	"""
	inputs: int n = number of qubits, int a = number of items that return 1 for function
	
	outputs: int k = number of iterations to run grovers algorithm
	"""
	N = 2**n
	#theta = np.arcsin(a/N)
	theta = a/np.sqrt(N)
	#heuristically determined if n<4 you overshoot if you round, and if n>4 you undershoot if you floor
	#didnt test past n=6 as each trial took non-negligible time to compile and run
	if n < 4:
		k = math.floor(abs(((np.pi)/(4*theta)) - 0.5))
	else:
		k = math.round(abs(((np.pi)/(4*theta)) - 0.5))
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
	ro = p.declare('ro', 'BIT', n)

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
		#measure each qubit
		for qubit in range(n):
			p += MEASURE(qubit, ro[qubit])
			
	#setup quantum computer
	structure = "%dq-qvm" %(n)
	qc = get_qc(structure)
	qc.compiler.client.timeout = 60000
	
	print("Starting compilation")
	start = time.time()
	executable = qc.compile(p)
	end = time.time()
	compile_time = int((end - start) * 1000)
	print("Compilation finished, time: %d ms"%compile_time)
	
	print("Starting quantum computer")
	start = time.time()
	results = qc.run(executable)
	end = time.time()
	run_time = int((end - start) * 1000)
	print("Quantum computer finished, Run-time: %d ms"%run_time)
	
	y = [ measurement for measurement in results[0] ]



	return y, compile_time, run_time

#test code	
class test_harness(unittest.TestCase):
	
	def test_correct(self):
		for a in range(1,3): 
			for n in range(1,5):
				all = all_f(n,a)
				result = True
				for func in all:
					z0 = create_z0(n)
					zf = create_zf(func,n)
					z_0 = DefGate("Z0", z0)
					z_f = DefGate("ZF", zf)
					result_string, compile_time, run_time = grover(z_0,z_f,n,a)
					result =  result and check_correctness(func,result_string)
				self.assertTrue( result )
		


if __name__== '__main__':
	num_inputs = len(sys.argv)
	inputs = sys.argv 
	
	

	#benchmarking code
	time_out_val = 10000
	#set qubit range
	n_min = 1
	n_max = 6
	n_list = list(range(n_min,n_max+1))
	#num_times increases the number of trials time averaged over
	#set equal to 1 as I average over all the possible functions for a given n
	num_times = 1
	compile_times = np.zeros([(n_max-n_min)+1, num_times])
	compute_times = np.zeros([(n_max-n_min)+1, num_times])

	#compute average compile and compute time for all functions for a range of n values and a given 'a'
	for ind, n in enumerate(n_list):
		for iter_ind in range(num_times):
			a = 1
			all = all_f(n,a)
			avg_compute = 0
			avg_compile = 0
			for func in all:
				print(func)
				z0 = create_z0(n)
				zf = create_zf(func,n)
				z_0 = DefGate("Z0", z0)
				z_f = DefGate("ZF", zf)
				result, compile_time, run_time = grover(z_0,z_f,n,a)
				avg_compile += compile_time
				avg_compute += run_time
				print(result)
				#check_correctness(func,result)
			compute_times[ind, iter_ind] = avg_compute/len(all)
			compile_times[ind, iter_ind] = avg_compile/len(all)
			
	#testing for uf computation time at a fixed n
	input_list = list(range(64)) #64 is (4 qubits choose 2) -> 2^4 choose 2 -> number of possible functions with a=2
	uf_compute_times = np.zeros([64,num_times])
	for iter_ind in range(num_times):
		a = 1
		n = 4
		all = all_f(n,a)
		for ind,func in enumerate(all):
			print(func)
			z0 = create_z0(n)
			zf = create_zf(func,n)
			z_0 = DefGate("Z0", z0)
			z_f = DefGate("ZF", zf)
			result, compile_time, run_time = grover(z_0,z_f,n,a)
			print(result)
			uf_compute_times[ind, iter_ind] = run_time
	
	#%% Save the data

	np.savez('grover_benchmarking.npz', n_list = n_list,compile_times = compile_times,compute_times = compute_times)
	np.savez('grover_uf.npz', input_list = input_list,uf_compute_times=uf_compute_times)
	   
	#%% Load data

	data = np.load('grover_benchmarking.npz')
	n_list = data['n_list']
	compilation_times = data['compile_times']
	computation_times = data['compute_times']
	uf_data = np.load('grover_uf.npz')
	input_list = uf_data['input_list']
	uf_compute_times = uf_data['uf_compute_times']

	#%%
	avg_time_compile = np.sum(compilation_times,1)/np.shape(compilation_times)[1]
	avg_time_compute = np.sum(computation_times,1)/np.shape(computation_times)[1]

	#%% Plot and save 

	#plot compilation time vs n
	plt.rcParams["font.family"] = "serif"
	fig = plt.figure(figsize=(16,10))

	z = np.polyfit(n_list, avg_time_compile, 10)
	p = np.poly1d(z)

	plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
	plt.plot(n_list, avg_time_compile, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
	plt.title('Compilation time scaling for Grovers algorithm, a = 1', fontsize=25)
	plt.xlabel('n (bit string length)',fontsize=20)
	plt.ylabel('Average time of compilation (ms)',fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)


	fig.savefig('Figures/grover_compile_a1.png', bbox_inches='tight')
	
	
	#plot computation time vs n

	fig = plt.figure(figsize=(16,10))

	z = np.polyfit(n_list, avg_time_compute, 10)
	p = np.poly1d(z)

	plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
	plt.plot(n_list, avg_time_compute, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
	plt.title('Compilation time scaling for Grovers algorithm, a = 1', fontsize=25)
	plt.xlabel('n (bit string length)',fontsize=20)
	plt.ylabel('Average time of computation (ms)',fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)


	fig.savefig('Figures/grover_compute_a1.png', bbox_inches='tight')

	
	#plot computation time vs uf for n= 4

	fig = plt.figure(figsize=(16,10))

	
	plt.plot(input_list, uf_compute_times, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
	plt.title('Uf time scaling for Grovers algorithm, n=4, a = 1', fontsize=25)
	plt.xlabel('n (bit string length)',fontsize=20)
	plt.ylabel('Average time of computation (ms)',fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	fig.savefig('Figures/grover_compute_uf.png', bbox_inches='tight')