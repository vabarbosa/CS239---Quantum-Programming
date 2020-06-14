import sys
import inspect
from qiskit import(
  QuantumCircuit,
  execute,
  Aer,
  IBMQ)
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from itertools import combinations
import time, random
import numpy as np
import math
from operator import add
import operator
import matplotlib.pyplot as plt
from qiskit.compiler import transpile, assemble
import time, random
from collections.abc import Iterable

class grover:

	def run_grover(self,f,n,a):
		"""
		inputs: f - the oracle function
				n - integer defining number of qubits
				a - integer defining number of inputs to f that ==1

		outputs: result - ket{x} s.t. f(x) = 1
				run_time - float number of seconds it took to run quantum simulator
				numerical_prob - probability of success based on qasm_simulator
				theo_prob - float corresponding to theoretical probability of success
				
		"""

		gate = self.create_G(f,n)
		G = Operator(gate)
		circuit, theo_prob = self.create_circuit(G,n,a)
		result, run_time, numerical_prob = self.execute_grover(circuit,n,f)
		return result, run_time, numerical_prob, theo_prob
	
	def create_circuit(self,G,n,a):
	
		""" inputs: G - Operator of matrix object from qiskit - custom gate for grovers circuit n,a
					n - int defining number of qubits
					a - int defining number of x s.t f(x)==1
			outputs: qiskit circuit object corresponding to to grovers algorithm with n,a
					 theo_prob - float corresponding to theoretical probability of success
		""" 
	
		circuit = QuantumCircuit(n, n)

		#1. Hadamard each qubit
		for i in range(n):
			circuit.h(i)

		#setup inputs to multi-bit gates
		num_arr = list(range(n))

		#2. apply G = -(H^n)zo(H^n)zf k times
		k , theo_prob = self.calc_lim(a,n)
		for iter in range(k):
			# circuit.append(G,reversed(range(n)))
			circuit.append(G,range(n))
		
		#measure each qubit
		circuit.measure(range(n), range(n))
		
		return circuit, theo_prob
		
			

	def convert_int_to_n_bit_string(self,integer, n):
		"""
		inputs: integer corresponding to x's unsigned value
				n is the number of bits

		outputs: array of ints that represents an unsigned bit string
		"""
		return [ int(char) for char in bin(integer)[2:].zfill(n) ]

	def convert_n_bit_string_to_int(self,x,n):
		"""
		inputs: x - array of ints that represents an unsigned bit string
				n is the number of bits

		outputs: integer corresponding to x's unsigned value
		"""
		return int("".join([str(num) for num in x]), 2)


	def check_correctness(self,func,result,n):
		"""
		input: func - np array of ints that represnts a function mapping with the index as input
			   result - np array of ints that represents an unsigned bit string
			   n is the number of bits
		
		output: bool if the index that corresponds to result is 1, then f(result) = 1
		"""
		if(str(type(func))!="<class 'list'>" and str(type(func))!="<class 'numpy.ndarray'>" and str(type(func))!="<class 'function'>"):
			raise TypeError('input for func not a list, numpy array or a function')
		
		
		x = self.convert_n_bit_string_to_int(result,n)
		
		
		if(str(type(func))=="<class 'function'>"):
			return (func(*result) == 1)
		else:
			return (func[x] == 1)


	def all_f(self,n,a=1):
		""" 
		input: int n = number of qubits, int a = number of items that return 1 for function
		
		output: list of functions (arrays that can be indexed with 'x' for function value 'f(x)'
		""" 
		if(str(type(n))!="<class 'int'>"):
			raise TypeError('input for n non-integer')	
		if(str(type(a))!="<class 'int'>"):
			raise TypeError('input for a non-integer')
		N = 2**n
		func_inputs = range(N)
		comb = combinations(func_inputs, 1)
		func_list = []

		for combination in comb:
			out_list = np.zeros(N)
			for i in combination:
				out_list[i] = 1
			func_list.append(out_list)
			
		return func_list

	def calc_lim(self, a,n):
		"""
		input: int n = number of qubits, int a = number of items that return 1 for function

		output: int k = number of iterations to run grovers algorithm
		"""
		if(str(type(n))!="<class 'int'>"):
			raise TypeError('input for n non-integer')	
		if(str(type(a))!="<class 'int'>"):
			raise TypeError('input for a non-integer')

		N = 2**n
		theta = np.arcsin(np.sqrt(a/N))
		
		k_approx = ((np.pi/(4*theta)) - (1/2))
		
		k_arr = np.array([np.ceil(k_approx),np.floor(k_approx)])
		
		prob_arr = np.sin((2*k_arr + 1) * theta ) ** 2
		max = np.argmax(prob_arr)
		
		return int(k_arr[max]), prob_arr[max]
		
	def H_tensored(self,n):
		""" Returns the matrix corresponding to the tensor product of n number of Hadamard gates
		Args:
			n: Integer
		Returns:
			ndarray of size [2**n, 2**n]
		"""
		H = 1/np.sqrt(2) * np.array([[-1,1],[1,1]])
		n = n - 1
		H_n = H
		while n > 0:
			H_n = np.kron(H_n, H)
			n -= 1
		return H_n
		
	def reverse_bit_int(self,i, n):
		"""
		inputs: i - integer
		outputs: integer corresponding to i's unsigned bit string representation flipped e.g 4 => 0100 => 0010 => 2
		"""
		bit_string = self.convert_int_to_n_bit_string(i,n)
	
		bit_string = bit_string[::-1]
		number = self.convert_n_bit_string_to_int(bit_string,n)
		return number
		
    
		
	def create_G(self,f,n):
		"""
		Given a function f:{0,1}^n ---> {0,1}, creates and returns the corresponding Grover operator G (unitary matrix)
		Args:
			f: ndarray of length 2**n, consisting of integers 0 and 1
		Returns:
			G: ndarray of size [2**(n+1), 2**(n+1)], representing a unitary matrix.
		"""
		N = 2**n
		
		G = np.zeros([N,N], dtype = complex)
		Zf = np.eye(N)
		for i in range(N):
			j = self.reverse_bit_int(i,n)
			Zf[j][j]*=((-1)**f[i])
		Z0 = np.eye(N)
		Z0[N-1,N-1] = -1

		
		
		H_n = self.H_tensored(n)
		G = - H_n @ Z0 @ H_n @ Zf
		return G
		
	

	def execute_grover(self,circuit,n,f,num_shots = 1000, backend = 'qasm_simulator'):
		
		"""
		inputs: 
			circuit - 
			n - int defining number of qubits
			f - np array of ints 0/1, with 2^n indicies for f[x]=0/1
		outputs: results of 1000 shots of grovers algorithm and numerical probability of success given f, n
		"""
				
		#setup quantum computer
		simulator = Aer.get_backend(backend)
		
		#get run-time
		start = time.time()
		job = execute(circuit, simulator, shots=num_shots)
		end = time.time()
		run_time = int((end - start) * 1000)
		
		#get results
		result = job.result()
		counts = result.get_counts(circuit)
		
		#get number of results that correctly identified f(x)==1
		total = 0
		for potential_soln in counts:
			intermediate = [int(i) for i in potential_soln][::-1]
			check = self.check_correctness(f,intermediate,n)
			if(check):
				total += counts[potential_soln]
		
		y = list(max(counts, key = lambda key: counts[key]))
		numerical_prob=(total/num_shots)

		return [int(char) for char in y][::-1], run_time, numerical_prob

API_TOKEN = '24cfb5c482acb6a69ae337a1170007421c647cb4220ab11c94a59af31adc8afaec284b8ff47d60654fec857f42a684400ef67bf549da24a2db38b5af02a11554'

class Program:
	def run(self, f, backend=None, num_shots=1000):
		"""
		High level function to run Berenstein Vazarani. This use f to generate Uf, then builds a circuit, runs it, and measures and returns if the function is balanced.
		f is assumed to be balanced or constant. If it measures 0^n, f is constant
		Args: 
		backend: backend object to run the circuit on
		f: input function represented by 2^n array where each entry f[i] represents f applied to that row indexf(i)
		num_shots: number of times to measure
		Returns:
		time_elapsed, answer: time taken to compile and measure. 0 if f is balanced or 1 o.w.
		"""
		if backend == None:
			provider = IBMQ.enable_account(API_TOKEN)
			backend = provider.backends.ibmq_burlington
			
		n = int(math.log(len(f), 2))        
		# create a gate from f
		uf = self.create_uf(n, f)
		# build the circuit
		circuit = self.build_circuit(n, uf)
		# measure the circuit
		counts, assembly_time, run_time = self.measure(circuit, backend, num_shots=num_shots)
		# grab the measurement
		y = self.evaluate(counts)

		return counts, y, assembly_time, run_time
		
	def build_circuit(self, n, uf, flag=False, custom_uf=None):
		"""
		Assemble the circuit. Create a circuit of n+1 qubits mapped to n bits. Flip helper bit to 1. Apply H to all. Apply Uf to all. Then H to all the n qubits and measure.
		Args: 
		n: number of bits - arity of f
		uf: Uf gate representation of f
		num_shots: number of times to measure
		Returns:
		qiskit circuit implementing BV
		"""
		g = grover()
		circ, theo = g.create_circuit(uf,n,1)
		return circ
		
	def measure(self, circuit, backend, num_shots=1000):
		"""
		Measure the circuit and return the output
		Args: 
		circuit: the qiskit circuit
		backend: what to run the ciruit it on
		num_shots: number of times to measure
		Returns:
		counts: map of measurement to count
		"""
		if not isinstance(circuit, QuantumCircuit):
			raise ValueError("circuit must be a QuantumCircuit")
		if not isinstance(num_shots, int):
			raise ValueError("num_shots must be of type int")
		start = time.time()
		qobj = assemble(transpile(circuit, backend=backend, optimization_level=3), backend=backend, shots=num_shots)
		end = time.time()
		assembly_time = int((end - start) * 1000)
		job = backend.run(qobj)
		job = backend.retrieve_job(job.job_id())
		while job._api_status != 'COMPLETED':
			if "ERROR" in job._api_status:
				print("Circ depth: %d" % circuit.depth())
				raise ValueError("ERROR: Job status is %s: " % job._api_status)
			print('Current job status: %s' % job._api_status)
			time.sleep(5)
			job = backend.retrieve_job(job.job_id())
		return job.result().get_counts(circuit), assembly_time/1000.0, job.result().time_taken
	def evaluate(self, counts, reverse=True):
		"""
		Evaluate the counts. Returns the most frequent item. If there are two it returns the larger.
		Args: 
		counts: map of measurement to count
		Retuns:
		result: CONSTANT (0) if 0^n was m
		"""
		if not isinstance(counts, dict):
			raise ValueError("counts must be of type dict")
		max_key = None
		for key in counts:
			if max_key == None or (counts[key] > counts[max_key]):
				max_key = key
		if reverse == True:
			return [int(char) for char in max_key][::-1]
		else:
			return [int(char) for char in max_key]
		
	def create_uf(self, n, f):
		"""
		Wrapper function to create Uf. Build the matrix and make it a gate
		Args: 
		n: number of bits - arity of f
		f: input function represented by 2^n array where each entry f[i] represents f applied to that row indexf(i)
		Returns:
		gate: gate representing uf
		"""
		
		if not isinstance(n, int):
			raise ValueError("n must be an int")
		matrix = self.build_matrix(n, f)
		uf = Operator(matrix)
		return uf
	def build_matrix(self, n, f):
		"""
		Convert f to a 2^(n+1) x 2^(n+1) matrix. Care must be taken since qiskit is little endian.
		for all x in {0, 1)^n, b in 0,1, calculate x (f(x) + b), place a 1 at matrix[2^(n+1) - xb][2^(n+1) - x (f(x) + b)]
		Args: 
		n: number of bits - arity of f
		f: input function represented by 2^n array where each entry f[i] represents f applied to that row indexf(i)
		Returns:
		matrix: matrix representing the uf gate
		"""
		g = grover()
		matrix = g.create_G(f,n)
		return matrix
	def to_int(self, bit_string):
		"""
		Convert a bit string (list of ints) to an int
		Args: 
		bit_string: list of 1s or 0s
		Returns:
		num: integer of binary represantation
		"""
		
		return int("".join([str(num) for num in bit_string]), 2)
	def bit_string(self, i, n):
		"""
		Convert an int to a bit string of length n
		Args: 
		i: number to convert
		n: desired length
		Returns:
		binary representation of i
		"""
		
		return [ int(char) for char in bin(i)[2:].zfill(n) ]
	def generate_functions(self, n, fully_random=False, random_trials=100):
		"""
		Generate test functions of input length n. If fully random is false, generate all possible constant and balanced function.
		For n > 3 or fully_random = True, generate m random trials where m = random_trials
		Args: 
		n: bit length
		fully_random: select random trials for n<=3
		random_trials: number of random functions to generate when selecting at random.
		Returns:
		list of tuples, first item is BALANCED/CONSTANT and second is the function
		"""
		
		N = 2**n
		func_inputs = range(N)
		comb = combinations(func_inputs, 1)
		func_list = []

		for combination in comb:
			out_list = np.zeros(N)
			for i in combination:
				out_list[i] = 1
			func_list.append((self.bit_string(i,n),out_list))
		if(not fully_random or n<3):
			return func_list
		else:
			return func_list[0:random_trials]

	def run_tests(self, max_bits):
		"""
		Test for 1 to max_bits. For 1 to 3, test all possible input functions. After that, select 100 at random.
		Args: 
		max_bits: number of bits to test up to
		"""
		
		for n in range(1, max_bits + 1):
			print("Running exhaustive tests for %d bits" % n)
			functions = self.generate_functions(n, True, 100)
			for expected, f in functions:
				counts, calculated, assembly_time, run_time = self.run(f)
				if calculated != expected:
					print("Test failed: (%s). Got: %s. Expected %s." % (f, calculated, expected))
				print("Test passed. Got: %s. Expected %s. Tooks %d ms." % (calculated, expected, run_time))
			print("%d tests passed." % (len(functions)))

	def build_histogram(self):
		provider = IBMQ.enable_account(API_TOKEN)
		backend = provider.backends.ibmq_burlington
		
		for n in range(1, 5):
			expected, f = self.generate_functions(n, True, 1)[0]
			while expected[0] == [0]*n:
				print("getting new f", f)
				expected, f = self.generate_functions(n, True, 1)[0] 
			counts, y, _, _ = self.run(f, backend, num_shots=1000)
			
			plt.rcParams["font.family"] = "serif"
			fig = plt.figure(figsize=(16,10))

			answers = [key[::-1] for key in counts.keys()]
			answers.sort()
			frequencies = [ counts[answer[::-1]] for answer in answers ]

			# a = "".join([str(i) for i in expected[0]])
			print(expected, counts)
			print(y)
			plt.bar(range(len(counts)), frequencies, align='center', tick_label=answers)
			plt.title('Results for n = %d (grovers algorithm)' % (n), fontsize=25)
			plt.xlabel('Measurement',fontsize=20)
			plt.ylabel('Frequency of occurence',fontsize=20)
			plt.xticks(fontsize=15)
			plt.yticks(fontsize=15)
			
			
			fig.savefig('figures/g_counts_histogram_%d.png' % (n), bbox_inches='tight')

	def collect_data(self, max_bits, num_trials, npz_filename):
		"""
		Generate benchmarking data.
		from 1 to max_bits, save the time for num_trials.
		Run 100 trials for n = 6 random ufs and collect time
		Args: 
		max_bits: number of bits to test up to
		num_trials: how many trials for each n
		npz_filename: where tp save output
		"""
		provider = IBMQ.enable_account(API_TOKEN)
		backend = provider.backends.ibmq_essex
		
		run_time_arr = np.zeros([max_bits, num_trials])
		assemble_time_arr = np.zeros([max_bits, num_trials])
		correct_arr = np.zeros([max_bits, num_trials])        

		# num_shots = 8192
		# for n in range(1, max_bits+1):
			# print("Starting benchmarking for n = %d" % (n))
			# functions = self.generate_functions(n, True, num_trials)
			# trial = 0
			# for trial in range(min(num_trials,len(functions))):

				# expected, f = functions[trial]
				# try:
					# counts, answer, assemble_time, run_time = self.run(f, backend, num_shots=num_shots)
				# except ValueError:
					# print("Error")
					# continue
				# print("Test finished for %s function. Got %s." % (expected, answer))
			
				# run_time_arr[n-1, trial] = run_time / num_shots
				# assemble_time_arr[n-1, trial] = assemble_time
				# print("HERE")
				# print(answer, expected)
				# if answer == expected:
					# correct_arr[n-1, trial] = 1.0
				# else:
					# correct_arr[n-1, trial] = 0.0

		num_times = 100
		variance_arr = []
		n = 3
		print("Starting benchmarking for different Ufs for n = %d" % (n))
		functions = self.generate_functions(n, True, num_times)
		for trial in range(min(num_trials,len(functions))):
			expected, f = functions[trial]
			counts, answer, assemble_time, run_time = self.run(f, backend, num_shots=num_times)
			if(run_time != 0):
				variance_arr.append(run_time)
			# try:
				# counts, answer, assemble_time, run_time = self.run(f, backend, num_shots=num_shots)
			# except:
				# print("Error")
				# continue
			print("Test finished for %s function. Got %s." % (expected, answer))
			
			# variance_arr[trial] = run_time
		np.savez(npz_filename, run_time_arr = run_time_arr, assemble_time_arr = assemble_time_arr, n_list = list(range(1, max_bits+1)), n = n, num_times = num_times, variance_arr = variance_arr, correct_arr = correct_arr)
	def plot_data(self, npz_filename):
		"""
		Plot benchmarking data. Plot average run time for each n and variance data for 100 random Ufs.
		Args: 
		npz_filename: where tp load data
		"""
		# #taken directly from ibmq quantum experience online
		# fig = plt.figure(figsize=(16,10))
		# # counts = {'0000': 60, '0001': 64.5, '0010': 61.5, '0100': 63.5, '0101': 62.5, '0110': 62.5, '0111': 59, '1000': 66, '1001': 57, '1010': 62.5, '1011': 68, '1100': 55, '1110': 62.5, "1111":70}
		# # frequencies = [60,64.5,61.5,63.5,62.5,62.5,59,66,57,62.5,68,55,62.5,70]
		# answers = [key for key in counts.keys()]
		# answers.sort()
		# plt.bar(range(len(counts)), frequencies, align='center', tick_label=answers)
		# plt.title('Results for n = 4 (grovers algorithm)', fontsize=25)
		# plt.xlabel('Measurement',fontsize=20)
		# plt.ylabel('Frequency of occurence',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)
		
		
		# fig.savefig('figures/g_counts_histogram_4.png', bbox_inches='tight')
		
		# data = np.load(npz_filename)

		# run_time_arr = data['run_time_arr']
		# n_list = data['n_list']
		# n_list = [1,2,3,4]

		# avg_time = np.sum(run_time_arr,1)/np.shape(run_time_arr)[1]
		# avg_time = [10.8,11.5,14.5,31.4]

		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		# z = np.polyfit(n_list, avg_time, 10)
		# p = np.poly1d(z)

		# plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
		# plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M')
		# plt.title('Execution time scaling for Grovers algorithm', fontsize=25)
		# plt.xlabel('n (bit string length)',fontsize=20)
		# plt.ylabel('Average time of execution (s)',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)

		# fig.savefig('figures/g_run_time.png', bbox_inches='tight')

		# num_times = data['num_times']
		# variance_arr = data['variance_arr']
		# n = data['n']

		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		# assemble_time_arr = data['assemble_time_arr']
		# n_list = data['n_list']
		# # n_list = [1,2,3,4]

		# avg_time = np.sum(assemble_time_arr,1)/np.shape(assemble_time_arr)[1]
		# # avg_time = [4.2,4.5,7.2,14.9] #taken directly from imbq experience

		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		# z = np.polyfit(n_list, avg_time, 10)
		# p = np.poly1d(z)

		# plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
		# plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M')
		# plt.title('Compilation time scaling for Grovers algorithm', fontsize=25)
		# plt.xlabel('n (bit string length)',fontsize=20)
		# plt.ylabel('Average time of execution (s)',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)

		# fig.savefig('figures/g_assemble_time.png', bbox_inches='tight')

		# correct_arr = data['correct_arr']
		# n_list = data['n_list']
		# # n_list = [1,2,3,4]

		# avg_correct = np.sum(correct_arr,1)/np.shape(correct_arr)[1]
		# # avg_correct = [55,96,20,1.5]

		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		# z = np.polyfit(n_list, avg_correct, 10)
		# p = np.poly1d(z)

		# plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
		# plt.plot(n_list, avg_correct, ls = '', markersize = 15, marker = '.',label = 'M')
		# plt.title('Percentage correct for grovers algorithm', fontsize=25)
		# plt.xlabel('n (bit string length)',fontsize=20)
		# plt.ylabel('Percentage correct',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)

		# fig.savefig('figures/g_correct.png', bbox_inches='tight')

		
		# num_times = data['num_times']
		# variance_arr = data['variance_arr']
		# n = data['n']

		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		

		# plt.hist(variance_arr)
		# plt.title('Dependence of execution time on $U_f$ (grovers algorithm)', fontsize=25)
		# plt.xlabel('Execution time (s)',fontsize=20)
		# plt.ylabel('Frequency of occurence',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)


		# fig.savefig('figures/g_hist.png', bbox_inches='tight')
		
		
		#Obtain transpiled depths of circuits
		# list_of_depths = [2,5,309,2984]
		# n_list = [1,2,3,4]
		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		# z = np.polyfit(n_list, list_of_depths, 10)
		# p = np.poly1d(z)

		# plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
		# plt.plot(n_list, list_of_depths, ls = '', markersize = 15, marker = '.',label = 'M')
		# plt.title('depth of circuit for grovers algorithm', fontsize=25)
		# plt.xlabel('n (bit string length)',fontsize=20)
		# plt.ylabel('Depth',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)

		# fig.savefig('figures/g_depth.png', bbox_inches='tight')
		
					
	# def get_uf(self, n, f):
		# """
		# get simplified uf for a given f by solving for a, b
		# n: arity of input function
		# f: input function
		# returns: uf circuit
		# """
		# num_qubits = n + 1
		# circuit = QuantumCircuit(num_qubits, n)
		# if b == 1:
			# circuit.x(num_qubits-1)
			# circuit.barrier()
		# for index in range(0, len(a)):
			# if a[index] == 1:
				# circuit.cx(n - 1 - index, num_qubits-1)
		# return circuit


	# def benchmark_hack(self, max_bits, num_trials, npz_filename):
		# provider = IBMQ.enable_account(API_TOKEN)
		# backend = provider.backends.ibmq_16_melbourne

		# run_time_arr = np.zeros([max_bits, num_trials])
		# correct_arr = np.zeros([max_bits, num_trials])
		# correct_percentage_arr = np.zeros([max_bits, num_trials])
		
		# num_shots = 8192
		# for n in range(1, max_bits+1):
			# print("Starting benchmarking customs for n = %d" % (n))
			# functions = self.generate_functions(n, True, num_trials)
			# jobs = []
			# for trial in range(num_trials):
				# expected, f = functions[trial]
				# uf = self.get_uf(n, f)
				# circuit = self.build_circuit(n, None, True, uf)
				# t = transpile(circuit, backend=backend, optimization_level=3)
				# print("Queing job for trial %d: %s. circuit depth %d." % (trial, expected, t.depth()))
				# qobj = assemble(t, backend=backend, shots=num_shots)
				# jobs.append((backend.run(qobj), circuit))
				

			# for trial in range(num_trials):
				# expected, f = functions[trial]                
				# job, circuit = jobs[trial]
				# job = backend.retrieve_job(job.job_id())
				# while job._api_status != 'COMPLETED':
					# if "ERROR" in job._api_status:
						# raise ValueError("Job status is %s: " % job._api_status)
					# print('Current job status for trial %d: %s' % (trial, job._api_status))
					# time.sleep(5)
					# job = backend.retrieve_job(job.job_id())
				# counts = job.result().get_counts(circuit)
				# run_time = job.result().time_taken
				# answer = self.evaluate(counts, False)
				# correct = 100 if answer == expected[0] else 0
				# s = "".join([str(num) for num in answer])
				# percentage = counts[s] / float(num_shots) * 100
				# print("Trial %s: expected %s. Got %s. Ran in %f. correct: %d. percentage %f" % (trial, expected, answer, run_time, correct, percentage))
				# print(counts)

				# run_time_arr[n-1, trial] = run_time / num_shots
				# correct_arr[n-1, trial] = correct
				# correct_percentage_arr[n-1, trial] = percentage

		# np.savez(npz_filename, run_time_arr = run_time_arr, n_list = list(range(1, max_bits+1)), n = n, correct_arr = correct_arr, correct_percentage_arr = correct_percentage_arr)
    
	# def plot_hack(self, npz_filename):
		# data = np.load(npz_filename)

		# run_time_arr = data['run_time_arr']
		# n_list = data['n_list']

		# avg_time = np.sum(run_time_arr,1)/np.shape(run_time_arr)[1]

		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		# z = np.polyfit(n_list, avg_time, 10)
		# p = np.poly1d(z)

		# plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
		# plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M')
		# plt.title('Execution time scaling for grover(custom circuit)', fontsize=25)
		# plt.xlabel('n (bit string length)',fontsize=20)
		# plt.ylabel('Average time of execution (s)',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)

		# fig.savefig('figures/g_run_time_custom.png', bbox_inches='tight')

		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		# correct_arr = data['correct_arr']
		# n_list = data['n_list']

		# avg_correct = np.sum(correct_arr,1)/np.shape(correct_arr)[1]
		
		# plt.bar(n_list, avg_correct, align='center')#, tick_label=answers)
		# plt.title('Percentage correct by majority. grovers algorithm (custom circuit)', fontsize=25)
		# plt.xlabel('n (bit string length)',fontsize=20)
		# plt.ylabel('Percentage Correct',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)
		
		# fig.savefig('figures/g_correct_custom.png', bbox_inches='tight')

		# correct_percentage_arr = data['correct_percentage_arr']
		# n_list = data['n_list']

		# avg_time = np.sum(correct_percentage_arr,1)/np.shape(correct_percentage_arr)[1]

		# plt.rcParams["font.family"] = "serif"
		# fig = plt.figure(figsize=(16,10))

		# plt.bar(n_list, avg_time, align='center')#, tick_label=answers)
		# plt.title('Percentage of correct trials. Grovers algorithm (custom circuit)', fontsize=25)
		# plt.xlabel('n (bit string length)',fontsize=20)
		# plt.ylabel('Percentage Correct',fontsize=20)
		# plt.xticks(fontsize=15)
		# plt.yticks(fontsize=15)
		
		# fig.savefig('figures/g_noise_custom.png', bbox_inches='tight')



if __name__== '__main__':
	num_inputs = len(sys.argv)
	inputs = sys.argv
	p = Program()
	# p.build_histogram()

	# p.run_tests(4)
	# p.collect_data(4, 50, "g.npz")
	# p.plot_data("g.npz")

	# g = grover()
	
	# #test correctness
	# min_n = 1
	# max_n = 6
	# max_a = 3
	# n_list = list(range(min_n,max_n+1))
	# a_list = list(range(1,max_a+1))
	# for n in n_list:
		# for a in a_list:
			# if(a>n):
				# break
			# all = g.all_f(n,a)
			# func = random.choice(all)
			# result, run_time, numerical_prob, theoretical_prob = g.run_grover(func,n,a)
			# print(" ")
			# print("N:%d, A:%d prob_success: %.4f"%(n,a,theoretical_prob))
			# print("         numerical_prob: %.4f"%numerical_prob)
	
	
	# #calculate computation time for n=6 a=1->6

	# min_n = 6
	# max_n = 6
	# max_a = 6
	# n_list = list(range(min_n,max_n+1))
	# a_list = list(range(1,max_a+1))
	# time_vs_a = []
	# for n in n_list:
		# for a in a_list:
			# if(a>n):
				# break
			# all = g.all_f(n,a)
			# func = random.choice(all)
			# result, run_time, numerical_prob, theoretical_prob = g.run_grover(func,n,a)
			# time_vs_a.append(run_time)
	
	
	# # #compute average compile and compute time for all functions for a range of n values and a given 'a'
	# a = 1
	# n_min = 1
	# n_max = 10
	# n_list = list(range(n_min,n_max+1))
	# num_times = 100
	# compute_times = np.zeros([(n_max-n_min)+1])
	# for ind, n in enumerate(n_list):
		# avg_compute = 0
		# for iter_ind in range(num_times):
			# all = g.all_f(n,a)
			# func = random.choice(all)
			# # print("---------------FUNCTION------------")
			# # print(func)
			# # print("---------------FUNCTION------------")
			# result, run_time = g.run_grover(func,n,a)
			# avg_compute += run_time
			# # print("---------------RESULT------------")
			# # print(result)
			# # print("---------------RESULT------------")
		# compute_times[ind] = avg_compute/num_times
		
	# print("compute times")
	# print(compute_times)
		
		
	# #testing for uf computation time at a fixed n
	# num_funcs = 16 #64 is (4 qubits choose 1) -> 2^4 choose 1 -> number of possible functions with a=2
	# input_list = list(range(num_funcs))
	# uf_compute_times = np.zeros([num_funcs,num_times])
	# for iter_ind in range(num_times):
		# a = 1
		# n = 4
		# all = g.all_f(n,a)
		# for ind,func in enumerate(all):
			# #print(func)
			# result, run_time = g.run_grover(func,n,a)
			# #print(result)
			# uf_compute_times[ind, iter_ind] = run_time
	
	# #%% Save the data

	# np.savez('grover_benchmarking.npz', n_list = n_list,compute_times = compute_times, time_vs_a = time_vs_a, a_list = a_list)
	# np.savez('grover_uf.npz', input_list = input_list,uf_compute_times=uf_compute_times)
	   
	# #%% Load data

	# data = np.load('grover_benchmarking.npz')
	# time_vs_a = data['time_vs_a']
	# a_list = data['a_list']
	# n_list = data['n_list']
	# computation_times = data['compute_times']
	# avg_time_compute = computation_times
	
	# uf_data = np.load('grover_uf.npz')
	# input_list = uf_data['input_list']
	# uf_compute_times = uf_data['uf_compute_times']
	
	
	# #%% Plot and save 

	#plot computation time vs n

	# fig = plt.figure(figsize=(16,10))

	# z = np.polyfit(n_list, avg_time_compute, 8)
	# p = np.poly1d(z)

	# plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
	# plt.plot(n_list, avg_time_compute, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
	# plt.title('Computation time scaling for Grovers algorithm, a = 1', fontsize=25)
	# plt.xlabel('n (bit string length)',fontsize=20)
	# plt.ylabel('Average time of computation (s)',fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)


	# fig.savefig('Figures/grover_computation_100trials.png', bbox_inches='tight')

	
	# #plot computation time vs uf for n= 4

	# fig = plt.figure(figsize=(16,10))

	
	# plt.hist(uf_compute_times)
	# plt.title('Dependence of execution time on $U_f$ (Grover algorithm)', fontsize=25)
	# plt.xlabel('Execution time (ms)',fontsize=20)
	# plt.ylabel('Frequency of occurence',fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)


	# fig.savefig('Figures/grover_hist.png', bbox_inches='tight')
	
	
	# #plot computation time vs a

	# fig = plt.figure(figsize=(16,10))

	# z = np.polyfit(a_list, time_vs_a, 5)
	# p = np.poly1d(z)

	# plt.plot(np.linspace(a_list[0], a_list[-1] + 0.1, 100), p(np.linspace(a_list[0], a_list[-1] + 0.1, 100)), ls = '-', color = 'r')
	# plt.plot(a_list, time_vs_a, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
	# plt.title('Computation time scaling for Grovers algorithm, n = 6, a=1->6', fontsize=25)
	# plt.xlabel('a',fontsize=20)
	# plt.ylabel('Average time of computation (s)',fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)


	# fig.savefig('Figures/grover_n3_va.png', bbox_inches='tight')