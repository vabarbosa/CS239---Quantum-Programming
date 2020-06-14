import sys
import inspect
import os
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.compiler import transpile, assemble
import qiskit
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from itertools import combinations
import time, random
import numpy as np
import math
from operator import add
import operator
import matplotlib.pyplot as plt
from datetime import datetime

# IBMQ.save_account('24cfb5c482acb6a69ae337a1170007421c647cb4220ab11c94a59af31adc8afaec284b8ff47d60654fec857f42a684400ef67bf549da24a2db38b5af02a11554')

def give_y_vals(counts):
    
    """
    Takes in the raw measurement outcomes from the circuit, and returns y values
    Args:
        counts: Dictionary
    Returns:
        new_counts: dictionary with strings reversed
        dict_y: dictionary with y values and counts
    """
    n = int(len(list(counts)[0]))
    
    n_list = [give_binary_string(z,n) for z in range(2**n)]
#    two_n_list = [give_binary_string(z,2*n), for z in range(2**(2*n))]
    new_counts ={}
    cropped_counts = {}
    for x in n_list:
        cropped_counts[x] = 0
    for i in counts:
        new_counts[i[::-1]] = counts[i]
        cropped_i = i[::-1][0:n]
        cropped_counts[cropped_i] += counts[i]
    
    dict_y = {}
    for key, value in cropped_counts.items():
        if value != 0:
            dict_y[key] = value
        
    return new_counts, dict_y

def get_all_backends(n_qubits = 1):
    """ Print a list of all backends with at least as many qubits as n_qubits
    Args:
        n_qubits: integer > 0
    Returns:
        backend_list: list of backends
        least_busy_backend: least busy backend
    """
    provider = IBMQ.get_provider(group='open')
    device_list = provider.backends(filters = lambda x: x.configuration().n_qubits >= n_qubits 
                                              and not x.configuration().simulator 
                                              and x.status().operational == True)
    least_busy_backend = least_busy(device_list)
    
    return device_list, least_busy_backend
def get_backends(device_preference = 0, n_qubits = 1):
    """
    Arg:
        n_qubits: int, number of qubits required for the circuit
        device_preference: If zero, return least busy backend.
                            Else, if string, return corresponding backend
        If value is 0, the lest busy backend is chosen for the quantum hardware
    Returns:
        simulator_backend: string denoting simulator backend
        device_backend: string denoting quantum hardware backend
    """
    provider = IBMQ.get_provider(group='open')
    simulator_backend = provider.get_backend('ibmq_qasm_simulator')
    
    if device_preference == 0:
        _, least_busy_backend = get_all_backends(n_qubits)
        device_backend = least_busy_backend
    else:
        device_backend = provider.get_backend(device_preference)
        
    return device_backend, simulator_backend

def give_binary_string(z,n):
	"""
	Function which takes in an integer input, and outputs the corresponding binary representation
	Args: 
		z: (Integer) z must lie in the set {0,1,...,2**n- 1}
		n: (Integer) Number of bits
	Returns: 
		s: String of length n containing the binary representation of integer z.
	"""
	if type(z) != int or type(n) != int:
		raise ValueError('Both arguments must be integers.')

	s = bin(z)
	m = n + 2 - len(s)
	if m < 0:
		raise ValueError('z is greater than the 2 raised to n. Need to input a larger value of n') 
	s = s.replace("0b", '0' * m)
	return s 

def generate_random_s(n, seed_val = 0):
	"""
	Generates a random bit string of length s
	Args:
		n: integer (length of bitstring)
	KwArgs:
		seed_val: integer. Seed. If = 0, s will be created randomly, else, the seed will be used
	Returns:
		s: string of length n, consisting of 0s and 1s only.
	"""

	if seed_val != 0:
		 random.seed(seed_val) 

	if type(n)!= int or n < 1 :
		return("The input must be a positive integer")
	# select random number 
	z = random.randint(0, 2**n - 1)
	# convert the number to binary representation
	s = give_binary_string(z,n)
	return s
		
		
def g_function(s,n, seed_val = 0):
	""" Given a string s, create a grover function f
	Args: 
		s: string of length n. Must contain only zeroes and ones.
	KwArgs:
		seed_val: integer. Seed. If = 0, f will be created randomly, else, the seed will be used
	Output:
		f: list of length 2^n. f[i] contains the value of f corresponding to i.
	"""
	
	if type(s) != str:  #If inputs are not strings, raise an error
		raise ValueError('s should be a string')
	if sum([1 for x in s if (x != '0' and x != '1')]) > 0 or sum([1 for x in s if (x != '0' and x != '1')]) > 0:  
		#check if s contains only zeros and ones
		raise ValueError('s should only contain 0s and 1s') 

	if seed_val != 0:
		 np.random.seed(seed_val) 
		
	a = 1
	n = len(s)
	N = 2**n
	func_inputs = range(N)
	comb = combinations(func_inputs, a)
	func_list = []

	for combination in comb:
		out_list = np.zeros(N)
		for i in combination:
			out_list[i] = 1
		func_list.append(out_list)
		
	f = random.choice(func_list)
		
	return f

def random_Uf(n, num_circs, backend):
	"""
	For a given value of n, create num_circs number of circuits with randomly created Ufs
	"""
	circ_list = []
	s_list = []
	f_list = []
	for i in range(num_circs):
		s = generate_random_s(n, seed_val = 0)
		f = g_function(s,n)
		s_list.append(s)
		f_list.append(f)
		g = grover()
		Uf = g.create_G(f,n)
		G = Operator(Uf)
		trans_circ = get_transpiled_circ(G,n,backend)
		print("depth: %d" % trans_circ.depth())
		circ_list.append(trans_circ)
	return s_list, f_list, circ_list



def load_account():
	IBMQ.load_account()

	provider = IBMQ.get_provider(group='open')
	provider.backends() 

def get_transpiled_circ(Uf, n, backend):
	g = grover()
	circ, theo = g.create_circuit(Uf,n,1)
	transpiled_circ = transpile(circ, backend = backend, optimization_level = 3)
	return transpiled_circ

def run_created_circuit(circ, backend, num_shots = 1024):
	"""
	Run the circuit on the backend
	Returns:
		job
		result
	"""
	job = execute(circ, backend = backend, shots = num_shots)
		
	return job, job.result() 


def retrieve_job_from_id(job_id, backend):
	retrieved_job = backend.retrieve_job(job_id)
	return retrieved_job
	
	
def get_meas_filter(num_qubits, backend, num_shots = 1024):
	meas_calibs, state_labels = complete_meas_cal(qr = num_qubits, 
											  circlabel='measureErrorMitigation')
	job_calib = qiskit.execute(meas_calibs, backend = backend, shots=1024, optimization_level = 0)
	job_monitor(job_calib)
	calib_job_id = job_calib.job_id()
	print(calib_job_id)

	cal_results = job_calib.result()  
	meas_fitter = CompleteMeasFitter(cal_results, state_labels)
	meas_filter = meas_fitter.filter
	return meas_filter, calib_job_id

compute_time = []
compile_time = []
uf_time = []	
	
def qc_vs_sim(n = 2, num_circs = 10, num_shots = 20, 
                                               device_preference = 'ibmq_london',
                                               save_fig_setting = 0):
    
    
   

	# load accounts and get backends
	load_account()
	device_backend, simulator_backend = get_backends(device_preference = 'ibmq_london', n_qubits = n)
	print("Chosen device backend is " + device_backend.name())

	n_qubits = n
	
	start = time.time()
	s_list, f_list, circ_list = random_Uf(n, num_circs, device_backend)
	end = time.time()
	run_time = int((end - start) * 1000)
	compile_time.append(run_time)

	# run on device hardware
	start = time.time()
	job_qc = execute(circ_list, backend = device_backend, shots = num_shots, optimization_level = 3)
	job_monitor(job_qc)
	qc_id = job_qc.job_id()
	print('Job id for device job' + qc_id )
	result_qc = job_qc.result()
	end = time.time()
	run_time = int((end - start) * 1000)
	compute_time.append(run_time)

	# run on IBM simulator
	job_sim = execute(circ_list, backend = simulator_backend, shots = num_shots, optimization_level = 3)
	job_monitor(job_sim)
	sim_id = job_sim.job_id()
	print('Job id for simulation job' + sim_id)
	result_sim = job_sim.result()

	# error mitigation
	meas_filter, calib_job_id = get_meas_filter(num_qubits = n, backend = device_backend, num_shots = 1024)
	print('Job id for mitigation circuits' + calib_job_id)
	mitigated_results = meas_filter.apply(result_qc)
	mitigated_counts = [mitigated_results.get_counts(i) for i in range(num_circs)]


	# print details to notepad file
	file2write=open("job_tracking.txt",'a')
	file2write.write("Current time is " + str(datetime.now()) + " \n")
	file2write.write("Multiple circuits run for n=%i, backend = %s, job ID = %s" 
					 %(n,device_backend, qc_id))
	file2write.write("\n Multiple circuits run for n=%i, backend = %s, job ID = %s" 
					 %(n,simulator_backend, sim_id))
	file2write.write("\n Error mitigation circuits run for n=%i, backend = %s, job ID = %s \n \n"
					 %(n,device_backend, calib_job_id))
	file2write.close()

	# Obtain y_lists in the form of dictionaries for the simulation job, the qc backend job, and mitigated counts
	raw_counts_qc = result_qc.get_counts()
	raw_counts_sim = result_sim.get_counts()
	raw_counts_mitigated = mitigated_counts
	list_y_dicts_sim = []
	list_y_dicts_qc = []
	list_y_dicts_mit = []

	for i in range(num_circs):
		_, cur_y_list_sim = give_y_vals(raw_counts_sim[i])
		_, cur_y_list_qc = give_y_vals(raw_counts_qc[i])
		_, cur_y_list_mit = give_y_vals(raw_counts_mitigated[i])
		list_y_dicts_sim.append(cur_y_list_sim)
		list_y_dicts_qc.append(cur_y_list_qc)
		list_y_dicts_mit.append(cur_y_list_mit)

	if save_fig_setting:
		# Plot and save all values
		legend = [device_backend.name(), simulator_backend.name(), 'After mitigation'] 
		figures_folder = 'Figures'
		if not os.path.isdir(figures_folder):
			os.mkdir(figures_folder)    
		
		
		for expt_num in range(num_circs):
			fig = plt.figure(figsize=(16,10))
			ax = fig.gca()
			plot_histogram([list_y_dicts_qc[expt_num],list_y_dicts_sim[expt_num],list_y_dicts_mit[expt_num]],
						   legend=legend, ax = ax)
		#    plt.tight_layout()
			plt.title('Comparison of outputs: n=%i, circuit_number = %i. Exact s = %s' %(n,expt_num,s_list[expt_num]),
					  fontsize=20)
			
			fig.savefig(figures_folder + '/n=%i_circ_num=%i_backend=%s.png' 
						%(n, expt_num, device_backend.name()), bbox_inches='tight')
			plt.close(fig)
		
	return list_y_dicts_sim, list_y_dicts_qc, list_y_dicts_mit, s_list, f_list, qc_id, sim_id, calib_job_id

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


	def all_f(self,n,a):
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
		comb = combinations(func_inputs, a)
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
		
	def reverse_bit_int(self,i):
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
			j = self.reverse_bit_int(i)
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


if __name__== '__main__':
	num_inputs = len(sys.argv)
	inputs = sys.argv
	correct = []
	n = 1
	num_circs = 10
	num_shots = 20
	device_preference = 'ibmq_burlington'
	save_fig_setting = 1
	for n in range(1,5):
		print(n)
		print("HERE")
		list_y_dicts_sim, list_y_dicts_qc, list_y_dicts_mit, s_list, f_list, qc_id, sim_id, calib_job_id = qc_vs_sim(n = n,																																				  num_circs = num_circs, num_shots = num_shots,																																			  device_preference = device_preference,																																			  save_fig_setting = save_fig_setting)
		test = s_list[0]
		print("test:", end ='')
		print(test)
		percent = list_y_dicts_qc[0][str(test)]/num_shots
		correct.append(percent)
	print(correct)	
	print(list_y_dicts_sim)
	print("##########################")
	print(list_y_dicts_qc)
	print("##########################")
	print(list_y_dicts_mit)
	("##########################")
	print(compile_time)
	print(compute_time)
	
	n = 3
	for i in range(8):
		start = time.time()
		list_y_dicts_sim, list_y_dicts_qc, list_y_dicts_mit, s_list, f_list, qc_id, sim_id, calib_job_id = qc_vs_sim(n = n,																																				  num_circs = num_circs, num_shots = num_shots,																																			  device_preference = device_preference,																																			  save_fig_setting = save_fig_setting)
		end = time.time()
		run_time = int((end - start) * 1000)
		uf_time.append(run_time)
		
	print(uf_time)
	
	# g = grover()
	
	# #test correctness
	# min_n = 1
	# max_n = 4
	# max_a = 1
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