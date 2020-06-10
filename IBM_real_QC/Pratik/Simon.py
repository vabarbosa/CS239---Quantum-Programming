





from qiskit import IBMQ
#IBMQ.save_account('d5797c2dfdae5724272e4a64865c101a5bbe11051a97f5b31d0ed8aefdf1ff7c823aab336c87065b287069c7f571f2721d5554146743b11bb19d00cd1efd7e6c')
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.compiler import transpile, assemble
import qiskit
from qiskit.visualization import plot_histogram
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)
#qiskit.__version__
#qiskit.__qiskit_version__



#provider = IBMQ.get_provider(group='open')
#provider.backends()
#
## Choose a backend
#backend = provider.get_backend('ibmq_essex')
#backend.configuration().basis_gates


# runing a job on a backednd
#job_exp = execute(qc, backend=backend)
#job_monitor(job_exp)
#result_exp = job_exp.result()
#counts_exp = result_exp.get_counts(qc)
#plot_histogram([counts_exp,counts], legend=['Device', 'Simulator'])

# retrieving a previously run job

#job_id = job_exp.job_id()
#print('JOB ID: {}'.format(job_id))
#retrieved_job = backend.retrieve_job(job_id)
#retrieved_job.result().get_counts(qc)

#%% Load required modules
import numpy as np
import os
from qiskit import(QuantumCircuit, execute, Aer)
import random as rd
import time
import matplotlib.pyplot as plt



#%% Define functions


def load_account():
    IBMQ.load_account()
    
    # Get a list of all backends
    provider = IBMQ.get_provider(group='open')
    provider.backends()  # print a list of all available backends
    

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

def binary_add(s1,s2):
        """
        Binary addition (XOR) of two bit strings s1 and s2.
        Args: 
            Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
        Returns:
            s: String s given by s = s1 + s2 (of length n) containing only zeros and ones
        """
        if type(s1) != str or type(s2) != str:  #If inputs are not strings, raise an error
            raise ValueError('The inputs are not strings')
        if sum([1 for x in s1 if (x != '0' and x != '1')]) > 0 or sum([1 for x in s2 if (x != '0' and x != '1')]) > 0:
            raise ValueError('Input strings contain characters other than 0s and 1s') 
        if len(s1) != len(s2):
            raise ValueError('Input strings are not of the same length') 
            
        x1 = np.array([int(i) for i in s1])
        x2 = np.array([int(i) for i in s2])
        x = (x1 + x2 ) % 2
        x = ''.join(str(e) for e in x.tolist())
        return x

def dot_product(s1,s2):
        """
        Dot product of strings s1 and s2.
        Args: 
            Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
        Returns:
            prod: Integer (0 or 1)
        """
        if type(s1) != str or type(s2) != str:  #If inputs are not strings, raise an error
            raise ValueError('The inputs are not strings')
        if sum([1 for x in s1 if (x != '0' and x != '1')]) > 0 or sum([1 for x in s2 if (x != '0' and x != '1')]) > 0:
            raise ValueError('Input strings contain characters other than 0s and 1s') 
        if len(s1) != len(s2):
            raise ValueError('Input strings are not of the same length') 
            
        x1 = np.array([int(i) for i in s1])
        x2 = np.array([int(i) for i in s2])
        x = np.sum(x1 * x2) %2
        return x


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
             rd.seed(seed_val) 
        
        if type(n)!= int or n < 1 :
            return("The input must be a positive integer")
        # select random number 
        z = rd.randint(0, 2**n - 1)
        # convert the number to binary representation
        s = give_binary_string(z,n)
        return s

def s_function(s, seed_val = 0):
        """ Given a string s, create a Simon function f
        Args: 
            s: string of length n. Must contain only zeroes and ones.
        KwArgs:
            seed_val: integer. Seed. If = 0, f will be created randomly, else, the seed will be used
        Output:
            f: list of length 2^n. f[i] contains the value of f corresponding to i. Each element of the list is a string
            f is such that, f(x) = f(y) iff x + y \in {0,s}
        """
        
        if type(s) != str:  #If inputs are not strings, raise an error
            raise ValueError('s should be a string')
        if sum([1 for x in s if (x != '0' and x != '1')]) > 0 or sum([1 for x in s if (x != '0' and x != '1')]) > 0:  
            #check if s contains only zeros and ones
            raise ValueError('s should only contain 0s and 1s') 
        
        if seed_val != 0:
             np.random.seed(seed_val) 
             
        n = len(s)
        N = 2**n
        f_temp = [-1 for i in range(N)]
        
        perm_random = np.random.permutation(N)
        if s == '0'*n:
            for i in range(N):
                f_temp = np.copy(perm_random)
        else:
            chosen_perm = perm_random[0:int(N/2)]
            ind_1 = 0
            for i in range(N):
                if f_temp[i] == -1:
                    f_temp[i] = chosen_perm[ind_1]
                    f_temp[int(binary_add(give_binary_string(i,n) , s), 2)] = chosen_perm[ind_1]
                    ind_1 += 1
        f = [give_binary_string(z.item(),n) for z in f_temp]
        return f

def create_Uf(f):
        """
        Given a function f:{0,1}^n ---> {0,1}^n, creates and returns the corresponding oracle (unitary matrix) Uf
        Args:
            f: list of length n. f[i] is a string of length n.
        Returns:
            Uf: ndarray of size [2**(2n), 2**(2n)], representing a unitary matrix.
        """
        if type(f) != list:
            raise ValueError('Input the function in the form of a list')
        if any((x != '0' and x != '1') for x in "".join([y for y in f])):
            raise ValueError('The input function should only contain zeros and ones.')
        len_fs = [len(x) for x in f]
        if np.sum(np.diff(len_fs)) != 0:
            raise ValueError('Each element of f must be a string of the same length as the others.')
        n = len_fs[0]
        N = 2 ** (2*n)
        Uf = np.zeros([N,N], dtype = complex )
        
        for z in range(2**n):
            f_z = f[z]
            z_bin = give_binary_string(z,n)
            
            all_binary_strings = [give_binary_string(c,n) for c in range(2**n)]
            for j in all_binary_strings:
                inp = z_bin + j
                out = z_bin + binary_add(f_z,j)
                Uf[int(out[::-1],2),int(inp[::-1],2)] = 1
        return Uf

def get_Simon_circuit(Uf):
        """
        Simon's algorithm: Determines the value of s, for a function f = a . x + b
        Args:
            Uf: numpy array. Must be of shape [N,N], with N being an even power of 2.
        Returns: 
            circ: Qiskit QuantumCircuit object (not transpiled).
        """
        if not isinstance(Uf, np.ndarray):
            raise ValueError("Uf must be a numpy array.")
        if Uf.ndim != 2:
            raise ValueError("Uf should be a two dimensional array.")
        if Uf.shape[0] != Uf.shape[1]:
            raise ValueError("Uf must be a square array.")
        if np.log2(Uf.shape[0])%2 != 0:
            raise ValueError("number of rows and columns of Uf must be an even power of 2")
#        if type(backend) != qiskit.providers.ibmq.ibmqbackend.IBMQBackend:
#            raise ValueError("The backend parameter is not a backend object.")
        n = int(np.log2(Uf.shape[0]) / 2)
        ## Define the circuit
#        circ = QuantumCircuit(2*n, n)   
        circ = QuantumCircuit(2*n, 2*n) 
        
        for i in range(n):
            circ.h(i)
        
        circ.iso(Uf, list(range(2*n)), [])
            
        for i in range(n):
            circ.h(i)
            
#        circ.measure(list(range(0,n))[::-1], list(range(n)))            
        circ.measure(list(range(0,2*n)), list(range(2*n)))  # have not reversed the qubits! Dont forget to reverse later
#        transpiled_circ = transpile(circ, backend = backend, optimization_level = 3) # = Aer.get_backend('qasm_simulator')
        
        
        return circ
def get_transpiled_circ(circ, backend):
    transpiled_circ = transpile(circ, backend = backend, optimization_level = 3)
    return transpiled_circ
    # = Aer.get_backend('qasm_simulator')
    
         
def run_created_circuit(circ, backend, num_shots = 1024):
    """
    Run the circuit on the backend
    Returns:
        job
        result
    """
    job = execute(circ, backend = backend, shots = num_shots)
        
#    qobj = assemble(circ, backend = backend, shots = num_shots)
#    job = backend.run(qobj)
    return job, job.result()

#        result = job.result()
#        counts = result.get_counts(circ)    
    

def retrieve_job_from_id(job_id, backend):
    retrieved_job = backend.retrieve_job(job_id)
    return retrieved_job

def give_y_vals(counts):
    
    """
    Takes in the raw measurement outcomes from the circuit, and returns y values
    Args:
        counts: Dictionary
    Returns:
        new_counts: dictionary with strings reversed
        cropped_counts: dictionary with y values and counts
    """
    n = int(len(list(counts)[0])/2)
    
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
    
        
    return new_counts, cropped_counts

def random_Uf_multiple_circuits(n, num_circs):
    """
    For a given value of n, create num_circs number of circuits with randomly created Ufs
    """
    circ_list = []
    s_list = []
    f_list = []
    for i in range(num_circs):
        s = generate_random_s(n, seed_val = 0)
        f = s_function(s, seed_val = 0)
        s_list.append(s)
        f_list.append(f)
        Uf = create_Uf(f)
        circ_list.append(get_Simon_circuit(Uf))
    return s_list, f_list, circ_list


def get_meas_filter(num_qubits, backend, num_shots = 1024):
    meas_calibs, state_labels = complete_meas_cal(qr = num_qubits, 
                                              circlabel='measureErrorMitigation')
    job_calib = qiskit.execute(meas_calibs, backend = device_backend, shots=1024, optimization_level = 0)
    job_monitor(job_calib)
    calib_job_id = job_calib.job_id()
    print(calib_job_id)
    
    cal_results = job_calib.result()  
    meas_fitter = CompleteMeasFitter(cal_results, state_labels)
#    meas_fitter.plot_calibration()
    #print(meas_fitter.cal_matrix)
    meas_filter = meas_fitter.filter
    # Results with mitigation
    mitigated_results = meas_filter.apply(result_qc)
    mitigated_counts = mitigated_results.get_counts(0)

    return meas_filter, calib_job_id

def s_solution_unmodified(list_y):
        """ The classical post-processing
        Args:
            list_y: List of strings of length n
        Returns:
            list_s: List of solutions to y.s=0
        """
        
        n = len(list_y[0])
        list_s = []
        
        for z in range(2**n):
            potential_s = give_binary_string(z,n)
            if np.sum([dot_product(potential_s, y) for y in list_y]) == 0:
                list_s.append(potential_s)
                
        return list_s

#def qc_vs_sim_comparison_execute(n = 2, num_circs = 10, num_shots = 20, device_preference = 'ibmq_burlington'):
#    """
#    Args:
#        n: Int- value of n for f
#        num_circs: Int - Number of random Ufs to be created
#        num_shots: Int - Number of runs per Uf
#        device_preference: If integer 0, obtain the least busy backend
#                            If string, obtain corresponding backend
#    """
n = 1
num_circs = 10

n_qubits = 2 * n
s_list, f_list, circ_list = random_Uf_multiple_circuits(n, num_circs)

# load accounts and get backends
load_account()
device_backend, simulator_backend = get_backends(device_preference = 'ibmq_burlington', n_qubits = n_qubits)
print("Chosen device backedn is " + device_backend.name())

# run on device hardware
job_qc = execute(circ_list, backend = device_backend, shots = num_shots, optimization_level = 3)
job_monitor(job_qc)
print('Job id for device job' + job_qc.job_id())
result_qc = job_qc.result()

# run on IBM simulator
job_sim = execute(circ_list, backend = simulator_backend, shots = num_shots, optimization_level = 3)
job_monitor(job_sim)
print('Job id for simulation job' + job_sim.job_id())
result_sim = job_sim.result()

# error mitigation
meas_filter, calib_job_id = get_meas_filter(num_qubits = n_qubits, backend = device_backend, num_shots = 1024)
print('Job id for mitigation circuits' + calib_job_id)
mitigated_results = meas_filter.apply(result_qc)
mitigated_counts = [mitigated_results.get_counts(i) for i in range(num_circs)]


# print details to notepad file
file2write=open("job_tracking.txt",'a')
file2write.write("Multiple circuits run for n=%i, backend = %s, job ID = %s" 
                 %(n,device_backend, job_qc.job_id()))
file2write.write("\n Multiple circuits run for n=%i, backend = %s, job ID = %s" 
                 %(n,simulator_backend, job_sim.job_id()))
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
    
# Raw processing assuming no error

list_of_s_lists_qc = [s_solution_unmodified(list(i)) for i in list_y_lists_qc]
list_of_s_lists_sim = [s_solution_unmodified(list(i)) for i in list_y_lists_sim]
list_of_s_lists_mit = [s_solution_unmodified(list(i)) for i in list_y_lists_mit]
    

#%% Testing

n = 2
num_circs = 10














    




job_qc, result_qc = run_created_circuit(circ, backend = device_backend, num_shots = 20)
print(job_qc())



#%%
s = generate_random_s(n, seed_val = 4)
#print(s)
f = s_function(s, seed_val = 5)
#print(f)
Uf = create_Uf(f)
circ = get_Simon_circuit(Uf)
circ_qc = get_transpiled_circ(circ, backend = device_backend)
circ_sim = get_transpiled_circ(circ, backend = simulator_backend)
print('Untranspiled circuit depth = %i, transpiled circuit depth qc = %i, transpiled circuit depth sim =%s'
      %(circ.depth(), circ_qc.depth(), circ_sim.depth()))

job_qc, result_qc = run_created_circuit(circ, backend = device_backend, num_shots = 20)
print(job_qc())
job_monitor(job_qc)
job_sim, result_sim = run_created_circuit(circ, backend = simulator_backend, num_shots = 20)
print(job_sim.job_id())
job_monitor(job_sim)

# Save job id
file2write=open("job_tracking.txt",'a')
file2write.write("n=%i, backend = %s, job ID = %s" %(n,device_backend, job_qc.job_id()))
file2write.write("\n n=%i, backend = %s, job ID = %s \n \n" %(n,simulator_backend, job_sim.job_id()))
file2write.close()

#plot_histograms for comparison
legend = ['On '+ device_backend.name(), 'On '+simulator_backend.name()]
plot_histogram([result_qc.get_counts(), result_sim.get_counts()], legend=legend)


# Calibration circuits
# Generate the calibration circuits
meas_calibs, state_labels = complete_meas_cal(qr = circ.qregs[0], 
                                              circlabel='measureErrorMitigation')
meas_calibs[1].draw()
job_calib = qiskit.execute(meas_calibs, backend = device_backend, shots=1024, optimization_level = 0)
print(job_calib.job_id())
job_monitor(job_calib)

file2write=open("job_tracking.txt",'a')
file2write.write("Calibration: n=%i, backend = %s, job ID = %s \n \n" %(n,device_backend, job_calib.job_id()))
file2write.close()

cal_results = job_calib.result()
#plot_histogram(cal_results.get_counts(meas_calibs[1]))

meas_fitter = CompleteMeasFitter(cal_results, state_labels)
meas_fitter.plot_calibration()
#print(meas_fitter.cal_matrix)
meas_filter = meas_fitter.filter
# Results with mitigation
mitigated_results = meas_filter.apply(result_qc)
mitigated_counts = mitigated_results.get_counts(0)

legend = ['On '+ device_backend.name(), 'On '+simulator_backend.name(), 'Mitigated counts']
plot_histogram([result_qc.get_counts(), result_sim.get_counts(), mitigated_counts], legend=legend)

new_counts_sim, y_dict_sim = give_y_vals(result_sim.get_counts())
new_counts_qc, y_dict_qc = give_y_vals(result_qc.get_counts())
new_counts_mit, y_dict_mit = give_y_vals(mitigated_counts)
plot_histogram([y_dict_qc, y_dict_sim, y_dict_mit], legend=legend)


#%% 
job_qc = retrieve_job_from_id('5edfdf84748037001236db67', backend = device_backend)
job_sim = retrieve_job_from_id('5edfe02d65afea001180b719', backend = simulator_backend)
result_sim = job_sim.result()
result_qc = job_qc.result()




#%% Modify this in order for it to work on IBM quantum computers instead of simulators
    
    
        
    def simon_solution(self,f, num_shots = 20):
        """ 
        For a higher level implementation of Simon's algorithm. Takes an input f, and number of times the circuit is to be run.
        Returns the result of the search.
        Args:
            f: list of length n. f[i] is a string of length n.
            num_shots: Integer>0. Higher the value, larger the probability of success
        Returns:
            list_s: List of strings, with each string being of length n. These will contain the solution of the search problem, as 
                    determined by the Grover's algorithm
        """
        Uf = self.create_Uf(f)
        circ = self.get_Simon_circuit(Uf)
        counts = self.run_created_circuit(circ, num_shots = num_shots)
        list_y = [y for y in counts]
        list_s = self.s_solution(list_y)
        return list_s

    def verify_Simon_output(self,f, list_s):
        """Verifies if the output from the Simon's algorithm is correct
        Args:
            f: list of length n. f[i] is a string of length n.
            list_s: List of strings, with each string being of length n. These will contain the solution of the search problem, as 
                    determined by the Grover's algorithm. The first element of list_s will always contain '000..0'.
        Returns:
            is_correct: Bool. True if output is correct, else false.
        """
        ## Type checking
        if type(f) != list:
            raise ValueError('Input the function in the form of a list')
        if any((x != '0' and x != '1') for x in "".join([y for y in f])):
            raise ValueError('The input function should only contain zeros and ones.')
        len_fs = [len(x) for x in f]
        if np.sum(np.diff(len_fs)) != 0:
            raise ValueError('Each element of f must be a string of the same length as the others.')
            
        if type(list_s) != list:
            raise ValueError('list_s must be a list')
        
        ## return true only if list_s contains exactly two strings, and the non-zero string satisfies the condition f[0] = f[0+s]
        
        if len(list_s) == 2 and f[0] == f[0 + int(list_s[1],2)]:
            return True
        elif len(list_s) == 1:
            return True
        else:
            return False
        
    def scaling_analysis( self, n_min = 1, n_max = 4, time_out_val = 10000, num_times = 20, save_data = True, num_shots = 20): 
        """ Obtains the time scaling as n is increased
        Kwargs:
            n_min = (integer) lowest value of n to be considered
            n_max = (integer) highest value of n to be considered
            time_out_val = (integer) timeout value for running the QISKIT circuit
            num_times: (integer) number of times Simon's algorithm is to be executed for each n.
            save_data: (bool) If True, save the data in a .npz file
        Returns:
            n_list: (list) contains all integers from n_min to n_max
            time_reqd_arr: [ndarray) [i,j] entry contains the amount of time it took for ith value of n, and jth iteration
            correctness_arr: (ndarray) True or False entries depending on whether the algorithm succeeded or failed.
            avg_time: (ndarray) avg_time[i] contains the average amount of time taken for n_list[i]
        """
            
        n_list = list(range(n_min,n_max+1))
        time_reqd_arr = np.zeros([(n_max-n_min)+1, num_times])
        correctness_arr = np.ones([(n_max-n_min)+1, num_times], dtype = int)
        ci_list = ["failed", "succeeded"]
        for ind, n in enumerate(n_list):
            for iter_ind in range(num_times):
                s = self.generate_random_s(n)
                f = self.s_function(s)
                
                start = time.time()
                list_s = self.simon_solution(f, num_shots = num_shots)
                end = time.time()
                time_reqd_arr[ind, iter_ind] = end-start
                correctness_arr[ind, iter_ind] *= self.verify_Simon_output(f, list_s)
                print("n = %i and iteration = %i:It took %i seconds. Algorithm %s. Correct s = %s, obtained values are"
                      %(n,iter_ind,(end-start), ci_list[correctness_arr[ind, iter_ind]], s),list_s)
        avg_time = np.sum(time_reqd_arr,1)/np.shape(time_reqd_arr)[1]
        if save_data:
            np.savez('Simon_scaling.npz', n_list = n_list, num_times = num_times, time_reqd_arr = time_reqd_arr,
                     correctness_arr = correctness_arr, avg_time = avg_time)
        return n_list, time_reqd_arr, correctness_arr, avg_time
    
    def load_scaling_analysis(self): 
        """
        Load and return the the data for time scaling vs n.
        """
        data = np.load('Simon_scaling.npz')
        n_list = data['n_list']
        num_times = data['num_times']
        correctness_arr = data['correctness_arr']
        time_reqd_arr = data['time_reqd_arr']
        avg_time = data['avg_time']
        
        return n_list, num_times, correctness_arr, time_reqd_arr, avg_time
    
    def plot_and_save_scaling(self, n_list, avg_time, save_data = False):
        """
        Plot average run time vs n
        Args:
            n_list: List of n values
            avg_time: list of average time of execution corresponding to n's from n_list
            save_data: save the figure if this is set to True.
        """
        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))
        
        z = np.polyfit(n_list, avg_time, 10)
        p = np.poly1d(z)
        
        plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
        plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
        plt.title('Execution time scaling for Simon\'s algorithm', fontsize=25)
        plt.xlabel('n (bit string length)',fontsize=20)
        plt.ylabel('Average time of execution (s)',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if save_data:
            fig.savefig('Figures/Simon_scaling.png', bbox_inches='tight')
        
    def check_correctness(self, n= 2, num_times = 100, num_shots = 20):
        """
        Run Simon's algorithm for a given n, num_times number of times. For each run, f is chosen randomly.
        KwArgs:
            n: Integer
            num_times: Integer
            num_shots: integer
        Returns: 
            correctness_arr: Array of bools
        """
        correctness_arr = np.ones(num_times, dtype = int)
        ci_list = ["failed", "succeeded"]
        for iter_ind in range(num_times):
            s = self.generate_random_s(n)
            f = self.s_function(s)
            list_s = self.simon_solution(f, num_shots = num_shots)
            correctness_arr[iter_ind] *= self.verify_Simon_output(f, list_s)
            print("Iteration = %i: Algorithm %s. Correct s = %s, obtained values are"
                      %(iter_ind, ci_list[correctness_arr[iter_ind]], s),list_s)
            
        return correctness_arr
    
    
    def Uf_dependence_analysis(self, n = 3, num_times= 100, time_out_val = 10000, save_data = True, num_shots = 20):
        """
        Runs Simon's algorithms for num_times number of times for randomly chosen f's correspoding to given value of n
        kwargs:
            n: Integer
            num_times: Integer (number of random f's to be tested)
            time_out_val: integer
            save_data: bool
            num_shots: integer
        returns:
            correctness_arr: array of bools
            time_required_arr: array of floats
        """
      
        correctness_arr = np.ones(num_times, dtype = int)
        time_reqd_arr = np.zeros(num_times)
        ci_list = ["failed", "succeeded"]
        for iter_ind in range(num_times):
            s = self.generate_random_s(n)
            f = self.s_function(s)
            
            start = time.time()
            list_s = self.simon_solution(f, num_shots = num_shots)
            end = time.time()
            time_reqd_arr[iter_ind] = end-start
            correctness_arr[iter_ind] *= self.verify_Simon_output(f, list_s)
            print("Iteration %i:It took %i seconds. Algorithm %s. Correct s = %s, obtained values are"
                  %(iter_ind, (end-start), ci_list[correctness_arr[iter_ind]], s),list_s)
        
        if save_data:
            np.savez('Simon_Uf_dependence.npz', num_times = num_times, n = n,
                     correctness_arr = correctness_arr, time_reqd_arr = time_reqd_arr)

            
        return correctness_arr, time_reqd_arr
    
    def load_Uf_analysis(self): 
        """
        Load data from file
        """
        data = np.load('Simon_Uf_dependence.npz')
        num_times = data['num_times']
        correctness_arr = data['correctness_arr']
        time_reqd_arr = data['time_reqd_arr']        
        n = data['n']
        
        return num_times, correctness_arr, n, time_reqd_arr
    
    def plot_and_save_UF_analysis(self, time_reqd_arr, save_data = False):
        """
        Plot histogram of run time, for randomly chosen fs
        Args:
            time_reqd_arr: array of floats
            save_data: save the figure if this is set to True.
        """
        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))
        
        
        plt.hist(time_reqd_arr)
        plt.title('Dependence of execution time on $U_f$ (Simon\'s algorithm)', fontsize=25)
        plt.xlabel('Execution time (s)',fontsize=20)
        plt.ylabel('Frequency of occurence',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        if save_data:
            fig.savefig('Figures/Simon_hist.png', bbox_inches='tight')

        

#%% Old stuff











#%% 



#%% Suggested template for a lower level implementation:
p = Program()         
#### define f here
#### Sample f:
n = 3
s = p.generate_random_s(n)
f = p.s_function(s)
######
Uf = p.create_Uf(f)
circ = p.get_Simon_circuit(Uf)
counts = p.run_created_circuit(circ, num_shots = 40)
list_y = [y for y in counts]
list_s = p.s_solution(list_y)
print("Correct s = %s, obtained values are"
      %s,list_s)

#%% Suggested template for a higher level implementation:
p = Program()         
#### define f here
list_s = p.simon_solution(f, num_shots = 20)
print("obtained values of s are",list_s)

#%% Suggested template for testing and benchmarking:
p = Program()
n_list, time_reqd_arr, correctness_arr, avg_time = p.scaling_analysis(n_min = 1, n_max = 4, time_out_val = 10000, num_times = 20, save_data = True)
n_list, num_times, correctness_arr, time_reqd_arr, avg_time = p.load_scaling_analysis()
p.plot_and_save_scaling(n_list, avg_time, save_data = False)
p.check_correctness()
correctness_arr, time_reqd_arr = p.Uf_dependence_analysis(save_data = False)
num_times, correctness_arr, n, time_reqd_arr = p.load_Uf_analysis() 
p.plot_and_save_UF_analysis(time_reqd_arr, save_data = False)
    