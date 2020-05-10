"""
Created on Sat May  2 18:58:42 2020

@author: sathe
"""

from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import X,H,MEASURE
import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt




#%% 

def binary_add(s1,s2):
    """
    Binary addition (XOR) of two bit strings s1 and s2.
    Args: 
        Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
    Returns:
        s: String s given by s = s1 + s2 (of length n) containing only zeros and ones
    """
    x1 = np.array([int(i) for i in s1])
    x2 = np.array([int(i) for i in s2])
    x = (x1 + x2 ) % 2
    x = ''.join(str(e) for e in x.tolist())
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
    s = bin(z)
    m = n + 2 - len(s)
    s = s.replace("0b", '0' * m)
    return s 


def create_bv_function(a,b):
    """
    Create a function f:{0,1}^n --> {0,1} given by f(x) = a . x + b
    Args: 
        a: string of length n
        b: string of length n
        All arrays must contain integers 0 and 1
    Returns:
        f: Array of length 2^n. f[i] is the value of f corresponding to binary representation of i being the imput
        f = a . x + b
    """
    n = len(a)
    f = np.zeros(2**n, dtype = int)
    a = np.array([int(i) for i in a])
    b = np.array([int(i) for i in b])
    for z in range(2**n):   
        z_bin = give_binary_string(z, n)
        x = np.array([int(i) for i in z_bin])  # Binary representation of index_num, in array form
        f[z] = np.sum((x * a + b)) % 2
    return  f

#a = '101'
#b = '001'
#print(create_bv_function(a,b))

def create_Uf(f):
    """
    Given a function f:{0,1}^n ---> {0,1}, creates and returns the corresponding oracle (unitary matrix) Uf
    Args:
        f: ndarray of length n, consisting of integers 0 and 1
    Returns:
        Uf: ndarray of size [2**(n+1), 2**(n+1)], representing a unitary matrix.
    """
    two_raized_n = np.size(f)
    n = int(np.log2(two_raized_n))
    N = 2 ** (n+1)
    Uf = np.zeros([N,N], dtype = complex )
    
    for z in range(2**n):
        f_z = f[z]
        z_bin = give_binary_string(z,n)
        
        for j in ['0','1']:
            inp = z_bin + j
            out = z_bin + binary_add(str(f_z),j)
            Uf[int(out,2),int(inp,2)] = 1
    return Uf


def get_random_f(n):
    """
    Create a random function f:{0,1}^n ---> {0,1} which can be expressed as f(x) = a . x + b
    Args:
        n: Integer. Denotes the number of bits.
    Returns:
        ndarray of length n, containing 0s and 1s.
    """
    ## Obtain a and b randomly
    a = give_binary_string(rd.randint(0, 2**n - 1), n)
    b = str(rd.randint(0,1))
    return create_bv_function(a,b)
    
    

def BV(Uf_quil_def, Uf_gate, n, time_out_val = 100):
    """
    Bernstrin-Vazirani algorithm: Determines the value of a, for a function f = a . x + b
    Args:
        Uf_gate: gate object which acts on n+1 qubits
        Uf_quil_def: DefGate object corresponding to oracle Uf
        n: Integer, the length on input bit strings which f takes.
    
    Kwargs:
        time_out_val: Integer. Timeout in seconds.
        
    Returns: 
        String: Measured state after executing the BV circuit. This corresponds to the value of a predicted by the BV circuit.
    """
    ## Define the circuit
    p = Program()
    ro = p.declare('ro', 'BIT', n)
    p += X(n)
    for gate_ind in range(n+1):
        p += H(gate_ind)
    p += Uf_quil_def
    p += Uf_gate(*tuple(range(n+1)))
    for gate_ind in range(n):
        p += H(gate_ind)
    for gate_ind in range(n):
        p += MEASURE(gate_ind, ro[gate_ind])

    ## compile and run
    qc = get_qc(str(n+1)+'q-qvm') 
    qc.compiler.client.timeout = time_out_val
    executable = qc.compile(p)
    result = np.reshape(qc.run(executable),n)
    print("The measured state, and hence a = "+str(result)[1:-1])
    return "".join(str(s) for s in result)
    

def verify_BV_output(BV_output, f):
    """
    Verifies if the output of BV algorithm is correct.
    Args:
        f: ndarray of length N = 2**n, contining the values of function.
        BV_output: string of length n, containing the value of a predicted by the BV algorithm.                    
    Returns:
        is_correct: bool (TRUE if the DJ output is correct, and FALSE otherwise)
    """
    b = str(f[0])
    a = [int(i) for i in BV_output]
    f_predicted = create_bv_function(a,b)
    
    delta_f = f_predicted - f 
    is_correct = np.count_nonzero(delta_f) == 0 
    return is_correct
    
    
#get_random_f(2)
#a = '101'
#b = '001'
#create_Uf(create_bv_function(a,b))
#%% Trial testing 
    
n = 3
f = get_random_f(n)
#f = create_bv_function('101','1')
Uf_matrix = create_Uf(f)
print("Done creating Uf matrix")
Uf_quil_def = DefGate("Uf", Uf_matrix)
Uf_gate = Uf_quil_def.get_constructor()    
BV_output = BV(Uf_quil_def, Uf_gate, n)
verify_BV_output(BV_output, f)

#%% 
### Testing
time_out_val = 10000
n_min = 1
n_max = 6
n_list = list(range(n_min,n_max+1))
num_times = 5
time_reqd_arr = np.zeros([(n_max-n_min)+1, num_times])

for ind, n in enumerate(n_list):
    for iter_ind in range(num_times):
        f = get_random_f(n)
        Uf = create_Uf(f)
        Uf_quil_def = DefGate("Uf", Uf)
        Uf_gate = Uf_quil_def.get_constructor()
        start = time.time()
        BV_output = BV(Uf_quil_def, Uf_gate, n, time_out_val)
        end = time.time()
        time_reqd_arr[ind, iter_ind] = end-start
        if not(verify_BV_output(BV_output, f)):
            print("BV algorithm failed to obtain the right output for n=%i"%n)
        print("done for n = %i and iteration = %i... took %i seconds"%(n,iter_ind,(end-start)))
            

#%% Save the data

np.savez('BV_benchmarking.npz', n_list = n_list,time_reqd_arr = time_reqd_arr)
       
#%% Load data

data = np.load('BV_benchmarking.npz')
n_list = data['n_list']
time_reqd_arr = data['time_reqd_arr']

#%%
avg_time = np.sum(time_reqd_arr,1)/np.shape(time_reqd_arr)[1]
#%% Plot and save 

plt.rcParams["font.family"] = "serif"
fig = plt.figure(figsize=(16,10))

z = np.polyfit(n_list, avg_time, 10)
p = np.poly1d(z)

plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
plt.title('Execution time scaling for Bernstein-Vazirani algorithm', fontsize=25)
plt.xlabel('n (bit string length)',fontsize=20)
plt.ylabel('Average time of execution (s)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.grid()


fig.savefig('Figures/BV.png', bbox_inches='tight')


#%% Dependance of time of execution on Uf

num_times = 100
time_out_val = 10000
time_array = np.zeros(num_times)
n = 3
for iter_ind in range(num_times):
    f = get_random_f(n)
    Uf = create_Uf(f)
    Uf_quil_def = DefGate("Uf", Uf)
    Uf_gate = Uf_quil_def.get_constructor()
    start = time.time()
    BV_output = BV(Uf_quil_def, Uf_gate, n, time_out_val)
    end = time.time()
    time_array[iter_ind] = end - start
    if not(verify_BV_output(BV_output, f)):
        print("DJ_algorithm failed to obtain the right output for n=%i"%n)
    print("done for n = %i and iteration = %i... took %i seconds"%(n,iter_ind,(end-start)))
    
#%% Save the data

np.savez('BV_Uf_dependence.npz', num_times = num_times,time_array = time_array, n = n, )

#%% Load data

data = np.load('BV_Uf_dependence.npz')
num_times = data['num_times']
time_array = data['time_array']
n = data['n']

#%% Plot histogram of time taken for a particular value of n

plt.rcParams["font.family"] = "serif"
fig = plt.figure(figsize=(16,10))


plt.hist(time_array)
plt.title('Dependence of execution time on $U_f$ (Bernstein-Vazirani algorithm)', fontsize=25)
plt.xlabel('Execution time (s)',fontsize=20)
plt.ylabel('Frequency of occurence',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


fig.savefig('Figures/BV_hist.png', bbox_inches='tight')

