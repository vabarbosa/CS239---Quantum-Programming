# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:50:05 2020

@author: sathe
"""

from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import X,H,MEASURE
from itertools import combinations 
import numpy as np
import random as rd
import time
from math import factorial as F
import matplotlib.pyplot as plt


#%% Define the function needed
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


def create_all_functions(n):
    """
    Creates all possible constant and balanced functions for n bits
    Args:
        n: Integer
    Returns:
        func_list_balanced: ndarray with dimensions [M, n], where M = (N) C (N/2) is the number of possible balanced functions. N = 2**n = dim(Hilbert space).
                            func_list_balanced[i,j] = 0 or 1, containing the value of f_i(j), where f_i = i^{th} function. 
        func_list_constant: ndarray with dimensions [2, n]. The two rows contain the two possible constant functions.
    """
    N = 2**n
    list_of_numbers = list(range(N))
    comb = combinations(list_of_numbers, int(N/2))
    func_list_balanced = []
    for i in list(comb):
        out_list = np.zeros(N, dtype = int)
        for j in list(i):
            out_list[j] = 1
        func_list_balanced.append(out_list)
    
    func_list_constant = [[0 for i in range(N)], [1 for i in range(N)]]
    
    return func_list_balanced, func_list_constant
        

#n = 4
#func_list_balanced, func_list_constant = create_all_functions(n)

def get_random_f(n):
    """
    Create a random function f:{0,1}^n ---> {0,1} which is either balanced or constant
    Args:
        n: Integer. Denotes the number of bits.
    Returns:
        ndarray of length n, containing 0s and 1s.
    """
    ## First, decide whether to output a constant function or a balanced function
    ## number of balanced functions = 2**n C 2**(n-1)
    ## number of constant functions = 2
    def nCr(n,r):
        """
        Calculate n-choose-r
        """
        return F(n) / F(r) / F(n-r)

    num_balanced = nCr(2**n, 2**(n-1))
    random_index = rd.randint(0, num_balanced+1)
    if random_index < num_balanced:
        #return a balanced function
         perm = np.random.permutation(2**n)
         perm = perm[0:2**(n-1)]
         f = np.zeros(2**n, dtype = int)
         for i in perm:
             f[i] = 1
         return f
    elif random_index == num_balanced:
        return np.zeros(2**n, dtype = int)
    else:
        return np.ones(2**n, dtype = int)


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

def verify_dj_output(DJ_output, f):
    """
    Verifies if the output of DJ algorithm is correct.
    Args:
        f: ndarray of length N, contining the values of function.
        DJ_output: Integer.
                    1 if function is constant, and 0 if the function is balanced
                    
    Returns:
        is_correct: bool (TRUE if the DJ output is correct, and FALSE otherwise)
    """
    N = len(f)
    is_constant = (np.sum(f) == N or np.sum(f) == 0)
    
    if is_constant == DJ_output:
        return True
    else:
        return False

#Uf = create_Uf(get_random_f(3))
#print(Uf)


def DJ(Uf_quil_def, Uf_gate, n, time_out_val = 100):
    """
    Deutsch-Jozsa algorithm: Determines if the given function is constant or balanced
    Args:
        Uf_gate: gate object which acts on n+1 qubits
        Uf_quil_def: DefGate object corresponding to oracle Uf
        n: Integer, the length on input bit strings which f takes.
    
    Kwargs:
        time_out_val: Integer. Timeout in seconds.
        
    Returns: 
        Integer: 0 if function is balanced, and 1 if function is constant
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
    
    #print("Measured state is "+str(result)[1:-1])
    if np.sum(result) != 0:
        print("Function is balanced")
        return 0
    else:
        print("function is constant")
        return 1
    
    

#%% Trial testing        

n = 3
f = get_random_f(n)
Uf = create_Uf(f)
print("Done creating Uf matrix")

Uf_quil_def = DefGate("Uf", Uf)  # Note: Uf_quil_def.matrix immediately gives the user the matrix representation of Uf!
Uf_gate = Uf_quil_def.get_constructor() # Get the gate constructor

start = time.time()

    
DJ_output = DJ(Uf_quil_def, Uf_gate, n, 1000)
end = time.time()
print(f)
print("It took %i seconds to complete the simulation"%(end-start))
verify_dj_output(DJ_output, f)
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
        DJ_output = DJ(Uf_quil_def, Uf_gate, n, time_out_val)
        end = time.time()
        time_reqd_arr[ind, iter_ind] = end-start
        if not(verify_dj_output(DJ_output, f)):
            print("DJ_algorithm failed to obtain the right output for n=%i"%n)
        print("done for n = %i and iteration = %i... took %i seconds"%(n,iter_ind,(end-start)))
            
        
#%% Save the data

np.savez('dj_benchmarking.npz', n_list = n_list,time_reqd_arr = time_reqd_arr)

#%% Load data

data = np.load('dj_benchmarking.npz')
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
plt.title('Execution time scaling for Deutsch-Jozsa algorithm', fontsize=25)
plt.xlabel('n (bit string length)',fontsize=20)
plt.ylabel('Average time of execution (s)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.grid()


fig.savefig('Figures/dj.png', bbox_inches='tight')
#