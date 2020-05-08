# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:50:05 2020

@author: sathe
"""

from pyquil import Program, get_qc
from pyquil.latex import display
from pyquil.quil import DefGate
from pyquil.gates import X,H,MEASURE
from itertools import combinations 
import numpy as np
import random as rd
import time



#%% Define the function needed
def binary_add(s1,s2):
    """
    Binary addition (XOR) of two bit strings s1 and s2.
    Input: Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
    Output: A string s = s1 + s2 (of length n) containing only zeros and ones
    """
    x1 = np.array([int(i) for i in s1])
    x2 = np.array([int(i) for i in s2])
    x = (x1 + x2 ) % 2
    x = ''.join(str(e) for e in x.tolist())
    return x


def give_binary_string(z,n):
    """
    function which takes in an integer input, and outputs the corresponding binary representation
    Input: 
        z: Integer. z must lie in the set {0,1,...,2**n- 1}
        n: Integer specifying the number of bits
    Output: 
        s: String of length n containing the binary representation of integer z.
    """
    s = bin(z)
    m = n + 2 - len(s)
    s = s.replace("0b", '0' * m)
    return s 


def create_all_functions(n):
    """
    Creates all possible constant and balanced functions for n bits
    Input:
        n: Integer
    Output:
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
    Input:
        n: Integer. Denotes the number of bits.
    Output:
        ndarray of length n, containing 0s and 1s.
    """
    func_list_balanced, func_list_constant = create_all_functions(n)
    num_balanced = np.shape(func_list_balanced)[0]
    random_index = rd.randint(0,num_balanced+1)
    if random_index >= num_balanced:
        return func_list_constant[random_index % num_balanced]
    else:
        return func_list_balanced[random_index]

#random_f = get_random_f(2)    


def create_Uf(f):
    """
    Given a function f:{0,1}^n ---> {0,1}, creates and returns the corresponding oracle (unitary matrix) Uf
    Input:
        f: ndarray of length n, consisting of integers 0 and 1
    Output:
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

def verify_quantum_output(DJ_output, f):
    """
    Verifies if the output of DJ algorithm is correct.
    Input:
        f: ndarray of length N, contining the values of function.
        DJ_output: Integer.
                    1 if function is constant, and 0 if the function is balanced
                    
    Output:
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


#%% Define DJ algorithm which predicts whether the function is balanced or constant
    


n = 3
f = get_random_f(n)
Uf = create_Uf(f)
print("Done creating Uf matrix")

Uf_quil_def = DefGate("Uf", Uf)
Uf_gate = Uf_quil_def.get_constructor() # Get the gate constructor


def DJ(Uf_quil_def, Uf_gate, n, time_out_val = 100):
    """
    Deutsch-Jozsa algorithm
    Input:
        Uf_gate object: which acts on n+1 qubits
        Uf_quil_def: DefGate object
        n: Integer, the length on input bit strings which f takes.
        time_out_val: Integer. Timeout in seconds.
        
    Output: 
        Integer: 0 if function is balanced, and 1 if function is constant
    """

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
        
    qc = get_qc(str(n+1)+'q-qvm') 
    qc.compiler.client.timeout = time_out_val
    
    executable = qc.compile(p)
    result = np.reshape(qc.run(executable),n)
    print("Measured state is "+str(result)[1:-1])
    
    if np.sum(result) != 0:
        print("Function is balanced")
        return 0
    else:
        print("function is constant")
        return 1
    
DJ_output = DJ(Uf_quil_def, Uf_gate, n)
verify_quantum_output(DJ_output, f)
#%% 


### Testing
time_out_val = 5000
n_min = 1
n_max = 8
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
        if not(verify_quantum_output(DJ_output, f)):
            print("DJ_algorithm failed to obtain the right output for n=%i"%n)
        print("done for n = %i and iteration = %i"%(n,iter_ind))
            
        


