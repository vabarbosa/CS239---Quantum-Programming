# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:46:19 2020

@author: sathe
"""


from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
#from itertools import combinations 
import numpy as np
import random as rd


#%% Given some value of bitstring s, obtain the corresponding function f


## function which takes in an integer input, and outputs the corresponding binary representation
def give_binary_string(z,n):
    """
    Input: Integer z
            Integer n specifying the number of bits
    Output: String s of length n containing the binary representation of integer z.
    """
    s = bin(z)
    m = n + 2 - len(s)
    s = s.replace("0b", '0' * m)
    return s 

def binary_add(s1,s2):
    """
    Input: Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
    Output: A string s = s1 + s2 (of length n) containing only zeros and ones
    """
    x1 = np.array([int(i) for i in s1])
    x2 = np.array([int(i) for i in s2])
    x = (x1 + x2 ) % 2
    x = ''.join(str(e) for e in x.tolist())
    return x


#%%

def s_function(s):
    """
    Input: 
        string s of length n
    Output:
        array f of size 2^n. f[i] contains the value of f corresponding to i. The value stored in f is in decimal.
        f is such that, f(x) = f(y) iff x + y \in {0,s}
    """
    n = len(s)
    N = 2**n
    f = np.ones(N)*-1
    
    perm_random = np.random.permutation(N)
    if s == '0'*n:
        for i in range(N):
            f = np.copy(perm_random)
    else:
        chosen_perm = perm_random[0:int(N/2)]
        ind_1 = 0
        for i in range(N):
            if f[i] == -1:
                f[i] = chosen_perm[ind_1]
                f[int(binary_add(give_binary_string(i,n) , s), 2)] = chosen_perm[ind_1]
                ind_1 += 1
    return f
        

############## Seems to work
############## Testing if this has worked
#n = 5
#s = give_binary_string(rd.randint(0,2**n-1),n)
#print('s=%s'%s)
#func_arr = s_function(s)
#for i in range(2**n):
#    print(give_binary_string(i,n) ,'\t',give_binary_string(int(func_arr[i]),n),'\t',int(func_arr[i]))
###############
###############
    
#%% Create U_f matrix given f_array
    
def create_Uf(f,n):
    """
    Input: f is an array of length 2**n, and contains integer values
            n is the length of bit-strings which the function in Simon's problem takes as input
    Output: Unitary matrix(array) Uf corresponding to f
    """

    N = 2**(2*n)
    
    Uf = np.zeros([N,N])
    
    for i in range(2**n):
        f_val = int(f[i])
        
        bin_rep_i = give_binary_string(i,n)
        bin_rep_f = give_binary_string(f_val,n)
        
        for j in range(2**n):
            attachment = give_binary_string(j,n)
            inp = bin_rep_i + attachment
            out = bin_rep_i + binary_add(attachment,bin_rep_f)
            Uf[int(out,2),int(inp,2)] = 1

    return Uf

############## Seems to work
############## Testing if this has worked
#n = 2
#s = give_binary_string(rd.randint(0,2**n-1),n)
#print('s=%s'%s)
#func_arr = s_function(s)
#Uf_matrix = create_Uf(func_arr,n)
#print('Uf=',Uf_matrix)
###############
###############
    

#%% Simon's algorithm quantum




def Simon_quantum(Uf_quil_def, Uf_gate, n):
    """
    Simon's algorithm quantum
    Input:
        Uf_gate object: which acts on n+1 qubits
        n: Integer, the length on input bit strings which f takes.
        
    Output: 
        String y of length 2**n corresponding to one bit-string which satisfies y.s = 0
    """

    p = Program()
    ro = p.declare('ro', 'BIT', 2 * n)
    for gate_ind in range(n):
        p += H(gate_ind)
    p += Uf_quil_def
    p += Uf_gate(*tuple(range(2 * n)))
    for gate_ind in range(n):
        p += H(gate_ind)
    for gate_ind in range(n):
        p += MEASURE(gate_ind, ro[gate_ind])
#    print(p)

    qc = get_qc(str(n*2)+'q-qvm') 
    executable = qc.compile(p)
    result = np.reshape(qc.run(executable),n)
    y = str(result)[1:-1]
    print("y = " + y)
    return y
  
#%% Testing
n = 2
s = '01'
func_arr = s_function(s)
Uf_matrix = create_Uf(func_arr,n)
print("Done creating Uf matrix")
Uf_quil_def = DefGate("Uf", Uf_matrix)
# Get the gate constructor
Uf_gate = Uf_quil_def.get_constructor()
#%%


y = Simon_quantum(Uf_quil_def, Uf_gate, n)
        