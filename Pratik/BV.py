# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:58:42 2020

@author: sathe
"""

from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
from itertools import combinations 
import numpy as np



#%% 
#### Maybe change this to a lambda function?
def ab_function(a,b):
    """
    Input: 
        string a of length n
        string b of length n
        All arrays only contain integers 0 and 1
    Output:
        array f of length 2^n. f[i] is the value of f corresponding to binary representation of i being the imput
        f = a . x + b
    """
    n = len(a)
    f = np.zeros(2**n)
    a = np.array([int(i) for i in a])
    b = np.array([int(i) for i in b])
    for index_num in range(2**n):   
        bin_rep = bin(index_num)
        m = n + 2 - len(bin_rep)
        bin_rep = bin_rep.replace("0b", '0' * m)
        x = np.array([int(i) for i in bin_rep])  # Binary representation of index_num, in array form
        f[index_num] = np.sum((x * a + b)) % 2
    return  f

a = '101'
b = '001'
print(ab_function(a,b))



#%% Create all possible functions 

#def create_function(n):
#    """
#    Takes as input an integer n, and outputs a list of functions f:{0,1}^n --> {0,1} which are either balanced or constant
#    """
#    N = 2**n
#    list_of_numbers = list(range(N))
#    comb = combinations(list_of_numbers, int(N/2))
#    func_list_balanced = []
#    
#    for i in list(comb):
#        out_list = np.zeros(N)
#        for j in list(i):
#            out_list[j] = 1
#        func_list_balanced.append(out_list)
#    
#    func_list_constant = [np.zeros(N), np.ones(N)]
#    
#    
#    return func_list_balanced, func_list_constant
#        
#
#n = 3
#func_list_balanced, func_list_constant = create_function(3)

#print(func_list_balanced)
#print(func_list_constant)
    

#%% Create Uf matrix from f

def create_Uf(f,n):
    """
    Input: f is an array of size 2**n
            n is the length of bit-strings which f takes as input
    Output: Unitary matrix(array) Uf corresponding to f
    """

    N = 2**(n+1)
    
    Uf = np.zeros([N,N])
    
    for i in range(2**n):
        f_val = int(f[i])
        
        bin_rep = bin(i)
        m = n + 2 - len(bin_rep)
        bin_rep = bin_rep.replace("0b", '0' * m)
        
        inp_1 = bin_rep + '0'
        out_1 = bin_rep + str(f_val)
        Uf[int(out_1,2),int(inp_1,2)] = 1
        
        inp_2 = bin_rep + '1'
        out_2 = bin_rep + str(int(not(f_val)))
        Uf[int(out_2,2),int(inp_2,2)] = 1
    return Uf
#n= 3
#create_Uf(ab_function(a,b),n)
#%% BV circuit

n = 3
a = '100'
b = '101'
Uf_matrix = create_Uf(ab_function(a,b), n)
print("Done creating Uf matrix")
Uf_quil_def = DefGate("Uf", Uf_matrix)
# Get the gate constructor
Uf_gate = Uf_quil_def.get_constructor()


def BV(Uf_quil_def, Uf_gate, n):
    """
    Bernstrin-Vazirani algorithm
    Input:
        Uf_gate object: which acts on n+1 qubits
        n: Integer, the length on input bit strings which f takes.
        
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
#    print(p)

    qc = get_qc(str(n+1)+'q-qvm') 
    executable = qc.compile(p)
    result = np.reshape(qc.run(executable),n)
    print("The measured state, and hence a = "+str(result)[1:-1])
    return str(result)[1:-1]
    
BV_output = BV(Uf_quil_def, Uf_gate, n)
        