# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:50:05 2020

@author: sathe
"""

#, get_qc
#from pyquil.gates import *
#import numpy as np
## construct a Bell State program
#p = Program(H(0), CNOT(0, 1))
## run the program on a QVM
#qc = get_qc('9q-square-qvm')
#result = qc.run_and_measure(p, trials=10)
#print(result[0])
#print(result[1])



## First, I want to run the DJ algorithm for f:{0,1} -> {0,1}
## Consider f(0)=f(1)=0
## Then, Uf |xy>=|xy>. or Uf = identity


from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
from itertools import combinations 
import numpy as np


#%% Trial processing for constant f -- seems to work
#Uf_matrix = np.eye(4)
##print(Uf_matrix)
## Get the Quil definition for the new gate
#Uf_quil_def = DefGate("Uf", Uf_matrix)
## Get the gate constructor
#Uf_gate = Uf_quil_def.get_constructor()
#
#p = Program()
#ro = p.declare('ro', 'BIT', 1)
#p += X(1)
#p += H(0)
#p += H(1)
#p += Uf_quil_def
#p += Uf_gate(0,1)
#p += H(0)
#p += MEASURE(0, ro[0])
#print(p)
#
#qc = get_qc('2q-qvm')  # You can make any 'nq-qvm' this way for any reasonable 'n'
#executable = qc.compile(p)
#
#num_runs = 10
#result = [qc.run(executable)[0,0] for i in range(num_runs)]
#print(result)


#%% Create all possible functions 

def create_function(n):
    """
    Takes as input an integer n, and outputs a list of functions f:{0,1}^n --> {0,1} which are either balanced or constant
    """
    N = 2**n
    list_of_numbers = list(range(N))
    comb = combinations(list_of_numbers, int(N/2))
    func_list_balanced = []
    
    for i in list(comb):
#        print(i)
        out_list = np.zeros(N)
        for j in list(i):
            out_list[j] = 1
        func_list_balanced.append(out_list)
    
    func_list_constant = [np.zeros(N), np.ones(N)]
    
    
    return func_list_balanced, func_list_constant
        

n = 4
func_list_balanced, func_list_constant = create_function(n)
#print(func_list_balanced)
#print(func_list_constant)
        
#%% Create Uf for a given f

#n=5
#for i in range(2**n):
#    bin_rep = bin(i)
#    m = n + 2 - len(bin_rep)
#    bin_rep = bin_rep.replace("0b", '0' * m)
#    print(bin_rep)

def create_Uf(f,n):
    """
    f is an array
    n is an integer (number of bits on which f acts)
    """
    N = 2**(n+1)
    
    Uf = np.zeros([N,N])
    
    for i in range(2**n):
        f_val = int(f[i])
        
        bin_rep = bin(i)
        m = n + 2 - len(bin_rep)
        bin_rep = bin_rep.replace("0b", '0' * m)
        print(bin_rep)
        inp_1 = bin_rep + '0'
        out_1 = bin_rep + str(f_val)
        Uf[int(out_1,2),int(inp_1,2)] = 1
        
        inp_2 = bin_rep + '1'
        out_2 = bin_rep + str(int(not(f_val)))
        Uf[int(out_2,2),int(inp_2,2)] = 1
    return Uf
#print(func_list_balanced[2])
Uf = create_Uf(func_list_balanced[2], n)
#print(Uf)
    


#%% Define DJ algorithm which predicts whether the function is balanced or constant
    


n = 4
func_list_balanced, func_list_constant = create_function(n)
Uf_matrix = create_Uf(func_list_balanced[2], n)
print("Done creating Uf matrix")
Uf_quil_def = DefGate("Uf", Uf_matrix)
# Get the gate constructor
Uf_gate = Uf_quil_def.get_constructor()


def DJ(Uf_quil_def, Uf_gate, n):
    """
    Deutsch-Jozsa algorithm
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
    print("Measured state is "+str(result)[1:-1])
    
    if np.sum(result) != 0:
        print("Function is balanced")
        return 0
    else:
        print("function is constant")
        return 1
    
DJ_output = DJ(Uf_quil_def, Uf_gate, n)

#%%


#Trying out stuff
def func(a,b,c,d):
    print(a+b+c+d)

func(*tuple(range(4)))

