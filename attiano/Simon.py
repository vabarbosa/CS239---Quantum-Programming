# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:46:19 2020

@author: sathe
"""


from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
from itertools import combinations 
import numpy as np


#%% Given some value of bitstring s, obtain the corresponding function f
def ab_function(s):
    """
    Input: 
        string s of length n
    Output:
        array f of size 2^n. f[i] contains the value of f corresponding to i. The value stored in f is in decimal.
        f is such that, f(x) = f(y) iff x + y \in {0,s}
    """
    n = len(s)
    f = np.nan([2**n,n])
    
    N = 2**n
    
    
    if s == '0'*n:
        perm_random = np.random.permutation(N)
        for i in range(N):
            f[i,:] = perm_random
    
    
    
    
    list_of_numbers = list(range(N))
    comb = combinations(list_of_numbers, int(N/2))
    func_list_balanced = []
    
    for i in list(comb):
        out_list = np.zeros(N)
        for j in list(i):
            out_list[j] = 1
        func_list_balanced.append(out_list)
    
    func_list_constant = [np.zeros(N), np.ones(N)]
    
    
    
    if s == '0'*n:
        
    
    a = np.array([int(i) for i in a])
    b = np.array([int(i) for i in b])
    for index_num in range(2**n):   
        bin_rep = bin(index_num)
        m = n + 2 - len(bin_rep)
        bin_rep = bin_rep.replace("0b", '0' * m)
        x = np.array([int(i) for i in bin_rep])  # Binary representation of index_num, in array form
        f[index_num] = np.sum((x * a + b)) % 2
    return  f

#a = '101'
#b = '001'
#print(ab_function(a,b))