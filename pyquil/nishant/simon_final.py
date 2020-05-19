import numpy as np

from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate

import time
import argparse
from argparse import RawTextHelpFormatter
import math
import re
import random
import matplotlib.pyplot as plt

m = 5

def compile_simon(Uf_quil_def, Uf_gate, n, time_out_val = 100):
    """
    Compile simon circuit
    Args:
        Uf_quil_def: DefGate object corresponding to oracle Uf
        Uf_gate: gate object which acts on 2n qubits
        n: Integer, the length on input bit strings which f takes.
    
    Kwargs:
        time_out_val: Integer. Timeout in seconds.
        
    Returns: 
        int, qc, executable: compilation time, quantum circuit, and compiled program
    """
    
    # helper bit
    num_qubits = n * 2

    p = Program()
    ro = p.declare('ro', 'BIT', num_qubits)

    # H^(2n) |0^(2n)>
    for gate_ind in range(n):
        p += H(gate_ind)

    p += Uf_quil_def
        
    # UF H^(2n) |0^(2n)>
    p += Uf_gate(*range(num_qubits))

    # H^nI^n UF H^(2n) |0^(2n)>
    for gate_ind in range(n):
        p += H(gate_ind)
    for gate_ind in range(n):
        p += MEASURE(gate_ind, ro[gate_ind])
    
    qc = get_qc(str(num_qubits)+'q-qvm') 
    qc.compiler.client.timeout = time_out_val

    start = time.time()
    print("Starting compilation")
    executable = qc.compile(p)
    print("Compilation finished")

    time_elapsed = int((time.time() - start) * 1000)
    return time_elapsed, qc, executable

def simon(num_bits, qc, executable):
    """
    Run simon's algorithm. Measures the quantum circuit and runs classical solver to determine s
    Args:
        num_bits: input arity of function
        qc: quantum circuit
        executable: compiled program
    
    Returns: 
        int, int, list<int>: run time, numbers of vectors measured before finding solution, s that was solved for
    """
    
    num_qubits = num_bits * 2

    solver = Solver(num_bits)
    start = time.time()

    vectors_generated = 0
    max_measurements = 4 * m * (num_bits - 1)
    while not solver.is_ready() and vectors_generated <= max_measurements:
        results = qc.run(executable)
        y = [ measurement for measurement in results[0] ]
        print(y)
        solver.add_vector(y)
        vectors_generated += 1

    if not solver.is_ready():
        print("Failure: Measured %d times and only got %d independent y's." % (max_measurements, len(solver.independent_equations)))
    s = solver.solve()
    time_elapsed = int((time.time() - start) * 1000)

    return time_elapsed, vectors_generated, s[0]

class Solver:
    def __init__(self, n):
        """
        Classical solver intended to track independent equations and solve for s once n-1 equations have been found.
        Args:
            n: number of bits - arity of f
        """
        self.n = n
        self.independent_equations = {}
    def is_ready(self):
        """
        Checks if there are sufficient equations to solve for s
        Returns:
            boolean: if equation can be solved
        """

        return len(self.independent_equations) >= self.n - 1
    def add_vector(self, y):
        """
        Add vector to set of equations. It will check for independence before adding it.
        Args:
            n: number of bits - arity of f
        Returns:
            None
        """
        # if the msb doesn't exist, y is independent so far
        msb = self.most_significant_bit(y)
        # keep subtracting existing vectors from y
        while msb >= 0 and msb in self.independent_equations:
            self.xor(self.independent_equations[msb], y, msb)
            msb = self.most_significant_bit(y, msb)
        # if no msb, throw it away, o.w. save it
        if msb >= 0:
            self.independent_equations[msb] = y
    def most_significant_bit(self, y, start=0):
        """
        Helper function for finding MSB
        Args:
            y: equation
            start: index to start at
        Returns:
            int: index of MSB
        """
        # msb refers to index
        # 01234
        for index in range(start, len(y)):
            if y[index] == 1:
                return index
        return -1
    def xor(self, x, y, start=0):
        """
        Helper function for y ^= x
        Args:
            x: equation
            y: equation
            start: index to start at
        Returns:
            None
        """
        
        for index in range(start, len(x)):
            y[index] = x[index] ^ y[index]
    def solve(self):
        """
        Solve the system of equations using gaussian elimination. As a fail safe, if is called with insufficient equations, it uses brute force and returns all possible values of s
        Returns:
            List<List<int>>: values for s. Length is 1 if sufficient equations
        """
        
        if not self.is_ready():
            return self.brute_force()
        system = []
        #0001
        #0101
        #1000
        ans = [0]*self.n
        for msb in range(self.n-1, -1, -1):
            if msb in self.independent_equations:
                system.append(self.independent_equations[msb])
            else:
                # assume missing equation is 1
                equation = [0] * self.n
                equation[msb] = 1
                ans[self.n - 1 - msb] = 1
                system.append(equation)

        # for each row, check if any row below it has the bit set
        # if it's set, add the row's answer
        for low in range(0, len(system)-1):
            yindex = self.n - 1 - low
            for high in range(low+1, len(system)):
                if system[high][yindex] == 1:
                    ans[high] = ans[low] ^ ans[high]

        # still needs to be checked with f(0) == f(ans)
        return [ans[::-1]]
    def brute_force(self):
        """
        Back up brute force method if insufficient equations.
        Returns:
            List<List<int>>: values for s. 
        """
        
        # This should rarely get called.
        # Only gets called if we have < n - 1 independent equations
        # We can keep increasing m to reduce the probability of ending up here
        valid = []
        for i in range(0, self.n):
            binary = [ int(char) for char in bin(i)[2:].zfill(self.n) ]
            if self.valid_s(s):
                valid.append(s)
        return valid
    def valid_s(self, s):
        for msb in self.independent_equations:
            y = self.independent_equations[msb]
            answer = 0
            for index in range(0, self.n):
                answer ^= (s[index] * y[index])
            if answer != 0:
                return False
        return True
        
def create_Uf(f):
    """
    Given a function f:{0,1}^n ---> {0,1}, creates and returns the corresponding oracle (unitary matrix) Uf
    Args:
        f: ndarray of length n, consisting of integers 0 and 1
    Returns:
        Uf: ndarray of size [2**(n+1), 2**(n+1)], representing a unitary matrix.
    """
    two_raized_n = len(f)
    n = int(np.log2(two_raized_n))
    q = n * 2
    N = 2**q
    Uf = np.zeros([N,N], dtype = complex )

    for i in range(2**q):
        binary = [ int(char) for char in bin(i)[2:].zfill(q) ]
        answer = f[list_to_num(binary[:n])]

        for index in range(0, len(answer)):
            binary[n + index] ^= answer[index]
        col = int("".join([str(num) for num in binary]) , 2)
        Uf[i, col] = 1
    return Uf

def num_to_list(i, n):
    """
    Helper function to create a binary list of ints
    Args:
        i: number to convert
        n: length of bits
    Returns:
        List<int>: binary representation
    """
    
    return [int(char) for char in bin(i)[2:].zfill(n) ]

def list_to_num(s):
    """
    Helper function to create an int from a binary string
    Args:
        s: the string to convert
    Returns:
        int: the integer representation
    """
    
    return int("".join([str(num) for num in s]), 2)

def generate_functions(n):
    """
    Create many function f:{0,1}^n --> {0,1}^n given by f(x) = f(y) iff x ^ y in {0, s}
    for all possible s.
    Generates all possible values for s, then selects a mapping at random.
    Args: 
        n: length of inputs
    Returns:
        list(s, f): Array of length 2^n. f[i] is the value of f corresponding to binary representation of i being the imput. s is the secret.
        f(x) = f(y) iff x ^ y in {0, s}
    """
    inputs = list(range(0, 2**n))
    
    functions = []
    for i in range(0, 2**n):
        s = [int(char) for char in bin(i)[2:].zfill(n) ]
        sint = i

        f = [0] * (2**n)
        random.shuffle(inputs)
        for x in inputs:
            y = x ^ sint
            if x <= y:

                f[x] = num_to_list(inputs[x], n)
                f[y] = f[x]

        functions.append((s, f))
    return functions

def plot_uf_variance():
    num_times = 100
    time_out_val = 10000
    time_array = np.zeros(num_times)
    n = 2
    functions = generate_functions(n)
    for iter_ind in range(num_times):
        functions = generate_functions(n)
        s, f = functions[random.randint(0, 2**n - 1)]
        uf = create_Uf(f)
        Uf_quil_def = DefGate("UF", uf)
        Uf_gate = Uf_quil_def.get_constructor()
        compile_time, qc, executable = compile_simon(Uf_quil_def, Uf_gate, n, time_out_val)
        run_time, vectors_generated, s_calc = simon(n, qc, executable)
        print("Compile time: %d ms" % compile_time)
        print("Run time: %d ms" % run_time)
        print("Measurements: %d" % vectors_generated)
        print("Expected s: %s, Calculated s: %s" % (s, s_calc))

        time_array[iter_ind] = (compile_time + run_time) / 1000.0
            
    #%% Save the data
    
    np.savez('dependence.npz', num_times = num_times,time_array = time_array, n = n, )
        
    #%% Load data
        
    data = np.load('dependence.npz')
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


    fig.savefig('figures/simon_hist.png', bbox_inches='tight')


def plot_time(npz_path):
    data = np.load('simon_benchmarking.npz')

    n_list = data['n_list']
    comp_time_arr = data['comp_time_arr']
    run_time_arr = data['run_time_arr'][1:]
    time_per_msmt = data['time_per_msmt'][1:]
    print(n_list)
    print(comp_time_arr)
    print(run_time_arr)
    print(time_per_msmt)
    
    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(16,10))
    
    z = np.polyfit(n_list, comp_time_arr, 10)
    p = np.poly1d(z)

    plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
    plt.plot(n_list, comp_time_arr, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
    plt.title('Compilation time scaling for Simon algorithm', fontsize=25)
    plt.xlabel('n (bit string length)',fontsize=20)
    plt.ylabel('Average time of compilaion (s)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.grid()

    fig.savefig('figures/simon_comp.png', bbox_inches='tight')

    n_list = n_list[1:]
    
    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(16,10))
    
    z = np.polyfit(n_list, run_time_arr, 10)
    p = np.poly1d(z)

    plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
    plt.plot(n_list, run_time_arr, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
    plt.title('Running time scaling for Simon algorithm', fontsize=25)
    plt.xlabel('n (bit string length)',fontsize=20)
    plt.ylabel('Average time of run (s)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.grid()

    fig.savefig('figures/simon_run.png', bbox_inches='tight')
    
    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(16,10))
    
    z = np.polyfit(n_list, run_time_arr, 10)
    p = np.poly1d(z)

    plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
    plt.plot(n_list, time_per_msmt, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
    plt.title('Measurement time scaling for Simon algorithm', fontsize=25)
    plt.xlabel('n (bit string length)',fontsize=20)
    plt.ylabel('Average time of each measurement (s)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.grid()

    fig.savefig('figures/simon_msmt.png', bbox_inches='tight')
        
if __name__ == "__main__":
    plot_uf_variance()
    exit()
    
    time_out_val = 10000
    n_min = 1
    n_max = 3
    n_list = list(range(n_min,n_max+1))

    num_compilation_trials = 1
    num_run_trials = 1
    
    comp_time_arr = np.zeros((n_max-n_min)+1)
    run_time_arr = np.zeros((n_max-n_min)+1)
    time_per_msmt = np.zeros((n_max-n_min)+1)

    for ind, n in enumerate(n_list):
        functions = generate_functions(n)

        total_compilation_time = 0
        total_run_time = 0
        total_vectors = 0
        for s, f in functions:
            for compile_trial in range(0, num_compilation_trials):
                uf = create_Uf(f)
                Uf_quil_def = DefGate("UF", uf)
                Uf_gate = Uf_quil_def.get_constructor()
                compile_time, qc, executable = compile_simon(Uf_quil_def, Uf_gate, n, time_out_val)
                total_compilation_time += compile_time
                for run_trial in range(0, num_run_trials):
                    run_time, vectors_generated, s_calc = simon(n, qc, executable)

                    # Only need to call f, is zero is a possibility
                    sint = list_to_num(s_calc)
                    if f[0] != f[sint]:
                        s_calc = [0]*n
                    if s_calc != s:
                        print("Fail:", s, s_calc)
                        exit()
                    print("Compile time: %d ms" % compile_time)
                    print("Run time: %d ms" % run_time)
                    print("Measurements: %d" % vectors_generated)
                    print("Expected s: %s, Calculated s: %s" % (s, s_calc))

                    total_run_time += run_time
                    total_vectors += vectors_generated
        comp_time_arr[ind] = total_compilation_time / float((2**n) * num_compilation_trials) / 1000
        run_time_arr[ind] = total_run_time / float((2**n) * num_compilation_trials * num_run_trials) / 1000
        time_per_msmt[ind] = (run_time_arr[ind] / total_vectors if total_vectors != 0 else 0.0) / 1000
                               
                               
    np.savez('simon_benchmarking.npz', n_list = n_list, comp_time_arr = comp_time_arr, run_time_arr = run_time_arr, time_per_msmt = time_per_msmt)
    plot_time('simon_benchmarking.npz')
    
    plot_uf_variance()
                    

