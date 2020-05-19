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

def compile_simon(num_bits, uf_gate_def):
    # helper bit
    num_qubits = num_bits * 2

    UF = uf_gate_def.get_constructor()
    
    p = Program()
    ro = p.declare('ro', 'BIT', num_qubits)
    p += uf_gate_def

    # H^n |0^(n-1)1>
    for qubit in range(num_bits):
        p += H(qubit)

    # UF^n H^n |0^(n-1)1>
    p += UF(*range(num_qubits))

    # H^nI UF^n H^n |0^(n-1)1>
    for qubit in range(num_bits):
        p += H(qubit)
    for qubit in range(num_bits):
        p += MEASURE(qubit, ro[qubit])
    
    structure = "%dq-qvm" % (num_qubits)
    qc = get_qc(structure)
    qc.compiler.client.timeout = 6000

    start = time.time()
    print("Starting compilation")
    executable = qc.compile(p)
    print("Compilation finished")

    time_elapsed = int((time.time() - start) * 1000)
    return time_elapsed, qc, executable

def simon(num_bits, qc, executable):
    num_qubits = num_bits * 2

    solver = Solver(num_bits)
    start = time.time()

    vectors_generated = 0
    while not solver.is_ready():
        results = qc.run(executable)
        y = [ measurement for measurement in results[0] ]
        print(y)
        solver.add_vector(y)
        vectors_generated += 1

    s = solver.solve()
    time_elapsed = int((time.time() - start) * 1000)

    return time_elapsed, vectors_generated, s

    
def simon2(num_bits, uf_gate_def):
    # helper bit
    num_qubits = num_bits * 2

    UF = uf_gate_def.get_constructor()
    
    p = Program()
    ro = p.declare('ro', 'BIT', num_qubits)
    p += uf_gate_def

    # H^n |0^(n-1)1>
    for qubit in range(num_bits):
        p += H(qubit)

    # UF^n H^n |0^(n-1)1>
    p += UF(*range(num_qubits))

    # H^nI UF^n H^n |0^(n-1)1>
    for qubit in range(num_bits):
        p += H(qubit)
    for qubit in range(num_bits):
        p += MEASURE(qubit, ro[qubit])
    

    structure = "%dq-qvm" % (num_qubits)
    qc = get_qc(structure)
    qc.compiler.client.timeout = 600

    #print("Start compile")
    #executable = qc.compile(p)
    #print("End compile")
    solver = Solver(num_bits)
    start = time.time()

    while not solver.is_ready():
        #results = qc.run_and_measure(p, trials=1)
        results = qc.run(executable)
        print(results)
        #y = [ results[qubit][0] for qubit in range(num_bits) ]
        y = [ measurement for measurement in results[0] ]
        solver.add_vector(y)

    s = solver.solve()
    time_elapsed = int((time.time() - start) * 1000)

    return time_elapsed, s

class Solver:
    def __init__(self, n):
        self.n = n
        self.independent_equations = {}
    def is_ready(self):
        return len(self.independent_equations) >= self.n - 1
    def add_vector(self, y):
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
        # msb refers to index
        # 01234
        for index in range(start, len(y)):
            if y[index] == 1:
                return index
        return -1
    def xor(self, x, y, start=0):
        for index in range(start, len(x)):
            y[index] = x[index] ^ y[index]
    def solve(self):
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
        return ans[::-1]


def evaluate_results(results):
    num_qubits = len(results) - 1
    a = [0] * num_qubits
    for index in range(len(results[0])):
        for qubit in range(num_qubits):
            a[qubit] = results[qubit][index]
    return a

def function_to_matrix(f, n):
    q = n * 2
    matrix = [ [ 0 for _ in range(2**q)] for _ in range(2**q) ]
    a = [0] * n
    for i in range(2**q):
        binary = [ int(char) for char in bin(i)[2:].zfill(q) ]
        answer = f(*binary[:n])
        for index in range(0, len(answer)):
            binary[n + index] ^= answer[index]
        col = int("".join([str(num) for num in binary]) , 2)
        matrix[i][col] = 1
    return np.array(matrix)

def run_simon(f, n):
    uf = function_to_matrix(f, n)
    gate_def = DefGate("UF", uf)
    compile_time, qc, executable = compile_simon(n, gate_def)
    run_time, vectors_generated, s = simon(n, qc, executable)
    #time_elapsed, s = simon(n, gate_def)
    zero = [0]*n
    if f(*zero) != f(*s):
        s = zero
    print("Compile time: %d ms" % compile_time)
    print("Run time: %d ms" % run_time)
    print("Measurements: %d" % vectors_generated)
    return compile_time, run_time, vectors_generated, s

def load_function_from_file(filename):
    with open(filename) as fd:
        function_definition = fd.read()
        # change function name to f
        function_definition = re.sub(r'def .*\(', 'def f(', function_definition, count=1)
        
        exec(function_definition, globals())
        return f

def num_to_list(i, n):
    return [int(char) for char in bin(i)[2:].zfill(n) ]

def list_to_num(s):
    return int("".join([str(num) for num in s]), 2)

def factory(mapping):
    return lambda *args: mapping[list_to_num(args)]

def generate_functions(n):
    inputs = list(range(0, 2**n))
    
    functions = []
    for i in range(2, 2**n):
        s = [int(char) for char in bin(i)[2:].zfill(n) ]
        sint = i

        mapping = {}
        random.shuffle(inputs)
        for x in inputs:
            y = x ^ sint
            if x <= y:
                mapping[x] = num_to_list(inputs[x], n)
                mapping[y] = mapping[x]

        f = factory(mapping)
        mapping = {}
        functions.append((s, f))
    return functions

    
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Run berenstein on a used defined function.', formatter_class=RawTextHelpFormatter)
    #parser.add_argument('--filename', type=str, required=True,
    #                    help="Filename of file with function definition:\n It must follow this format:\n def f(*args):\n   pass")
    #parser.add_argument('--num_bits', type=int, required=True,
    #                    help="Number of input bits of function definition.")
    #args = parser.parse_args()
    #
    #f = load_function_from_file(args.filename)
    #print("Running simon on quantum simulator")
    #run_simon(f, args.num_bits)

    fd = open("trials.csv", "w")
    fd.write("n,s,compile_trial,run_trial,compile_time,run_time,vectors_generated\n")
    for n in range(4, 5):
        functions = generate_functions(n)
        
        for s, f in functions:
            for compile_trial in range(0, 1):
                uf = function_to_matrix(f, n)
                gate_def = DefGate("UF", uf)
                compile_time, qc, executable = compile_simon(n, gate_def)
                for run_trial in range(0, 1):
                    run_time, vectors_generated, s_calc = simon(n, qc, executable)
                    zero = [0]*n
                    if f(*zero) != f(*s_calc):
                        s_calc = zero
                    if s_calc != s:
                        print("Fail:", s, s_calc)
                        exit()
                    print("Compile time: %d ms" % compile_time)
                    print("Run time: %d ms" % run_time)
                    print("Measurements: %d" % vectors_generated)
                    print("Expected s: %s, Calculated s: %s" % (s, s_calc))
                    fd.write("%d,%s,%d,%d,%d,%d,%d\n" % (n, "".join([str(num) for num in s]), compile_trial, run_trial, compile_time, run_time, vectors_generated))
    fd.close()
                    

