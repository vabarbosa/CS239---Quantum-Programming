import numpy as np

from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate

import time
import argparse
from argparse import RawTextHelpFormatter
import math
import re

def berenstein_vazarani(num_bits, uf_gate_def):
    # helper bit
    num_qubits = num_bits + 1

    UF = uf_gate_def.get_constructor()
    
    p = Program()
    p += uf_gate_def

    # |0^(n-1)1>
    p += X(num_qubits - 1)

    # H^n |0^(n-1)1>
    for qubit in range(num_qubits):
        p += H(qubit)

    # UF^n H^n |0^(n-1)1>
    p += UF(*range(num_qubits))

    # H^nI UF^n H^n |0^(n-1)1>
    for qubit in range(num_qubits-1):
        p += H(qubit)

    structure = "%dq-qvm" % (num_qubits)
    qc = get_qc(structure)

    start = time.time()
    results = qc.run_and_measure(p, trials=1)
    time_elapsed = int((time.time() - start) * 1000)

    a = evaluate_results(results)
    return time_elapsed, a


def evaluate_results(results):
    num_qubits = len(results) - 1
    a = [0] * num_qubits
    for index in range(len(results[0])):
        for qubit in range(num_qubits):
            a[qubit] = results[qubit][index]
    return a

def function_to_matrix(f, n):
    q = n + 1
    matrix = [ [ 0 for _ in range(2**q)] for _ in range(2**q) ]
    a = [0] * n
    for i in range(2**q):
        binary = [ int(char) for char in bin(i)[2:].zfill(q) ]
        answer = f(*binary[:-1])
        if sum(binary[:-1]) == 0:
            b = answer
        elif binary[-1] == 0 and sum(binary[:-1]) == 1:
            a[int(math.log(i//2, 2))] = answer ^ b
        binary[-1] = answer ^ binary[-1] # x^n \oplus f(x) ^ b
        col = int("".join([str(num) for num in binary]) , 2)
        matrix[i][col] = 1
        
    print("Classical programming determined a = %s and b = %d." % (a[::-1], b))
    return np.array(matrix)
                       
        
def run_berenstein_vazarani(f, n):
    zero = [0] * n
    b = f(*zero)
                    
    uf = function_to_matrix(f, n)
    gate_def = DefGate("UF", uf)
    time_elapsed, a = berenstein_vazarani(n, gate_def)
    print("Circuit ran in %d ms" % (time_elapsed))
    print("a = %s" % a)
    print("b = %d" % b)


def load_function_from_file(filename):
    with open(filename) as fd:
        function_definition = fd.read()
        # change function name to f
        function_definition = re.sub(r'def .*\(', 'def f(', function_definition, count=1)
        
        exec(function_definition, globals())
        return f

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run berenstein on a used defined function.', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--filename', type=str, required=True,
                        help="Filename of file with function definition:\n It must follow this format:\n def f(*args):\n   pass")
    parser.add_argument('--num_bits', type=int, required=True,
                        help="Number of input bits of function definition.")
    args = parser.parse_args()

    f = load_function_from_file(args.filename)
    print("Running BV on quantum simulator")
    run_berenstein_vazarani(f, args.num_bits)
