import numpy as np

from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate

import time
import argparse
from argparse import RawTextHelpFormatter
import math
import re

CONSTANT=0
BALANCED=1

def deutsch_jozsa(num_bits, uf_gate_def):
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
    results = qc.run_and_measure(p, trials=10)
    time_elapsed = int((time.time() - start) * 1000)

    if evaluate_results(results) == 0:
        function_type = CONSTANT
        type_name = "constant"
    else:
        function_type = BALANCED
        type_name = "balanced"

    print("Circuit ran in %d ms" % (time_elapsed))
    print("Function is %s" % type_name)
    return time_elapsed, function_type

def evaluate_results(results): 
    is_nonzero = [ 0 ] * len(results[0])
    num_qubits = len(results) - 1
    for index in range(len(is_nonzero)):
        for qubit in range(num_qubits):
            is_nonzero |= results[qubit][index]
    return sum(is_nonzero)

def function_to_matrix(f, n):
    q = n + 1
    matrix = [ [ 0 for _ in range(2**q)] for _ in range(2**q) ]
    output = 0
    for i in range(2**q):
        binary = [ int(char) for char in bin(i)[2:].zfill(q) ]
        answer = f(*binary[:-1])
        output += int(math.pow(-1, answer))
        binary[-1] = answer ^ binary[-1] # x^n \oplus f(x) ^ b
        col = int("".join([str(num) for num in binary]) , 2)
        matrix[i][col] = 1
    if output == 0:
        type_name = "balanced"
    else:
        type_name = "constant"
        
    print("Classical programming determined this function to be %s" % (type_name))
    return np.array(matrix)
        
def run_deutsch_jozsa(f, n):
    uf = function_to_matrix(f, n)
    gate_def = DefGate("UF", uf)
    time_elapsed, function_type = deutsch_jozsa(n, gate_def)

def load_function_from_file(filename):
    with open(filename) as fd:
        function_definition = fd.read()
        # change function name to f
        function_definition = re.sub(r'def .*\(', 'def f(', function_definition, count=1)
        
        exec(function_definition, globals())
        return f

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run deutsch-jozsa on a used defined function.', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--filename', type=str, required=True,
                        help="Filename of file with function definition:\n It must follow this format:\n def f(*args):\n   pass")
    parser.add_argument('--num_bits', type=int, required=True,
                        help="Number of input bits of function definition.")
    args = parser.parse_args()

    f = load_function_from_file(args.filename)
    print("Running DJ on quantum simulator")
    run_deutsch_jozsa(f, args.num_bits)

    
    
