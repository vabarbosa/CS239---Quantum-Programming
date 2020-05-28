# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:26:13 2020

@author: Pratik
"""

# First qiskit file

#%%
#from qiskit import IBMQ
#IBMQ.save_account('d5797c2dfdae5724272e4a64865c101a5bbe11051a97f5b31d0ed8aefdf1ff7c823aab336c87065b287069c7f571f2721d5554146743b11bb19d00cd1efd7e6c')

#%%

import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
circuit.draw(output = 'mpl')

plot_histogram(counts)