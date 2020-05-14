

from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import X,H,MEASURE
import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt


#%% 

#def binary_add(s1,s2):
#    """
#    Binary addition (XOR) of two bit strings s1 and s2.
#    Args: 
#        Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
#    Returns:
#        s: String s given by s = s1 + s2 (of length n) containing only zeros and ones
#    """
#    if type(s1) != str or type(s2) != str:  #If inputs are not strings, raise an error
#        raise ValueError('The inputs are not strings')
#    if sum([1 for x in s1 if (x != '0' and x != '1')]) > 0 or sum([1 for x in s2 if (x != '0' and x != '1')]) > 0:
#        raise ValueError('Input strings contain characters other than 0s and 1s') 
#    if len(s1) != len(s2):
#        raise ValueError('Input strings are not of the same length') 
#        
#    x1 = np.array([int(i) for i in s1])
#    x2 = np.array([int(i) for i in s2])
#    x = (x1 + x2 ) % 2
#    x = ''.join(str(e) for e in x.tolist())
#    return x

def give_binary_string(z,n):
    """
    Function which takes in an integer input, and outputs the corresponding binary representation
    Args: 
        z: (Integer) z must lie in the set {0,1,...,2**n- 1}
        n: (Integer) Number of bits
    Returns: 
        s: String of length n containing the binary representation of integer z.
    """
    if type(z) != int or type(n) != int:
        raise ValueError('Both arguments must be integers.')
    
    s = bin(z)
    m = n + 2 - len(s)
    if m < 0:
        raise ValueError('z is greater than the 2 raised to n. Need to input a larger value of n') 
    s = s.replace("0b", '0' * m)
    return s 


def create_random_Grover_function(n,a):
    """
    Create a function f:{0,1}^n --> {0,1} given by so that exactly a number of inputs have output 1
    Args: 
        n: integer
        a: integer less than 2**n
    Returns:
        f: Array of length 2^n. f[i] is the value of f corresponding to binary representation of i being the imput
    """
    if type(n) != int or type(a) != int:  #If inputs are not integers, raise an error
        raise ValueError('The inputs are not integers')
    if n < 0 or a < 0:
        raise ValueError('Please inputs positive integers') 
    if a >= 2**n:
        raise ValueError('Enter a valoe of a which is less than 2**n') 
        
    f = np.zeros(2**n, dtype = int)
    #choose a number of random indices with replacement
    indices = np.random.choice(list(range(2**n)), a , False)
    for index in indices:
        f[index]  = 1 
 
    return  f

#n = 3
#a = 2
#print(create_random_Grover_function(n,a))
def H_tensored(n):
    """ Returns the matrix corresponding to the tensor product of n number of Hadamard gates
    Args:
        n: Integer
    Returns:
        ndarray of size [2**n, 2**n]
    """
    H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
    n = n - 1
    H_n = H
    while n > 0:
        H_n = np.kron(H_n, H)
        n -= 1
    return H_n
    
    
    
def create_G(f):
    """
    Given a function f:{0,1}^n ---> {0,1}, creates and returns the corresponding Grover operator G (unitary matrix)
    Args:
        f: ndarray of length 2**n, consisting of integers 0 and 1
    Returns:
        G: ndarray of size [2**(n+1), 2**(n+1)], representing a unitary matrix.
    """
    if type(f) != np.ndarray:
        raise ValueError('Input the function in the form of an ndarray')
    if any((x!= 0 and x != 1) for x in f.tolist()):
        raise ValueError('The input function should only contain zeros and ones.')
    
    N = np.size(f)
    n = int(np.log2(N))
    if n != np.log2(N):
        raise ValueError("n must be a power of 2!")
    
    G = np.zeros([N,N], dtype = complex)
    Zf = np.diag([(-1)**f[i] for i in range(N)])
    Z0 = np.eye(N)
    Z0[0,0] = -1
    
    
    H_n = H_tensored(n)
    G = - H_n @ Z0 @ H_n @ Zf
    
    return G

#n = 3
#a = 2
#f = create_random_Grover_function(n,a)
#G = create_G(f)
#print(G)

def get_k_g(n, a):
    """ Obtain the optimal values of k and probability of success, given n and a
    Args:
        n: Integer
        a: Integer
    Returns:
        k: Integer
        prob: Float (Probability of success)
    """
    N = 2 ** n
    theta = np.arcsin(np.sqrt(a/N))
    k = np.pi / 4 / theta - 1/2
    k_arr = np.array([np.floor(k), np.ceil(k)])
    prob_arr = np.sin( (2 * k_arr + 1 ) * theta ) ** 2
    ind = np.argmax(prob_arr)
    return int(k_arr[ind]), prob_arr[ind]
    
    
    
    
def Grover(G_quil_def, n, a, time_out_val = 1000):
    """
    Grover's algorithm: Determines the value of and input string x, for which f(a) = 1
    Args:
        G_quil_def: DefGate object corresponding to Grover's operator G
        n: Integer, the length on input bit strings which f takes.
        a: Integer, number of inputs of f that give an output of 1
    
    Kwargs:
        time_out_val: Integer. Timeout in seconds.
        
    Returns: 
        String: String x of length 2**n
    """
    if not isinstance(G_quil_def, DefGate):
        raise ValueError("G_quil_def must be a gate definition!")
        
    G_gate = G_quil_def.get_constructor() # Get the gate constructor
    k, prob = get_k_g(n, a)
    
    ## Define the circuit
    p = Program()
    ro = p.declare('ro', 'BIT', n)
    
    for gate_ind in range(n):
        p += H(gate_ind)
      
    p += G_quil_def
    for iter_ind in range(k):
        p += G_gate(*tuple(range(n)))
    
    for gate_ind in range(n):
        p += MEASURE(gate_ind, ro[gate_ind])

    ## compile and run
    qc = get_qc(str(n)+'q-qvm') 
    qc.compiler.client.timeout = time_out_val
    executable = qc.compile(p)
    result = np.reshape(qc.run(executable),n)
    x_str = "".join(str(s) for s in result)
#    print("The measured state, and hence a = "+str(result)[1:-1])
    return x_str
#
def run_Grover(f, a, time_out_val = 1000):
    """
    Grover's algorithm: Determines the value of x which satisfies f(x)=1
    Args:
        f: ndarray of length N = 2**n, contining the values of function.
        a: Integer, number of inputs of f which give an output of 1
    Kwargs:
        time_out_val: Integer. Timeout in seconds.
    Returns: 
        Integer: String x of length 2**n
    """
    
    if type(f) != np.ndarray:
        raise ValueError('Input the function in the form of an ndarray')
    if any((x!= 0 and x != 1) for x in f.tolist()):
        raise ValueError('The input function should only contain zeros and ones.')
        
    N = np.size(f)
    n = int(np.log2(N))
    if n != np.log2(N):
        raise ValueError("f should have a size equal to a power of 2!")
    
    G = create_G(f)
    
    G_quil_def = DefGate("G", G)
    x = Grover(G_quil_def, n, a, 1000)
    return x
    

def verify_Grover_output(Grover_output, f):
    """
    Verifies if the output of Grover's algorithm is correct.
    Args:
        f: ndarray of length N = 2**n, contining the values of function f.
        Grover_output: string of length n, containing the value of a predicted by Grover's algorithm.                    
    Returns:
        is_correct: bool (TRUE if the DJ output is correct, and FALSE otherwise)
    """
    
    is_correct = f[int(Grover_output,2)] == 1
    
    return is_correct
    

#%% Trial testing 
    
n = 3
a = 1
f = create_random_Grover_function(n, a)
G = create_G(f)
print("Done creating Grover matrix")
G_quil_def = DefGate("G", G) 
Grover_output = Grover(G_quil_def, n, a, 1000)
verify_Grover_output(Grover_output, f)   

#%% Testing success rate
num_trials = 40

n = 3
a = 3
num_correct = 0
for trial_ind in range(num_trials):
    f = create_random_Grover_function(n, a)
    Grover_output = run_Grover(f, a, 1000)
    print("i=%i done"%trial_ind)
    if verify_Grover_output(Grover_output, f):
        num_correct += 1

numerical_success_rate = num_correct / num_trials
k , prob = get_k_g(n, a)
print(" Numerical success rate = %f"%numerical_success_rate)
print(" Theoretical success rate = %f" %prob)