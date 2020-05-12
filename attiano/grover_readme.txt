grover.py contains a set of methods needed to run grovers algorithm in pyquil.

	Check_correctness: checks to make sure the f(x)=1 for the x output from Grover's algorithm
	Convert_n_bit_string_to_int: converts an int to an unsigned representation of an int
	Create_minus_gate: makes a unitary transformation that maps |x> to -|x>
	Create_zf: zf = (-1)^f(x)|x>, returns the identity matrix with -1 on the corresponding rows that f(x) returns 1
	Create_z0: z0 = -|x> if |x> = 0^n else |x>, returns the identity matrix with -1 on first row
	All_f: creates all possible functions that return f(x)=1 for a given number of qubits n, and number of inputs x that evaluate to 1
	Calc_lim: calculates how many iterations (k) of grovers algorithm should be ran to ensure it returns the correct value for a given number of qubits, and the number of inputs x that evaluate to 1
	Grover: runs grovers algorithim k times on the input oracle pyquil gate defintion z_f that corresponds to the oracle function f

To run grover(z0,zf,n,a) you need to input the corresponding gate definition for z0 and zf for n qubits, the number of qubits, and the number of inputs that evaluate to 1. grover() then returns the bit string |x> that corresponds to an input x that evaluates 1.
	This can be proven by the proof of grovers algorithm that applying G = -(H^n)zo(H^n)zf k times where k is proportional to the sqrt(2^n) sets the state of the qubits to match x with overwhelming probability.


If you run it through the command line as the main argument to the python interpreter it runs exhaustive testing and benchmarking for n=1-6 qubits, with the results averaged over 5 trials, and every possible function with a=1 (the number of inputs to f(X) that equal 1 -> |x| s.t. f(x)=1 )