~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grover

Simple-use case example:

>>> from grover import run_grover
>>> result, compile_time, compute_time = run_grover(f)

#(assumes you have an oracle function f that has each bit as a different argument to f from LSB to MSB)
#(e.g. f(a,b,c) a is LSB c is MSB)
#->assumes f has n inputs and each input is a 0 or 1

Advanced use:
To run grover(z0,zf,n,a) you need to input the corresponding gate definition for z0 and zf for n qubits, the number of qubits, and the number of inputs that evaluate to 1. grover() then returns the bit string |x> that corresponds to an input x that evaluates 1.
	(This can be proven by the proof of grovers algorithm that applying G = -(H^n)zo(H^n)zf k times where k is proportional to the sqrt(2^n) sets the state of the qubits to match x with overwhelming probability.)
	
	>> z0 = create_z0(n)
		Creates the unitary matrix z0 for n qubits.
	>> z0_def = DefGate("Z0", z0)
		Create a gate object using z0
	>> zf = create_zf(f,n)
		Creates the unitary matrix z0 for n qubits.
	>> zf_def = DefGate("ZF", zf)
		Create a gate object using zf
	>> result, compile_time, computation_time = grover(z0,zf,n,a)
		Pass the oracle gate, and the value of n in order to execute the grover circuit. The function will return ket{x} and the complilation and execution time of the quantum circuit.
	

To test correctness you can pass in the oracle function and the result of Grovers algorithm to check_correctness.
	>> Correct = check_correctness(func,result)
		

If you run "python grover.py" the command line as the main argument to the python interpreter it runs exhaustive testing and benchmarking for n=1-6 qubits, with the results averaged over 5 trials, and every possible function with a=1 (the number of inputs to f(X) that equal 1 -> |x| s.t. f(x)=1 )

	>> $python grover.py

grover.py contains the following set of methods:
	
	Run_grover: can be used to run grover with an arbitrary oracle function f
	Check_correctness: checks to make sure the f(x)=1 for the x output from Grover's algorithm returns true if true.
	Convert_n_bit_string_to_int: converts an n bit string representing an unsigned int to an int
	Convert_int_to_n_bit_string: converts an int to an unsigned representation of an int
	Create_minus_gate: makes a unitary transformation that maps |x> to -|x>
	Create_zf: zf = (-1)^f(x)|x>, returns the identity matrix with -1 on the corresponding rows that f(x) returns 1
	Create_z0: z0 = -|x> if |x> = 0^n else |x>, returns the identity matrix with -1 on first row
	All_f: creates all possible functions that return f(x)=1 for a given number of qubits n, and number of inputs x that evaluate to 1
	Calc_lim: calculates how many iterations (k) of grovers algorithm should be ran to ensure it returns the correct value for a given number of qubits, and the number of inputs x that evaluate to 1
	Grover: runs grovers algorithim k times on the input oracle pyquil gate defintion z_f that corresponds to the oracle function f


