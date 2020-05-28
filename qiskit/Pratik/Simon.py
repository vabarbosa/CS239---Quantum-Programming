import numpy as np
from qiskit import(QuantumCircuit, execute, Aer)
from qiskit.compiler import transpile
import random as rd
import time
import matplotlib.pyplot as plt
#%% 
class Program:
    def binary_add(self, s1,s2):
        """
        Binary addition (XOR) of two bit strings s1 and s2.
        Args: 
            Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
        Returns:
            s: String s given by s = s1 + s2 (of length n) containing only zeros and ones
        """
        if type(s1) != str or type(s2) != str:  #If inputs are not strings, raise an error
            raise ValueError('The inputs are not strings')
        if sum([1 for x in s1 if (x != '0' and x != '1')]) > 0 or sum([1 for x in s2 if (x != '0' and x != '1')]) > 0:
            raise ValueError('Input strings contain characters other than 0s and 1s') 
        if len(s1) != len(s2):
            raise ValueError('Input strings are not of the same length') 
            
        x1 = np.array([int(i) for i in s1])
        x2 = np.array([int(i) for i in s2])
        x = (x1 + x2 ) % 2
        x = ''.join(str(e) for e in x.tolist())
        return x


    def dot_product(self, s1,s2):
        """
        Dot product of strings s1 and s2.
        Args: 
            Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
        Returns:
            prod: Integer (0 or 1)
        """
        if type(s1) != str or type(s2) != str:  #If inputs are not strings, raise an error
            raise ValueError('The inputs are not strings')
        if sum([1 for x in s1 if (x != '0' and x != '1')]) > 0 or sum([1 for x in s2 if (x != '0' and x != '1')]) > 0:
            raise ValueError('Input strings contain characters other than 0s and 1s') 
        if len(s1) != len(s2):
            raise ValueError('Input strings are not of the same length') 
            
        x1 = np.array([int(i) for i in s1])
        x2 = np.array([int(i) for i in s2])
        x = np.sum(x1 * x2) %2
        return x


    def give_binary_string(self, z,n):
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

    def generate_random_s(self, n):
        """
        Generates a random bit string of length s
        Args:
            n: integer (length of bitstring)
        Returns:
            s: string of length n, consisting of 0s and 1s only.
        """
        if type(n)!= int or n < 1 :
            return("The input must be a positive integer")
        # select random number 
        z = rd.randint(0, 2**n - 1)
        # convert the number to binary representation
        s = self.give_binary_string(z,n)
        return s


    def s_function(self, s):
        """
        Args: 
            s: string of length n. Must contain only zeroes and ones.
        Output:
            f: list of length 2^n. f[i] contains the value of f corresponding to i. Each element of the list is a string
            f is such that, f(x) = f(y) iff x + y \in {0,s}
        """
        
        if type(s) != str:  #If inputs are not strings, raise an error
            raise ValueError('s should be a string')
        if sum([1 for x in s if (x != '0' and x != '1')]) > 0 or sum([1 for x in s if (x != '0' and x != '1')]) > 0:  
            #check if s contains only zeros and ones
            raise ValueError('s should only contain 0s and 1s') 
            
        n = len(s)
        N = 2**n
        f_temp = [-1 for i in range(N)]
        
        perm_random = np.random.permutation(N)
        if s == '0'*n:
            for i in range(N):
                f_temp = np.copy(perm_random)
        else:
            chosen_perm = perm_random[0:int(N/2)]
            ind_1 = 0
            for i in range(N):
                if f_temp[i] == -1:
                    f_temp[i] = chosen_perm[ind_1]
                    f_temp[int(self.binary_add(self.give_binary_string(i,n) , s), 2)] = chosen_perm[ind_1]
                    ind_1 += 1
        f = [self.give_binary_string(z.item(),n) for z in f_temp]
        return f


    
    def create_Uf(self,f):
        """
        Given a function f:{0,1}^n ---> {0,1}^n, creates and returns the corresponding oracle (unitary matrix) Uf
        Args:
            f: list of length n. f[i] is a string of length n.
        Returns:
            Uf: ndarray of size [2**(2n), 2**(2n)], representing a unitary matrix.
        """
        if type(f) != list:
            raise ValueError('Input the function in the form of a list')
        if any((x != '0' and x != '1') for x in "".join([y for y in f])):
            raise ValueError('The input function should only contain zeros and ones.')
        len_fs = [len(x) for x in f]
        if np.sum(np.diff(len_fs)) != 0:
            raise ValueError('Each element of f must be a string of the same length as the others.')
        n = len_fs[0]
        N = 2 ** (2*n)
        Uf = np.zeros([N,N], dtype = complex )
        
        for z in range(2**n):
            f_z = f[z]
            z_bin = self.give_binary_string(z,n)
            
            all_binary_strings = [self.give_binary_string(c,n) for c in range(2**n)]
            for j in all_binary_strings:
                inp = z_bin + j
                out = z_bin + self.binary_add(f_z,j)
                Uf[int(out[::-1],2),int(inp[::-1],2)] = 1
        return Uf

    def get_Simon_circuit(self,Uf):
        """
        Simon's algorithm: Determines the value of s, for a function f = a . x + b
        Args:
            Uf: numpy array. Must be of shape [N,N], with N being an even power of 2.
                
        Returns: 
            circ: Qiskit QuantumCircuit object.
        """
        if not isinstance(Uf, np.ndarray):
            raise ValueError("Uf must be a numpy array.")
        if Uf.ndim != 2:
            raise ValueError("Uf should be a two dimensional array.")
        if Uf.shape[0] != Uf.shape[1]:
            raise ValueError("Uf must be a square array.")
        if np.log2(Uf.shape[0])%2 != 0:
            raise ValueError("number of rows and columns of Uf must be an even power of 2")
        
        n = int(np.log2(Uf.shape[0]) / 2)
        ## Define the circuit
        circ = QuantumCircuit(2*n, n)    
        
        for i in range(n):
            circ.h(i)
        
        circ.iso(Uf, list(range(2*n)), [])
            
        for i in range(n):
            circ.h(i)
            
        circ = transpile(circ, backend = Aer.get_backend('qasm_simulator'))
        circ.measure(list(range(0,n))[::-1], list(range(n)))
        return circ

    def run_created_circuit(self,circ, num_shots = 20):
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circ, simulator, shots = num_shots)
        result = job.result()
        counts = result.get_counts(circ)
        
        return counts

    def s_solution(self,list_y):
        """
        Args:
            list_y: List of strings of length n
        Returns:
            list_s: List of solutions to y.s=0
        """
        
        n = len(list_y[0])
        list_s = []
        
        for z in range(2**n):
            potential_s = self.give_binary_string(z,n)
            if np.sum([self.dot_product(potential_s, y) for y in list_y]) == 0:
                list_s.append(potential_s)
                
        return list_s
        
    def simon_solution(self,f, num_shots = 20):
        """ 
        For a higher level implementation of Simon's algorithm. Takes an input f, and number of times the circuit is to be run.
        Returns the result of the search.
        Args:
            f: list of length n. f[i] is a string of length n.
            num_shots: Integer>0. Higher the value, larger the probability of success
        Returns:
            list_s: List of strings, with each string being of length n. These will contain the solution of the search problem, as 
                    determined by the Grover's algorithm
        """
        Uf = self.create_Uf(f)
        circ = self.get_Simon_circuit(Uf)
        counts = self.run_created_circuit(circ, num_shots = num_shots)
        list_y = [y for y in counts]
        list_s = self.s_solution(list_y)
        return list_s

    def verify_Simon_output(self,f, list_s):
        """Verifies if the output from the Simon's algorithm is correct
        Args:
            f: list of length n. f[i] is a string of length n.
            list_s: List of strings, with each string being of length n. These will contain the solution of the search problem, as 
                    determined by the Grover's algorithm. The first element of list_s will always contain '000..0'.
        Returns:
            is_correct: Bool. True if output is correct, else false.
        """
        ## Type checking
        if type(f) != list:
            raise ValueError('Input the function in the form of a list')
        if any((x != '0' and x != '1') for x in "".join([y for y in f])):
            raise ValueError('The input function should only contain zeros and ones.')
        len_fs = [len(x) for x in f]
        if np.sum(np.diff(len_fs)) != 0:
            raise ValueError('Each element of f must be a string of the same length as the others.')
            
        if type(list_s) != list:
            raise ValueError('list_s must be a list')
        
        ## return true only if list_s contains exactly two strings, and the non-zero string satisfies the condition f[0] = f[0+s]
        
        if len(list_s) == 2 and f[0] == f[0 + int(list_s[1],2)]:
            return True
        elif len(list_s) == 1:
            return True
        else:
            return False
        
    def scaling_analysis( self, n_min = 1, n_max = 4, time_out_val = 10000, num_times = 20, save_data = True): 
        """ Obtains the time scaling as n is increased
        Kwargs:
            n_min = (integer) lowest value of n to be considered
            n_max = (integer) highest value of n to be considered
            time_out_val = (integer) timeout value for running the QISKIT circuit
            num_times: (integer) number of times Simon's algorithm is to be executed for each n.
            save_data: (bool) If True, save the data in a .npz file
        Returns:
            n_list: (list) contains all integers from n_min to n_max
            time_reqd_arr: [ndarray) [i,j] entry contains the amount of time it took for ith value of n, and jth iteration
            correctness_arr: (ndarray) True or False entries depending on whether the algorithm succeeded or failed.
            avg_time: (ndarray) avg_time[i] contains the average amount of time taken for n_list[i]
        """
            
        n_list = list(range(n_min,n_max+1))
        time_reqd_arr = np.zeros([(n_max-n_min)+1, num_times])
        correctness_arr = np.ones([(n_max-n_min)+1, num_times], dtype = int)
        ci_list = ["failed", "succeeded"]
        for ind, n in enumerate(n_list):
            for iter_ind in range(num_times):
                s = self.generate_random_s(n)
                f = self.s_function(s)
                
                start = time.time()
                list_s = self.simon_solution(f, num_shots = 20)
                end = time.time()
                time_reqd_arr[ind, iter_ind] = end-start
                correctness_arr[ind, iter_ind] *= self.verify_Simon_output(f, list_s)
                print("n = %i and iteration = %i:It took %i seconds. Algorithm %s. Correct s = %s, obtained values are"
                      %(n,iter_ind,(end-start), ci_list[correctness_arr[ind, iter_ind]], s),list_s)
        avg_time = np.sum(time_reqd_arr,1)/np.shape(time_reqd_arr)[1]
        if save_data:
            np.savez('Simon_scaling.npz', n_list = n_list, num_times = num_times, time_reqd_arr = time_reqd_arr,
                     correctness_arr = correctness_arr, avg_time = avg_time)
        return n_list, time_reqd_arr, correctness_arr, avg_time
    
    def load_scaling_analysis(self): 
        """
        Load and return the the data for time scaling vs n.
        """
        data = np.load('Simon_scaling.npz')
        n_list = data['n_list']
        num_times = data['num_times']
        correctness_arr = data['correctness_arr']
        time_reqd_arr = data['time_reqd_arr']
        avg_time = data['avg_time']
        
        return n_list, num_times, correctness_arr, time_reqd_arr, avg_time
    
    def plot_and_save_scaling(self, n_list, avg_time, save_data = False):
        """
        Plot average run time vs n
        Args:
            n_list: List of n values
            avg_time: list of average time of execution corresponding to n's from n_list
            save_data: save the figure if this is set to True.
        """
        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))
        
        z = np.polyfit(n_list, avg_time, 10)
        p = np.poly1d(z)
        
        plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
        plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
        plt.title('Execution time scaling for Simon\'s algorithm', fontsize=25)
        plt.xlabel('n (bit string length)',fontsize=20)
        plt.ylabel('Average time of execution (s)',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if save_data:
            fig.savefig('Figures/Simon_scaling.png', bbox_inches='tight')
        
    def check_correctness(self, n= 3, num_times = 100):
        """
        Run Simon's algorithm for a given n, num_times number of times. For each run, f is chosen randomly.
        Args:
            n: Integer
            num_times: Integer
        Returns: 
            correctness_arr: Array of bools
        """
        correctness_arr = np.ones(num_times, dtype = int)
        ci_list = ["failed", "succeeded"]
        for iter_ind in range(num_times):
            s = self.generate_random_s(n)
            f = self.s_function(s)
            list_s = self.simon_solution(f, num_shots = 20)
            correctness_arr[iter_ind] *= self.verify_Simon_output(f, list_s)
            print("Iteration = %i: Algorithm %s. Correct s = %s, obtained values are"
                      %(iter_ind, ci_list[correctness_arr[iter_ind]], s),list_s)
            
        return correctness_arr
    
    
    def Uf_dependence_analysis(self, n = 3, num_times= 100, time_out_val = 10000, save_data = True):
        """
        Runs Simon's algorithms for num_times number of times for randomly chosen f's correspoding to given value of n
        kwargs:
            n: Integer
            num_times: Integer (number of random f's to be tested)
            time_out_val: integer
            save_data: bool
        returns:
            correctness_arr: array of bools
            time_required_arr: array of floats
        """
      
        correctness_arr = np.ones(num_times, dtype = int)
        time_reqd_arr = np.zeros(num_times)
        ci_list = ["failed", "succeeded"]
        for iter_ind in range(num_times):
            s = self.generate_random_s(n)
            f = self.s_function(s)
            
            start = time.time()
            list_s = self.simon_solution(f, num_shots = 20)
            end = time.time()
            time_reqd_arr[iter_ind] = end-start
            correctness_arr[iter_ind] *= self.verify_Simon_output(f, list_s)
            print("Iteration %i:It took %i seconds. Algorithm %s. Correct s = %s, obtained values are"
                  %(iter_ind, (end-start), ci_list[correctness_arr[iter_ind]], s),list_s)
        
        if save_data:
            np.savez('Simon_Uf_dependence.npz', num_times = num_times, n = n,
                     correctness_arr = correctness_arr, time_reqd_arr = time_reqd_arr)

            
        return correctness_arr, time_reqd_arr
    
    def load_Uf_analysis(self): 
        """
        Load data from file
        """
        data = np.load('Simon_Uf_dependence.npz')
        num_times = data['num_times']
        correctness_arr = data['correctness_arr']
        time_reqd_arr = data['time_reqd_arr']        
        n = data['n']
        
        return num_times, correctness_arr, n, time_reqd_arr
    
    def plot_and_save_UF_analysis(self, time_reqd_arr, save_data = False):
        """
        Plot histogram of run time, for randomly chosen fs
        Args:
            time_reqd_arr: array of floats
            save_data: save the figure if this is set to True.
        """
        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))
        
        
        plt.hist(time_reqd_arr)
        plt.title('Dependence of execution time on $U_f$ (Simon\'s algorithm)', fontsize=25)
        plt.xlabel('Execution time (s)',fontsize=20)
        plt.ylabel('Frequency of occurence',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        if save_data:
            fig.savefig('Figures/Simon_hist.png', bbox_inches='tight')

        

#%% Suggested code for implementation 
    
p = Program()
n_list, time_reqd_arr, correctness_arr, avg_time = p.scaling_analysis(n_min = 1, n_max = 3, time_out_val = 10000, num_times = 1, save_data = True)
n_list, num_times, correctness_arr, time_reqd_arr, avg_time = p.load_scaling_analysis()
p.plot_and_save_scaling(n_list, avg_time, save_data = False)
p.check_correctness()
correctness_arr, time_reqd_arr = p.Uf_dependence_analysis(save_data = True)
num_times, correctness_arr, n, time_reqd_arr = p.load_Uf_analysis() 
p.plot_and_save_UF_analysis(time_reqd_arr, save_data = False)
    