#!/usr/bin/python

import unittest
import random

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

def dot(x, y):
    res = 0
    for index in range(0, len(x)):
        res += x[index]*y[index]
    return res % 2

def xor(x, y):
    return [ x[index] ^ y[index] for index in range(len(x)) ]
    
class Test(unittest.TestCase):
    def test_one(self):
        solver = Solver(7)
        arrays = [[1,0,1,0,1,1,0],
                  [0,0,1,0,0,0,1],
                  [1,1,0,0,1,0,1],
                  [0,0,1,1,0,1,1],
                  [0,1,0,1,0,0,1],
                  [0,1,1,0,1,1,1]]
        for array in arrays:
            solver.add_vector(array)
        self.assertEqual(solver.solve(), [1, 1, 0, 1, 0, 1, 0])
    def test_random(self):
        for n in range(1, 6):
            for trial in range(0, min(2**n, 10)):
                print("n = %d, trial = %d" % (n, trial))
                solver = Solver(n)
                if 2**n < 10:
                    s = [ int(char) for char in bin(trial)[2:].zfill(n) ]
                else:
                    s = [ random.randrange(0, 2) for _ in range(n) ]
                print("s = %s" % (s))
                f = lambda x: min(x, xor(x, s)) 
                while not solver.is_ready():
                    y = [ random.randrange(0, 2) for _ in range(n) ]
                    if dot(y, s) == 0:
                        solver.add_vector(y)
                possible_s = solver.solve()
                if f([0]*n) == f(possible_s):
                    final_s = possible_s
                else:
                    final_s = [0]*n
                print("solution = %s" % (final_s))      
                self.assertEqual(final_s, s)
                print()
                    
                
if __name__ == "__main__":
    unittest.main()


                

        
        
