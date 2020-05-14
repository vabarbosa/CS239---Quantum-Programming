from grover import run_grover

def test(a,b):
	if(a == 0 and b == 1):
		return 1;
	else:
		return 0;
	
if __name__== '__main__':
	result = run_grover(test)
	print("in test wrapper")
	print(result)