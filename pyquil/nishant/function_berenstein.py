def f2(*args):
    a = [1, 0, 1]
    b = 0
    result = b
    for index in range(0, len(a)):
        result ^= a[index] * args[index]
    return result
        
