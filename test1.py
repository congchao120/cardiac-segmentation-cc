class Num123(object):
    def __init__(self, i):
        self.i = i


input = [1,2,3,4]
nums = map(lambda x: Num123(x), input)
print(list(nums))

def f(x):
    return x*x
f_map = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
print (list(f_map))