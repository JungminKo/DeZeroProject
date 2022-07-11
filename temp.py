import numpy as np
#from dezero.core_simple import Variable
from dezero import Variable, utils

x = Variable(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print(x)
print(utils.sum_to(x, (3, 1)))