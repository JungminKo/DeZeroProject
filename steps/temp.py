import numpy as np

from step06 import *

x = Variable(np.array(0.5))
y = square(exp(square(x)))

# 역전파
#y.grad = np.array(1.0)
y.backward()
print(x.grad )


#a = Variable(1.0)
a = Variable(np.array(1.0))