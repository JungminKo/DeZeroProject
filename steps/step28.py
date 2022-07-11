if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

### 예제1 : 기울기벡터 구하기
# x0 = Variable(np.array(0.0))
# x1 = Variable(np.array(2.0))

# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)

### 예제2 : 경사하강법
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 # 학습률
iters = 1000 # 반복 횟수

for i in range(iters):
    print(x0, x1)
    
    y = rosenbrock(x0, x1)
    
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad