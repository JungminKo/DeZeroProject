if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


# x = Variable(np.array(2.0))
# y = f(x)
# y.backward(create_graph=True)
# print(x.grad)

# # 두 번째 역전파 진행
# # # 이렇게 하면 오류 발생
# # gx = x.grad
# # gx.backward()
# # print(x.grad)

# gx = x.grad
# x.cleargrad() # 미분값 재설정
# gx.backward()
# print(x.grad)

#####################################
# 뉴턴 방법을 활용한 최적화
x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)
    
    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    
    x.data -= gx.data / gx2.data