if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

# # 예제 1
# # sin 함수 고차 미분
# x = Variable(np.array(1.0))
# y = F.sin(x)
# y.backward(create_graph=True)

# for i in range(3):
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)
#     print(x.grad) # n차 미분

# 예제 2
x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    
# 그래프 그리기
labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, v, label=labels[i])
plt.legend(loc='lower right')
plt.show()