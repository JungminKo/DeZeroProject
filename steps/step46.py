if '__file__' in globals():
    import os, sys  
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, optimizers
import dezero.functions as F
from dezero.models import MLP

# 데이터셋 생성
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 하이퍼파라미터 설정
lr = 0.2
max_iter = 10000
hidden_size = 10
    
model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)
# 또는 다음처럼 한줄에도 작성 가능
# optimizer = optimizers.SGD(lr).setup(model)

# 학습 시작
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    
    model.cleargrads()
    loss.backward()
    
    optimizer.update()
        
    if i % 1000 == 0:
        print(loss)