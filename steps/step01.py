'''
numerical differentiation (수치미분) 에 대한 코드
- 계산에 오차가 포함되어 있으며, 계산량이 많기 때문에 이후에는 역전파 사용
- BUT 역전파의 경우, 복잡한 알고리즘이라서 구현하면서 버그가 섞여 들어가기 쉬움 
-> 수치미분 결과와 비교하여 오류있는지 확인 필요 : 기울기 확인(gradient checking) 방법
'''

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 구체적인 계산은 forward method에서 진행
        output = Variable(y)

        return output

    def forward(self, in_data): 
        raise NotImplementedError() # 상속해서 구현하지 않으면 error 발생

    
class Square(Function):
    def forward(self, x):
        return x**2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def numerical_diff(f, x, eps=1e-4): # eps : epsilon
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)

    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)

