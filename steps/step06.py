'''
역전파 추가 코드
'''

import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None: # data type : np.ndarray만 지원
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        '''
        ## 재귀를 이용한 방법
        ## 재귀를 이용하면, 중간 결과를 메모리에 유지하면서 처리를 이어가기 때문에 반복문 방식의 효율이 일반적으로 더 좋음

        f = self.creator # 1. 함수를 가져옴
        if f is not None:
            x = f.input # 2. 함수의 입력을 가져옴
            x.grad = f.backward(self.grad) # 3. 함수의 backward 메서드를 호출
            x.backward() # 하나 앞의 변수의 backward 메서드 호출(재귀)
        '''
        ## 반복문을 이용하는 방식

        if self.grad is None:
            self.grad = np.ones_like(self.data) # 처음시작할 때 grad 를 1로 시작안해도 역전파 시작 가능

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 함수를 가져오기
            x, y = f.input, f.output # 함수의 입력과 출력을 가져오기
            x.grad = f.backward(y.grad) # backward 메서드 호출

            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 list에 추가
        
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 구체적인 계산은 forward method에서 진행

        output = Variable(as_array(y))
        output.set_creator(self) # 출력 변수에 참조자를 설정
        
        self.input = input # 입력 변수를 보관
        self.output = output
        return output

    def forward(self, in_data): 
        raise NotImplementedError() # 상속해서 구현하지 않으면 error 발생

    def backward(self, gy):
        raise NotImplementedError()

##################################################################


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 클래스를 파이썬 함수로 사용 가능
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)