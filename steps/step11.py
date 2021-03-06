import numpy as np
import unittest

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
    def __call__(self, inputs):
        xs = [x.data for x in inputs] 
        ys = self.forward(xs)
        
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self) # 출력 변수에 참조자를 설정
        
        self.inputs = inputs # 입력 변수를 보관
        self.outputs = outputs
        return output

    def forward(self, in_data): 
        raise NotImplementedError() # 상속해서 구현하지 않으면 error 발생

    def backward(self, gy):
        raise NotImplementedError()
    
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

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)



class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y, )