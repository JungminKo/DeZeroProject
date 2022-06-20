from tkinter import Y
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None: 
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) 

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs] 
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None: # 동일한 변수일 때 문제 발생하는 것을 방지하기 위해서 다음과 같이 설정
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)

            '''
            # 위와 동일한 코드
            for i, x in enumerate(f.inputs):
                 if x.grad is None: # 동일한 변수일 때 문제 발생하는 것을 방지하기 위해서 다음과 같이 설정
                    x.grad = gxs[i]
                else:
                    x.grad = x.grad + gxs[i]

                if x.creator is not None:
                    funcs.append(x.creator)
            '''
    def cleargrad(self):
        self.grad = None
        
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # 가변길이 수정
        xs = [x.data for x in inputs] 
        ys = self.forward(*xs) # 리스트 언팩
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 튜플로 만들어주기
            ys = (ys, ) 
            
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self) 
        
        self.inputs = inputs 
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, in_data): 
        raise NotImplementedError() 

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.inputs[0].data # 이전에는 x = self.input.data
        gx = 2 * x * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)