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
        self.generation = 0 # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 세대를 기록. 부모세대 + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) 
        
        ### 변경된 부분
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation) # 리스트 정렬, generation이 가장 큰 값이 맨 뒤로

        ########

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs] 
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator) # 변경 ; 이전 : funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs] 
        ys = self.forward(*xs)
        if not isinstance(ys, tuple): 
            ys = (ys, ) 
            
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs]) ## 추가

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
        x = self.inputs[0].data 
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