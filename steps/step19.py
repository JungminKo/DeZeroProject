import weakref 
import numpy as np

import contextlib 

class Config:
    enable_backdrop = True

class Variable:
    def __init__(self, data, name=None): # name 추가
        if data is not None: 
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.name = name # 수많은 변수를 처리하기 위해
        self.grad = None
        self.creator = None
        self.generation = 0 

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data) 
        
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation) 
  
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs] 
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None 

    def cleargrad(self):
        self.grad = None

    ##### 추가 ######
    @property # 메서드를 인스턴스 변수처럼 활용 가능
    def shape(self):
        return self.data.shape
    
    @property 
    def ndim(self):
        return self.data.ndim

    @property 
    def size(self):
        return self.data.size
    
    @property 
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self): # print 함수 호출시
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9) # 여러줄일 때는 공백문자로 시작위치 조정
        return 'variable(' + p + ')'

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
        
        if Config.enable_backdrop:
            self.generation = max([x.generation for x in inputs]) 

            for output in outputs:
                output.set_creator(self) 
            
            self.inputs = inputs 
            self.outputs = [weakref.ref(output) for output in outputs] 
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

def square(x):
    return Square()(x)

@contextlib.contextmanager
def using_config(name, value): 
    old_value = getattr(Config, name) 
    setattr(Config, name, value) 
    try:
        yield
    
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backdrop', False)

