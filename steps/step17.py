import weakref # 추가
import numpy as np

import contextlib # 추가

# 역전파 활성모드 / 역전파 비활성 모드 전환하는 Config 클래스
class Config:
    enable_backdrop = True

class Variable:
    def __init__(self, data):
        if data is not None: 
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad = False): # retain_grad 추가 : 중간 변수의 미분값 제거
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
            #gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs] # 약한 참조때문에 그 값에 접근하려고 () 추가
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
                    y().grad = None # y는 약한 참조(weakref)


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
        
        if Config.enable_backdrop:
            self.generation = max([x.generation for x in inputs]) 

            for output in outputs:
                output.set_creator(self) 
            
            self.inputs = inputs 
            #self.outputs = outputs
            self.outputs = [weakref.ref(output) for output in outputs] # 약한 참조로 변경 -> 순환 참조 없앰
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

# with 문을 활용하여 역전파 비활성 모드 전환
@contextlib.contextmanager
def using_config(name, value): # name : str, Config 클래스 속성 이름을 넣어줌
    old_value = getattr(Config, name) 
    setattr(Config, name, value) #  Config 클래스 속성 이름을 value 새로운 값 설정
    try:
        yield
    
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backdrop', False)

# Example 1
# for i in range(10):
#     x = Variable(np.random.randn(10000)) # 거대한 데이터
#     y = square(square(square(x))) # 복잡한 계산을 수행한다.
#     print(y.grad)

# Example 2
# x0 = Variable(np.array(1.0))
# x1 = Variable(np.array(1.0))
# t = add(x0, x1)
# y = add(x0, t)
# y.backward()

# print(y.grad, t.grad)
# print(x0.grad, x1.grad)


# Example 3
# Config.enable_backdrop = True
# x = Variable(np.ones((100, 100, 100)))
# y = square(square(square(x)))
# y.backward()

# Config.enable_backdrop = False
# x = Variable(np.ones((100, 100, 100)))
# y = square(square(square(x)))


# Example 4
# 순전파 코드만 실행 1
# with using_config('enable_backdrop', False):
#     x = Variable(np.array(2.0))
#     y = square(x)

# 순전파 코드만 실행 2
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)