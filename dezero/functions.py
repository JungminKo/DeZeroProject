import numpy as np
from dezero.core import Function, Variable, as_variable, as_array
from dezero import utils

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs # x = self.inputs[0]
        gx = gy * cos(x) # Dezero의 cos함수
        return gx
        
class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
    
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]() # 약한 참조때문에 그 값에 접근하려고 () 추가
        gx = gy * (1 - y * y)
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]() ## 약한 참조때문에 그 값에 접근하려고 () 추가
        gx = gy * y
        return gx
    
class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx
    
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx
    
# # 축 고려하는 Transpose
# class Transpose(Function):
#     def __init__(self, axes=None):
#         self.axes = axes

#     def forward(self, x):
#         y = x.transpose(self.axes)
#         return y

#     def backward(self, gy):
#         if self.axes is None:
#             return transpose(gy)

#         axes_len = len(self.axes)
#         inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
#         return transpose(gy, inv_axes)

# def transpose(x, axes=None):
#     return Transpose(axes)(x)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        y = x[self.slices]
        return y
    
    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)
    
class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
        
    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        
    def forward(self, x):   
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    
    y = t + b
    t.data = None # t의 데이터 삭제 for memory efficiency
    return y

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb
        
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1
    
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y
    
class Sigmoid(Function):
    def forward(self, x):
        # y = 1 / (1 + exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5 # Better implementation
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis
        
    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y
    
def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    
    p = softmax(x) # softmax_simple(x)
    p = clip(p, 1e-15, 1.0) # log0 을 방지하기 위해 p의 값을 1e-15 이상으로 한다.
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y
    
    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape
        
        gy *= 1 / N
        y = softmax(x)
        
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y
        
class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        
    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()
        
        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond
    
class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y
    
class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
        
    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y
    
    def backward(self, gy):
        x,  = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx
    
def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

def exp(x):
    return Exp()(x)

def log(x):
    return Log()(x)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)    
    return Reshape(shape)(x)

def transpose(x):
    return Transpose()(x)

def get_item(x, slices):
    f = GetItem(slices)
    return f(x)

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

def matmul(x, W):
    return MatMul()(x, W)

def linear(x, W, b=None):
    return Linear()(x, W, b)

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def sigmoid(x):
    return Sigmoid()(x)

def relu(x):
    return ReLU()(x)

def softmax(x, axis=1):
    return Softmax(axis)(x)

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)

def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

def accuracy(y, t):
    # 내부 계산은 ndarray 인스턴스를 사용해서 수행하므로
    # 미분 불가능
    y, t = as_variable(y), as_variable(t)
    
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))