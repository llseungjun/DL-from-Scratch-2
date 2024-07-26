import numpy as np
from common.functions import softmax, sigmoid, cross_entropy_error
"""
    TODO
    [x]Sigmoid
    [x]Relu
    [x]MatMul 
    [x]Affine 
    [x]Softmax 
    [x]SoftmaxWithLoss 
    [ ]SigmoidWithLoss 
    [ ]Dropout 
    [ ]Embedding
    [ ]SoftmaxWithLoss backpward 함수 직접 구현
    여기 있는 class 들은 기본적으로 구현 가능해야함
"""

class Sigmoid:
    def __init__(self) -> None:
        self.params, self.grads = [],[] # 책에서 이야기하는 '구현 규칙'을 지키기 위한 코드
        self.out = None

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Relu:
    def __init__(self) -> None:
        self.params, self.grads = [],[]
    
    def forward(self,x):
        return np.maximum(x, 0)
    
    def backward(self,dout):
        """
            dout[dout>=0] = 1 
            dx = dout
            return dx
            설명:
                위와 같은 구현이 안되는 이유
                    - dx는 0또는 1로 구성돼야 함
                    - 위와 같은 구현이면 1 또는 0보다 작은 수들로 구성되기 때문에 불가능
        """
        dx = np.zeros(dout)
        dx[dout>=0] = 1
        return dx

class MatMul:
    def __init__(self, W) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        W, = self.params
        out = np.matmul(x,W)
        self.x = x        
        return out
    
    def backward(self,dout):
        W, = self.params
        dW = np.matmul(self.x.T, dout)
        dx = np.matmul(dout, W.T)
        self.grads[0][...] = dW
        return dx

class Affine:
    def __init__(self, W, b) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x = x
        out = np.matmul(x, W) + b
        return out

    def backward(self, dout):
        W, b = self.params
        dW = np.matmul(self.x.T,dout)
        dx = np.matmul(dout,W.T)
        db = np.sum(dout , axis = 0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class Softmax:
    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.out = None
    
    def forward(self, x):
        out = softmax(x)
        self.out = out
        return out
    
    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out*sumdx    
        return dx

class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.params, self.grads = [],[]
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        
        # t가 원 핫 벡터일 경우 정답 인덱스로 변환
        if self.t.size == self.y.size: # ndarray size -> https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html
            self.t = np.argmax(self.t, axis = 1)
        
        loss = cross_entropy_error(self.y,self.t)
        return loss

    def backward(self,dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size),self.t] -= 1
        dx *= dout
        dx =  dx / batch_size

        return dx