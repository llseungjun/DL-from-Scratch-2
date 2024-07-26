# reference : https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/common/optimizer.py
import numpy as np
"""
    TODO
    [x]SGD 
    [x]Momentum 
    [x]Adam 
    [ ]Nesterov 
    [ ]AdaGrad 
    [ ]RMSprop
    여기 있는 class 들은 기본적으로 구현 가능해야함
"""
class SGD:
    def __init__(self,lr=0.01) -> None:
        self.lr = lr

    def update(self, params: list, grads: list):
        for i,param in enumerate(params):
            params[i] -= self.lr * grads[i]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads) -> None:
        if self.v == None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))
        
        for i in range(len(params)):
            self.v[i] = self.momentum*self.v[i] - self.lr*grads
            params[i] += self.v[i]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    
    def update(self, params, grads)->None:
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1-self.beta1)*(grads[i] - self.m[i])
            self.v[i] += (1-self.beta2)*(grads[i]**2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
