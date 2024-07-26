import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        # 가중치 초기화
        W1 = 0.01*np.random.randn(input_size,hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = 0.01*np.random.randn(hidden_size,output_size)
        b2 = np.zeros(output_size)

        # layer 생성
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]
        
        self.loss_layer = SoftmaxWithLoss()

        # TwoLayerNet의 params와 grads관리
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        
        return x
    def forward(self,x,t):
        y = self.predict(x)
        loss = self.loss_layer.forward(y,t)

        return loss
    
    def backward(self, dout =1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout