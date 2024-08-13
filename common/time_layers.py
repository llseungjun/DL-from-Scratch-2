import numpy as np
from common.layers import Embedding
from common.functions import softmax, sigmoid

class RNN:
    def __init__(self, Wx, Wh, b) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x,Wx) + b 
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1-h_next**2)
        db = np.sum(dt, axis = 0)
        dWh = np.matmul(h_prev.T,dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T,dt)
        dx = np.matmul(dt,Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False) -> None:
         self.params = [Wx, Wh, b]
         self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
         self.layers = None

         self.h, self.dh = None, None
         self.stateful = stateful

    def set_state(self,h):
        self.h = h
    
    def reset_state(self):
        self.h = None
    
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape # N개의 미니배치, T개의 시점, D개의 차원 수
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N,T,H), dtype='f')

        if not self.stateful or self.h is None: # 은닉 상태를 유지하지 않거나, 처음 학습을 진행할 경우 self.h 초기화
            self.h = np.zeros((N,H),dtype='f')

        for t in range(T):
            layer = RNN(*self.params) # arguments unpacking => https://choi-records.tistory.com/entry/Python-Phython-Asterisk-%EC%82%AC%EC%9A%A9%EB%B2%95
            self.h = layer.forward(xs[:,t,:], self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)
        
        return hs
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:,t,:] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1) # 각 데이터의 모든 시점 하나로 통합
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelㅇㅔ 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx

class LSTM:
    def __init__(self, Wx, Wh, b) -> None:
        self.params = [Wx, Wh, b] # Wx, Wh, b 각각은 이미 gate 4개로 concat된 parameter
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape # batch_size, hidden_size

        A = np.dot(x,Wx) + np.dot(h_prev,Wh) + b

        # A를 slice해서 forwarding
        # 열 방향으로 concat이기 때문에 아래와 같이 구현
        f = A[:,:H]
        g = A[:,H:2*H]
        i = A[:,2*H:3*H]
        o = A[:,3*H:]

        # activation
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f*c_prev + g*i
        h_next = o*np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)

        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 -tanh_c_next **2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = ds * tanh_c_next
        dg = ds * i

        # sigmoid, tanh 역전파
        di *= i * (1 - i)
        df *= f * (1 - f)
        do = dh_next * tanh_c_next
        dg *= (1 - g**2)

        dA = np.hstack((df,dg,di,do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis = 0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev
    
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache = None
        self.stateful = stateful
        self.layers = None
        self.h, self.c = None, None
        self.dh = None