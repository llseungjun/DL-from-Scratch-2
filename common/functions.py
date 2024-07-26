import numpy as np
"""
    TODO  
    [x]sigmoid
    [x]relu
    [x]softmax
    [x]cross_entropy_error
    여기 있는 함수 들은 기본적으로 구현 가능해야함
"""
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x, axis = 1, keepdims=True)
        return exp_x / sum_exp_x
    elif x.ndim == 1:
        c = np.max(x)
        exp_x = np.exp(x-c)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x
    
def cross_entropy_error(y,t):
    if y.ndim == 1: # batch_size == 1
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)

    # t가 원 핫 벡터일 경우
    if t.size == y.size:
        t = t.argmax(axis = 1)
    
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size),t]+ 1e-7)) / batch_size
