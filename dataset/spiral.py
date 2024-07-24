# reference : https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/dataset/spiral.py
#             https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/ch01/show_spiral_dataset.py
import numpy as np
import matplotlib.pyplot as plt
def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 클래스당 샘플 수
    DIM = 2  # 데어터 요소 수
    CLS_NUM = 3  # 클래스 수

    x = np.zeros((N*CLS_NUM, DIM))
    # t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int) # numpy 1.23.0 이하로만 np.int를 지원함
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=int) # numpy 1.26.4에서는 np.int를 지원하지 않음

    for j in range(CLS_NUM):
        for i in range(N): # N*j, N*(j+1)):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t

if __name__ == "__main__":
    x, t = load_data()
    print('x shape: ', x.shape)  # (300, 2)
    print('t shape: ', t.shape)  # (300, 3)
    
    # 데이터점 플롯
    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
    plt.show()