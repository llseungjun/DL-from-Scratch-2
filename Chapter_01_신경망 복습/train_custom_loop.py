import sys,os
sys.path.append(os.pardir)
from common.optimizer import SGD
from dataset.spiral import load_data
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
import numpy as np

# data import
x, t = load_data()

# hyperparameter setting
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# model, optimizer setting
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

data_size = len(x)
max_iters = data_size // batch_size # 1 epoch 마다 배치 뽑아서 훈련하는 횟수
loss_list = []
acc_list = []
# train
for epoch in range(max_epoch):
    # 매 epoch 마다 data shuffle
    idx = np.random.permutation(data_size)
    x = x[idx] # 랜덤한 idx 순서로 shuffle 됨
    t = t[idx]
    
    for i in range(max_iters):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        t_batch = t[i*batch_size:(i+1)*batch_size]

        # forward, backward 수행 후 grads 구하기
        loss = model.forward(x_batch,t_batch)
        model.backward()
        optimizer.update(model.params,model.grads)

        loss_list.append(loss)
        
        # batch_size * 10번에 1번씩 학습 경과 출력
        if (i+1) % 10 == 0:
            # 모델의 정확도 계산
            y = model.predict(x)
            if t.ndim != 1: # 정답레이블이 원 핫 인코딩일 경우 
                t = np.argmax(t, axis=1)
            if y.ndim != 1:
                y = np.argmax(y, axis=1)

            acc = np.sum(y==t) / y.shape[0]
            
            acc_list.append(acc)
            
            print('| epoch %d | iter %d / %d | acc %.2f | loss %.2f' %(epoch+1,i+1,max_iters, acc, loss))

# 학습 결과 플롯
plt.plot(np.arange(len(loss_list)//10), loss_list[::10], label='train')
plt.xlabel('반복 (x10)')
plt.ylabel('손실')
plt.show()

# 경계 영역 플롯
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 데이터점 플롯
x, t = load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()

