import sys, os
sys.path.append(os.pardir)
from common.layers import Embedding, SigmoidWithLoss
import numpy as np
import collections
"""
    TODO
    [ ]NegativeSamplingLoss 구현 다시 해보기(책 174p 참고, 굉장히 어려움)

"""
class EmbeddingDot:
    def __init__(self,W) -> None:
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
    
    def forward(self, h, idx:list):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W*h, axis = 1) # element-wise..? 왜 dot product가 아닐까..
        # 만약에 target이 단어 여러개가 아닌 '하나'라고 생각해보자
        # (1Xhidden_dim) dot (hidden_dimX1)을 수행해야 한다. (원래는) 
        # 하지만 위와 같이 구현했을 때 어짜피 dot product를 했을 때와 결과는 같다!(최종적으로 axis=1 방향으로 sum을 수행하고 난 후)

        self.cache = (h,target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache

        dout = dout.reshape(dout.shape[0], 1) #(batch_size X 1)
        dtarget_W = dout*h # target_W의 역전파니까 dout에 h를 곱함, 이때 dout이 h의 각 행 별로 broadcast 되도록 설계됨.
        self.embed.backward(dtarget_W) 
        dh = dout*target_W # h의 역전파니까 dout에 target_W를 곱함

        return dh
    
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        GPU = False
        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5) -> None:
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, h, target:list):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # positive sample forward
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=int) # 시그모이드의 입력 t, positive sample이므로 t가 1이 돼야함
        loss = self.loss_layers[0].forward(score, correct_label)

        # negative sample forward
        negative_label = np.zeros(batch_size,dtype=int)
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i] # 모든 행(단어)의 i번째 negative sample
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss = self.loss_layers[i+1].forward(score, negative_label)
        
        return loss
    
    def backward(self,dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers,self.embed_dot_layers): # positive, negative 상관없이 모든 layer grads통합
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        
        return dh