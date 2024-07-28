import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import Embedding
from negative_sampling_layer import NegativeSamplingLoss
"""
    이해 안간다면 chapter3의 simple_cbow와 같이보면 이해가 그나마 된다..
"""
class CBOW:
    def __init__(self, vocab_size, hidden_Size, window_size, corpus) -> None:
        V, H = vocab_size, hidden_Size

        # 가중치 초기화
        W_in = 0.01*np.random.randn(V,H).astype('f')
        W_out = 0.01*np.random.randn(V,H).astype('f') # simple cbow랑 다르게 H,V가 아니라 V,H => 아마 embedding layer를 거치기 때문일 듯
                                                      # 놀랍게도 해당 내용이 책 178p WARNING 부분에 나온다. 
                                                      # 책을 여러번 다시 읽어봐야하는 이유.. 중요한 내용을 놓치지 말자..    

        # 계층 생성
        self.in_layers = []
        for i in range(2*window_size): # ex) window size가 1일 경우에 양 옆으로 하나씩 in_layer를 계산하기 때문
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)


        # 모든 layer 관리
        layers = self.in_layers + [self.ns_loss]
        # 모든 layer의 params와 grads 관리
        self.params, self.grads = [],[]
        for layer in layers:
            self.params += layers.params
            self.grads += layers.grads
        
        # CBOW 인스턴스 변수에 단어의 분산 표현(word2vec) 저장
        self.word_vecs = W_in

        def forward(self, contexts, target): # 여기서 받는 contexts와 target이 원핫벡터가 아니란다. (책 179 참고)
            h = 0
            for i, layer in enumerate(self.in_layers):
                h += layer.forward(contexts[:,i]) # 모든 행(배치)들의 i번째 context
            h *= 1 / len(self.in_layers)

            loss = self.ns_loss.forward(h,target)
            return loss

        def backward(self, dout=1):
            dout = self.ns_loss.backward(dout)
            dout *= 1 / len(self.in_layers)
            for layer in self.in_layers: # reverse일 필요가 없는게 구조를 보면 병렬적으로 구성되어 있다.
                layer.backward(dout)
            return None # 어차피 self.word_vecs(W_in, word2vec)을 구하는 게 목표기 때문에 backward값이 필요가 없다.
                 