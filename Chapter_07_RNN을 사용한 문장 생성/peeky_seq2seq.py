import sys
sys.path.append('..')
import numpy as np
from common.time_layers import TimeLSTM, TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss
from common.base_model import BaseModel
from seq2seq import Encoder, Seq2seq

# 인코더 구조는 기존 seq2seq 모델과 동일하나, 디코더 구조가 다르다.
class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size) -> None:
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D)/100).astype('f')
        lstm_Wx = (rn(H+D,4*H)/np.sqrt(H+D)).astype('f')
        lstm_Wh = (rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b =  np.zeros(4*H).astype('f')
        affine_W = (rn(H+H,V)/np.sqrt(H+H)).astype('f')
        affine_b =  np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        self.affine = TimeAffine(affine_W,affine_b)

        self.params = self.embed.params + self.lstm.params + self.affine.params
        self.grads = self.embed.grads + self.lstm.grads + self.affine.grads

        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        _, H = h.shape
        out = self.embed.forward(xs)
        hs = np.repeat(h,T,axis=0).reshape(N,T,H)
        out = np.concatenate((hs, out) , axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs,out), axis=2)
        
        score = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh
    
    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        while len(sampled) < sample_size:
            x = np.array(sample_id).reshape(1,1)
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled

class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V,D,H)
        self.decoder = PeekyDecoder(V,D,H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
    