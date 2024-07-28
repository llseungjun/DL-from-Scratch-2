import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.trainer import Trainer
from common.optimizer import Adam
import pickle
from cbow import CBOW
from common.util import create_contexts_target, to_cpu
from dataset import ptb

# 하이퍼파라미터 세팅
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# data import
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)

# init model & optimizer & trainer
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# train
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 필요한 데이터 pkl로 저장
word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(dtype=np.float16) # 임베딩 된 word 벡터
params['word_to_id'] = word_to_id 
params['id_to_word'] = id_to_word # 인덱스를 넣으면 해당 인덱스의 단어 str을 알 수 있는 딕셔너리 # 단어와 단어 ID 변환을 위해
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)

