import sys, os
sys.path.append(os.pardir)
from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus,vocab_size)

c0 = C[word_to_id['you']] # you의 벡터화 된 표현
c1 = C[word_to_id['i']] # i의 벡터화 된 표현
print(cos_similarity(c0,c1))
# 0.7071067758832467
