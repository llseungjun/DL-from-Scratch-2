import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# 동시 발생 행렬 (책 88p)
def create_co_matrix(corpus,vocab_size,window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size),dtype=int) # np.int32는 현재 numpy 버전에서 지원 안함
    #그리고 파이썬 int는 어짜피 8byte라고 하는데 왜 이렇게 한거지..

    for idx, word_id in enumerate(corpus): # corpus에 있는 모든 word에 대해 탐색
        for i in range(1, window_size + 1): # corpus의 해당 단어를 기준으로 window_size만큼 양 옆의 단어들을 표기함
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0: # corpus idx 범위 안이라면
                left_word_id = corpus[left_idx]
                co_matrix[word_id,left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id,right_word_id] += 1

    return co_matrix

def cos_similarity(x,y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2)+eps) # x의 정규화
    ny = y / np.sqrt(np.sum(y**2)+eps) # y의 정규화
    return np.dot(nx,ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # query가 word_to_id에 존재하는지 체크
    if query not in word_to_id:
        print(f'{query}를 찾을 수 없습니다.')
        return
    
    print(f'\n[query]: {query}')
    # query를 vector로 변환
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
   
    # 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec,word_matrix[i])
    
    # 유사도 기준 내림차순 출력
    count = 0
    for i in (-1*similarity).argsort(): # argsort()로 간단하게 구현 가능
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]}: {similarity[i]}')

        count+=1
        if count >= top:
            return

def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
    return M