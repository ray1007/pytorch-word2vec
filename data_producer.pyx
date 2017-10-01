import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cbow_producer(sent_id, int sent_id_len, int window, int negative, int vocab_size):
    cdef int i,j,n
    cdef int ctx_count
    cdef np.ndarray data = np.zeros([sent_id_len,2*window+1+negative], dtype=np.int64)
    cdef np.ndarray neg_indices = np.random.random_integers(0, vocab_size, (sent_id_len, negative))

    for i in range(sent_id_len):
        #ctx_indices = []
        ctx_count = 0
        for j in range(i-window, i+window+1):
            if j < 0 or j >= sent_id_len or j == i:
                data[i, ctx_count] = vocab_size
                continue
            #ctx_indices.append( sent_id[j] )
            data[i, ctx_count] = sent_id[j]
            ctx_count += 1

        #word_idx = sent_id[i]
        data[i, 2*window] = sent_id[i]

        # data = ([ctx_indices], word_idx)
        #data_queue.put((ctx_indices, word_idx, 1))
        #yield (ctx_indices, word_idx, 1)
        #yield word_idx

        # negative sampling
        for n in range(negative):
            #neg_idx = np.random.randint(vocab_size)
            data[i, 2*window+n] = neg_indices[i,n]
            #data_queue.put((ctx_indices, neg_idx, 0))
            #yield neg_idx
    return data

