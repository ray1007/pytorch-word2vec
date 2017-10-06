import numpy as np
cimport numpy as np
import cython

#def cbow_producer(sent_id, int sent_id_len, int window, int negative, int vocab_size):
@cython.boundscheck(False)
@cython.wraparound(False)
def cbow_producer(sent_id, int sent_id_len, int window, int negative, int vocab_size, int batch_size):
    cdef int i,j,n,q,r
    cdef int ctx_count
    cdef np.ndarray data = np.zeros([sent_id_len,2*window+1+negative], dtype=np.int64)
    cdef np.ndarray neg_indices = np.random.random_integers(0, vocab_size, (sent_id_len, negative))

    for i in range(sent_id_len):
        ctx_count = 0
        for j in range(i-window, i+window+1):
            if j < 0 or j >= sent_id_len or j == i:
                data[i, ctx_count] = vocab_size
                continue
            data[i, ctx_count] = sent_id[j]
            ctx_count += 1

        data[i, 2*window] = sent_id[i]

        # negative sampling
        for n in range(negative):
            data[i, 2*window+n] = neg_indices[i,n]
    #return data

    # batch generator 
    q, r = divmod(sent_id_len, batch_size)
    #q = sent_id_len // batch_size
    if q > 0:
        for i in range(q):
            yield data[i*batch_size:(i+1)*batch_size, :]
        if r > 0:
            yield data[sent_id_len-r:sent_id_len, :]
    else:
        yield data

