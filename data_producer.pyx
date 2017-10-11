import numpy as np
cimport numpy as np
import cython
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, fseek, ftell, SEEK_END, rewind, fread

#def cbow_producer(sent_id, int sent_id_len, int window, int negative, int vocab_size):
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cbow_producer(sent_id, int sent_id_len, int window, int negative, int vocab_size, int batch_size):
    cdef int i,j,n,q,r
    cdef int ctx_count
    #cdef np.ndarray data = np.zeros([sent_id_len,2*window+1+negative], dtype=np.int64)
    cdef int[:,:] data = np.zeros([sent_id_len,2*window+1+negative], dtype=np.int64)
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
    q = sent_id_len // batch_size
    r = sent_id_len % batch_size
    if q > 0:
        for i in range(q):
            yield data[i*batch_size:(i+1)*batch_size, :]
        if r > 0:
            yield data[sent_id_len-r:sent_id_len, :]
    else:
        yield data

def write_embs(str fn, word_list, float[:,:] embs, int vocab_size, int dim):
    cdef int i,j
    #fo = fopen(fn, "rb") 
    #out_f.write('</s>\t')
    with open(fn, 'w') as out_f:
        out_f.write('%d\t%d\n' % (vocab_size, dim));

        out_f.write('</s>\t')
        for j in range(dim):
            out_f.write( '%.18f\t' % embs[vocab_size, j] )
        out_f.write('\n')

        for i in range(vocab_size):
            out_f.write('%s\t' % word_list[i])
            for j in range(dim):
                out_f.write( '%.18f\t' % embs[vocab_size, j] )
            out_f.write('\n')
        

