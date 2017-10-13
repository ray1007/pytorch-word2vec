import numpy as np
cimport numpy as np
import cython
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, fseek, ftell, SEEK_END, rewind, fread
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.math cimport pow
from libc.stdint cimport uintptr_t

#cdef int table_size = int(1e8)
#cdef unsigned long long* next_randoms 
cdef int* unigram_table

@cython.cdivision(True)
def init_unigram_table(word_list, freq, int train_words):
    global unigram_table
    #cdef int* unigram_table
    cdef int table_size = int(1e8)
    cdef int a, idx, vocab_size 
    cdef double power = 0.75
    cdef double d1
    cdef double train_words_pow = pow(train_words, power)
    unigram_table = <int *>malloc(table_size * sizeof(int));
    idx = 0
    vocab_size = len(word_list)

    d1 = pow(freq[ word_list[idx] ], power) / train_words_pow;
    for a in range(table_size):
        unigram_table[a] = idx
        if (<double>a / table_size) > d1:
            idx += 1
            d1 += pow(freq[ word_list[idx] ], power) / train_words_pow;
        if idx >= vocab_size:
            idx = vocab_size - 1;
    #return [ unigram_table[a] for a in range(table_size) ]
    return <uintptr_t>unigram_table

def test_ptr(uintptr_t ptr_val):
    #global unigram_table
    cdef int* unigram_table
    unigram_table = <int*>ptr_val
    return [ unigram_table[a] for a in range(int(1e8)) ]
    #pass

#cdef uniform():
#    return <double> rand() / RAND_MAX

#def cbow_producer(sent_id, int sent_id_len, int window, int negative, int vocab_size):
#def cbow_producer(sent_id, int sent_id_len, neg_sample_table, int window, int negative, int vocab_size, int batch_size):
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cbow_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size):
    cdef int i,j,t,n,q,r
    cdef int ctx_count
    #cdef np.ndarray data = np.zeros([sent_id_len,2*window+1+negative], dtype=np.int64)
    cdef long[:,:] data = np.zeros([sent_id_len,2*window+1+negative], dtype=np.int64)
    #cdef np.ndarray neg_indices = np.random.random_integers(0, vocab_size, (sent_id_len, negative))
    #cdef long[:,:] neg_indices = np.random.random_integers(0, vocab_size, (sent_id_len, negative))
    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    for i in range(sent_id_len):
        ctx_count = 0
        for j in range(i-window, i+window+1):
            if j < 0 or j >= sent_id_len or j == i:
                data[i, ctx_count] = vocab_size
                #continue
            else:
                data[i, ctx_count] = sent_id[j]
            ctx_count += 1

        data[i, 2*window] = sent_id[i]

        # negative sampling
        n = 0
        while n < negative:
            t = unigram_table[ <int>(<double> rand() / RAND_MAX * 1e8) ]
            if t == sent_id[i]:
                continue

            data[i, 2*window+n] = t
            n += 1
        #print(unigram_table[11694980])
        #for n in range(negative):    
            #data[i, 2*window+n] = 0
        #    data[i, 2*window+n] = unigram_table[ <int>(<double> rand() / RAND_MAX * 1e8) ]
        #print("%")
    #return data

    # batch generator 
    q = sent_id_len // batch_size
    r = sent_id_len % batch_size
    if q > 0:
        for i in range(q):
            yield np.asarray(data[i*batch_size:(i+1)*batch_size, :])
        if r > 0:
            yield np.asarray(data[sent_id_len-r:sent_id_len, :])
    else:
        yield np.asarray(data)

def write_embs(str fn, word_list, float[:,:] embs, int vocab_size, int dim):
    cdef int i,j
    #fo = fopen(fn, "rb") 
    #out_f.write('</s>\t')
    with open(fn, 'w') as out_f:
        out_f.write('%d\t%d\n' % (vocab_size+1, dim));

        out_f.write('</s>\t')
        for j in range(dim):
            out_f.write( '%.18f\t' % embs[vocab_size, j] )
        out_f.write('\n')

        for i in range(vocab_size):
            out_f.write('%s\t' % word_list[i])
            for j in range(dim):
                out_f.write( '%.18f\t' % embs[i, j] )
            out_f.write('\n')
        

