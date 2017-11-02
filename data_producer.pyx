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
    #cdef double train_words_pow = pow(train_words, power)
    cdef double train_words_pow = 0.0
    unigram_table = <int *>malloc(table_size * sizeof(int));
    idx = 0
    vocab_size = len(word_list)

    for word in freq:
        train_words_pow += pow(freq[word], power)

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
#def cbow_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size):
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cbow_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random):
    cdef int i,j,t,n,q,r
    cdef int ctx_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL
    #cdef np.ndarray data = np.zeros([sent_id_len,2*window+1+negative], dtype=np.int64)
    cdef long[:,:] data = np.zeros([sent_id_len,2*window+1+1+2*negative], dtype=np.int64)

    # [0:2*window]ctx_word_id  [1]ctx_len [1]tar_word_id, [negative]neg_word_id
    #cdef np.ndarray neg_indices = np.random.random_integers(0, vocab_size, (sent_id_len, negative))
    #cdef long[:,:] neg_indices = np.random.random_integers(0, vocab_size, (sent_id_len, negative))
    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    for i in range(sent_id_len):
        ctx_count = 0
        #ctx_len = 0
        actual_window = rand() % window + 1
        #print(actual_window)
        for j in range(i-actual_window, i+actual_window+1):
            if j < 0 or j >= sent_id_len or j == i:
                #data[i, ctx_count] = vocab_size
                continue
            else:
                data[i, ctx_count] = sent_id[j]
                #ctx_len += 1
                ctx_count += 1
        for j in range(ctx_count, 2*window+1):
            data[i, j] = vocab_size

        data[i, 2*window] = ctx_count
        data[i, 2*window+1] = sent_id[i]

        # negative sampling
        #n = 0
        #while n < negative:
        neg_count = 0
        for n in range(negative):
            t= unigram_table[ (next_random >> 16) % int(1e8) ]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #t = rand() % vocab_size
            #t = unigram_table[ <int>(<double>(rand()>>16) / RAND_MAX * 1e8) ]
            #t = unigram_table[ <int>(<double> rand() / RAND_MAX * 1e8) ]
            #t = unigram_table[ rand() % 1e8 ]
            #t = rand() % 1e8 ]
            if t == sent_id[i]:
                continue

            data[i, 2*window+2+neg_count] = t
            neg_count += 1

        # neg mask
        for n in range(neg_count):
            data[i, 2*window+2+negative+n] = 1

        #print(unigram_table[11694980])
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
        #out_f.write('%d\t%d\n' % (vocab_size, dim));

        out_f.write('</s>\t')
        for j in range(dim):
            #out_f.write( '%.18f\t' % embs[vocab_size, j] )
            out_f.write( '%.6f\t' % embs[vocab_size, j] )
        out_f.write('\n')

        for i in range(vocab_size):
            out_f.write('%s\t' % word_list[i])
            for j in range(dim):
                #out_f.write( '%.18f\t' % embs[i, j] )
                out_f.write( '%.6f\t' % embs[i, j] )
            out_f.write('\n')
        

