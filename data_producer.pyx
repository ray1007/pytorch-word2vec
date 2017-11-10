import numpy as np
cimport numpy as np
import cython
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, fseek, ftell, SEEK_END, rewind, fread
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.math cimport pow
from libc.stdint cimport uintptr_t

cdef int* unigram_table

@cython.cdivision(True)
def init_unigram_table(word_list, freq, int train_words):
    global unigram_table
    cdef int table_size = int(1e8)
    cdef int a, idx, vocab_size 
    cdef double power = 0.75
    cdef double d1
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
            
    return <uintptr_t>unigram_table

#def test_ptr(uintptr_t ptr_val):
#    cdef int* unigram_table
#    unigram_table = <int*>ptr_val
#    return [ unigram_table[a] for a in range(int(1e8)) ]

#cdef uniform():
#    return <double> rand() / RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cbow_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random):
    cdef int i,j,t,n,q,r
    cdef int ctx_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL
    
    # columns of data and its length:  
    #   ctx_indices:  [2 * window]
    #   ctx_lens:     [1]
    #   word_idx:     [1]
    #   neg_indices:  [negative]
    #   neg_mask:     [negative]
    cdef long[:,:] data = np.zeros([sent_id_len,2*window+1+1+2*negative], dtype=np.int64)

    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    for i in range(sent_id_len):
        ctx_count = 0
        actual_window = rand() % window + 1
        for j in range(i-actual_window, i+actual_window+1):
            if j < 0 or j >= sent_id_len or j == i:
                continue
            else:
                data[i, ctx_count] = sent_id[j]
                ctx_count += 1
        for j in range(ctx_count, 2*window+1):
            data[i, j] = vocab_size

        data[i, 2*window] = ctx_count
        data[i, 2*window+1] = sent_id[i]

        # negative sampling
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sg_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random):
    cdef int i,j,t,n,q,r
    cdef int batch_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL

    # columns of data and its length:  
    #   word_idx:     [1]
    #   ctx_idx:      [1]
    #   neg_indices:  [negative]
    #   neg_mask:     [negative]
    #cdef long[:,:] data = np.zeros([batch_size, 2+2*negative], dtype=np.int64)
    cdef long[:,:] data = np.zeros([sent_id_len*2*window, 2+2*negative], dtype=np.int64)

    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    batch_count = 0
    for i in range(sent_id_len):
        actual_window = rand() % window + 1
        for j in range(i-actual_window, i+actual_window+1):
            if j < 0 or j >= sent_id_len or j == i:
                continue
            
            data[batch_count, 0] = sent_id[j]
            data[batch_count, 1] = sent_id[i]
            
            # negative sampling
            neg_count = 0
            for n in range(negative):
                t= unigram_table[ (next_random >> 16) % int(1e8) ]
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
                if t == sent_id[i]:
                    continue

                data[batch_count, 2+neg_count] = t
                neg_count += 1

            # neg mask
            for n in range(neg_count):
                data[batch_count, 2+negative+n] = 1

            batch_count += 1 

            # if batch_count reaches batch_size 
            #if batch_count >= batch_size:
            #    yield np.asarray(data)
            #    batch_count = 0 

    q = batch_count // batch_size
    r = batch_count % batch_size
    if q > 0:
        for i in range(q):
            yield np.asarray(data[i*batch_size:(i+1)*batch_size, :])
        if r > 0:
            yield np.asarray(data[batch_count-r:batch_count, :])
    else:
        #yield np.asarray(data)
        yield np.asarray(data[:batch_count, :])
    # the remaining data 
    #if batch_count > 0:
    #    yield np.asarray(data[:batch_count, :])


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
        

