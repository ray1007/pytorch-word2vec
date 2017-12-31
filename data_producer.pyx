import numpy as np
cimport numpy as np
import cython
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, fseek, ftell, SEEK_END, rewind, fread
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.math cimport pow, sqrt
from libc.stdint cimport uintptr_t

try:
    from scipy.linalg.blas import fblas
except ImportError:
    import scipy.linalg.blas as fblas

cdef extern from "voidptr.h":      
    void* PyCObject_AsVoidPtr(object obj)

ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil  
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer) # y += alpha * x

def dot(int size, float[:] x, float[:] y):
    cdef int ONE = 1
    v1 = sdot(&size, &x[0], &ONE, &y[0], &ONE)
    return v1

#cdef cos_sim(int size, float[:] x, float[:] y):
#    cdef int ONE = 1
#    return sdot(&size, &x[0], &ONE, &y[0], &ONE) / snrm2(&size, &x[0], &ONE) / snrm2(&size, &y[0], &ONE)

cdef float cosine(const int size, const float* x, const float* y):
    cdef int d
    cdef float sim, n1, n2
    sim = 0.0
    n1 = 0.0
    n2 = 0.0
    for d in range(size):
        sim += x[d] * y[d]
        n1 += x[d] * x[d]
        n2 += y[d] * y[d]
    return sim / sqrt(n1) / sqrt(n2)

cdef float cos_sim(const int size, const float* x, const float* y):
    cdef int ONE = 1
    return <float> sdot(&size, x, &ONE, y, &ONE) / snrm2(&size, x, &ONE) / snrm2(&size, y, &ONE)

cdef my_saxpy(const int size, const float a, const float* x, float* y):
    cdef int ONE = 1
    saxpy(&size, &a, x, &ONE, y, &ONE)

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

cdef int get_unigram_table_at_idx(int* arr, unsigned long long next_random):
    #return *(arr + ((next_random >> 16) % 100000000))
    return arr [ (next_random >> 16) % 100000000 ]
#def test_ptr(uintptr_t ptr_val):
#    cdef int* unigram_table
#    unigram_table = <int*>ptr_val
#    return [ unigram_table[a] for a in range(int(1e8)) ]

#cdef uniform():
#    return <double> rand() / RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cbow_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random, bint dynamic_window=True):
    cdef int i,j,tar_id,pos,t,n,q,r
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
        actual_window = rand() % window + 1 if dynamic_window else window
        tar_id = sent_id[i]
        for j in range(i-window, i-actual_window):
            pos = j - (i-window)
            data[i, pos] = vocab_size
        for j in range(i-actual_window, i):
            pos = j - (i-window)
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
            else:
                data[i, pos] = sent_id[j]
                ctx_count += 1
        for j in range(i+1, i+1+actual_window):
            pos = j - (i-window) - 1
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
            else:
                data[i, pos] = sent_id[j]
                ctx_count += 1
        for j in range(i+1+actual_window, i+1+window):
            pos = j - (i-window) - 1
            data[i, pos] = vocab_size

        data[i, 2*window] = ctx_count
        data[i, 2*window+1] = tar_id

        # negative sampling
        neg_count = 0
        for n in range(negative):
            #t= unigram_table[ (next_random >> 16) % int(1e8) ]
            t = get_unigram_table_at_idx(unigram_table, next_random)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #t = rand() % vocab_size
            #t = unigram_table[ <int>(<double>(rand()>>16) / RAND_MAX * 1e8) ]
            #t = unigram_table[ <int>(<double> rand() / RAND_MAX * 1e8) ]
            #t = unigram_table[ rand() % 1e8 ]
            #t = rand() % 1e8 ]
            if t == tar_id:
                continue

            data[i, 2*window+2+neg_count] = t
            neg_count += 1

        # neg mask
        for n in range(neg_count):
            data[i, 2*window+2+negative+n] = 1

    return np.asarray(data)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sg_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random, bint dynamic_window=True):
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
        actual_window = rand() % window + 1 if dynamic_window else window
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
    
    return np.asarray(data)

def write_embs(str fn, word_list, float[:,:] embs, int vocab_size, int dim):
    cdef int i,j
    with open(fn, 'w') as out_f:
        out_f.write('%d %d\n' % (vocab_size+1, dim));

        out_f.write('</s> ')
        for j in range(dim):
            out_f.write( '%.6f ' % embs[vocab_size, j] )
        out_f.write('\n')

        for i in range(vocab_size):
            out_f.write('%s ' % word_list[i])
            for j in range(dim):
                out_f.write( '%.6f ' % embs[i, j] )
            out_f.write('\n')
        
def create_n_update_sense(long[:] type_ids, float[:,:] context_feats, float[:,:] sense_embs, word2sense, float[:] counter_list, int type_ids_len, int emb_dim, float delta, int current_n_sense):
    cdef int b, d, pos, t_id, s_id
    cdef int max_sense_id, new_sense_id, create_count
    cdef float sim, max_sim

    create_count = 0
    for b in range(type_ids_len):
        t_id = type_ids[b]
        max_sense_id = -1
        max_sim = delta 
        for s_id in word2sense[t_id]:
            if counter_list[s_id] == 0.0:
                max_sense_id = s_id
                break
            sim = cosine(emb_dim, &context_feats[b,0], &sense_embs[s_id,0])
            if sim > max_sim:
                max_sim = sim
                max_sense_id = s_id
        
        if counter_list[max_sense_id] != 0.0 and len(word2sense[t_id]) < 5:
            if max_sim < delta:
                max_sense_id = -1

        if max_sense_id == -1:
            new_sense_id = current_n_sense + create_count
            word2sense[t_id].append(new_sense_id)
            sense_embs[new_sense_id, :] = context_feats[b, :]
            create_count += 1

            counter_list[ new_sense_id ] = 1.0
        else:
            for d in range(emb_dim):
                sense_embs[max_sense_id,d] += context_feats[b,d]

            counter_list[ max_sense_id ] += 1.0

    return create_count

def select_sense(long[:,:] chunk, float[:,:] context_feats, float[:,:] cluster_embs, word2sense, int chunk_size, int emb_dim, int window, int negative):
    cdef int b, d, pos, t_id, s_id
    cdef int max_sense_id
    cdef float sim, max_sim
    #cdef long[:,:] data = np.zeros([chunk_size, 2*window+2+2*negative], dtype=np.int64)

    for b in range(chunk_size):
        pos = 2*window+1
        t_id = chunk[b, pos]
        max_sense_id = -1
        max_sim = -10.0
        for s_id in word2sense[t_id]:
            sim = cosine(emb_dim, &context_feats[b,0], &cluster_embs[s_id,0])
            if sim > max_sim:
                max_sim = sim
                max_sense_id = s_id
        chunk[b, pos] = max_sense_id
        
        for pos in range(2*window+2, 2*window+2+negative):
            t_id = chunk[b, pos]
            chunk[b, pos] = word2sense[t_id][ rand() % len(word2sense[t_id]) ]

    return chunk

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def npmssg_producer(sent_id, int sent_id_len, uintptr_t ptr_val, int window, int negative, int vocab_size, int batch_size, unsigned long long next_random):
    cdef int i,j,tar_id,pos,t,n,q,r
    cdef int ctx_count, neg_count, actual_window
    cdef unsigned long long modulo = 281474976710655ULL
    
    # columns of data and its length:  
    #   ctx_indices:  [2 * window]
    #   ctx_mask:     [2 * window]
    #   word_idx:     [1]
    #   neg_indices:  [negative]
    #   neg_mask:     [negative]
    cdef long[:,:] data = np.zeros([sent_id_len,4*window+1+2*negative], dtype=np.int64)

    cdef int* unigram_table
    unigram_table = <int*>ptr_val

    for i in range(sent_id_len):
        ctx_count = 0
        actual_window = rand() % window + 1
        tar_id = sent_id[i]
        for j in range(i-window, i-actual_window):
            pos = j - (i-window)
            data[i, pos] = vocab_size
        for j in range(i-actual_window, i):
            pos = j - (i-window)
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
            else:
                data[i, pos] = sent_id[j]
                data[i, pos + 2*window] = 1
                ctx_count += 1
        for j in range(i+1, i+1+actual_window):
            pos = j - (i-window) - 1
            if j < 0 or j >= sent_id_len:
                data[i, pos] = vocab_size
            else:
                data[i, pos] = sent_id[j]
                data[i, pos + 2*window] = 1
                ctx_count += 1
        for j in range(i+1+actual_window, i+1+window):
            pos = j - (i-window) - 1
            data[i, pos] = vocab_size

        #data[i, 2*window] = ctx_count
        data[i, 4*window] = tar_id

        # negative sampling
        neg_count = 0
        for n in range(negative):
            t = get_unigram_table_at_idx(unigram_table, next_random)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if t == tar_id:
                continue

            data[i, 4*window+1+neg_count] = t
            neg_count += 1

        # neg mask
        for n in range(neg_count):
            data[i, 4*window+1+negative+n] = 1

    return np.asarray(data)

def npmssg_select_sense(long[:] word_ids, float[:,:] context_feats, float[:,:] cluster_embs, 
        int[:,:] word2sense, int[:] word_sense_cnts, float[:] counter_list, int word_ids_len, 
        int emb_dim, int max_senses, float delta, int current_n_senses):
    cdef int b, d, pos, w_id, s_id
    cdef int max_sense_id, new_sense_id, create_count
    cdef float sim, max_sim
    cdef float[:] cluster_emb = np.zeros([emb_dim], dtype=np.float32)
    cdef long[:] senses = np.zeros([word_ids_len], dtype=np.int64)

    create_count = 0
    for b in range(word_ids_len):
        w_id = word_ids[b]
        
        # first encounter
        if counter_list[w_id] == 0.0:
            senses[b] = w_id
            for d in range(emb_dim):
                cluster_embs[w_id, d] += context_feats[b, d]
            counter_list[w_id] += 1.0
            continue

        # not first encounter
        max_sense_id = -1
        max_sim = -10.0
        for s_id in range(word_sense_cnts[w_id]):
            s_id = word2sense[w_id][s_id]
            if counter_list[s_id] == 0:
                print("zero:", s_id)
            for d in range(emb_dim):
                cluster_emb[d] = cluster_embs[s_id, d] / counter_list[s_id]
            sim = cosine(emb_dim, &context_feats[b,0], &cluster_emb[0])
            if sim > max_sim:
                max_sim = sim
                max_sense_id = s_id

        # create new sense
        if word_sense_cnts[w_id] < max_senses:
            if max_sim < delta:
                max_sense_id = current_n_senses + create_count
                word2sense[w_id][ word_sense_cnts[w_id] ] = max_sense_id
                word_sense_cnts[w_id] += 1
                create_count += 1

        senses[b] = max_sense_id
        for d in range(emb_dim):
            cluster_embs[max_sense_id, d] += context_feats[b, d]
        counter_list[max_sense_id] += 1.0

    return senses, create_count


