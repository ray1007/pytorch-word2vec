## pytorch-word2vec

The aim of this project is to build a model for neural network based word embedding research. This implementation provides autograd with pytorch, and is optimized to have reasonable training speed.

### Features:
- Inherits the same command line arguments and code structure of the original word2vec EXCEPT hierachical softmax. 
- Performance optimized with Cython, multiprocessing, data batching, and CUDA GPU acceleration. 
- Easy modification of the embedding model. (Defined with pytorch) 

### Usage:
- `python3 setup.py build_ext --inplace`
- `CUDA_VISIBLE_DEVICES=<device_id> ./main.py --cuda --train <your_corpus.txt> --output --iters --processes`
- `./main.py --train <your_corpus.txt>`

### Benchmarking:
- Training speed
- GPU Memory Usage

### Related works:
- The original word2vec is really fast. Besides the fact that it is written in C, it splits the training file into chunks. Each chunk is read by multiple threads and each thread asynchronously updates the model parameters. Sampling is done by random number generator. The calcualtion of gradients is further optimized by using pre-calculated look-up tables instead of performing exponential arithmetics in the runtime. 
- [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) is reported to be a faster python implementation of word2vec. Gensim cythonized each thread. It also built a look-up table for exponential. The author wrote 3 articles about the techniques used in gensim. Version 1: [With Numpy](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim/), Version 2: [Add Cython & BLAS](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/), Version 3: [Add Parallelizing](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/).
- There is also a Tensorflow [tutorial]() and [implementation]() of word2vec, along with an [optimized version]() which uses a kernel. C optimized code to produce negative samples in order to have a reasonable training speed.

- This implementation could be viewed as a variant of gensim. speed up from Cython optimized code. The double loop to traverse a sentence to get the word and context is cythonized. 

### Tracing the source code:
- `data_producer.pyx` ...
- `main.py` ...

### Optimization notes:
- Cython
python variable. Providing type information.
- Multiprocessing
When using CUDA with multiprocessing, one has to set the start method to 'spawn or 'forkserver' with `set_start_method()` method in `__main__()`.
- #put embedding in CPU, only move to GPU after lookup operation.
- batching prevents GPU out-of-memory problem.
- found that adding `sparse=True` of `torch.nn.Embedding()` reduces GPU memory usage.

### Features to be added:
- random number generator in C (Cython)
- lr anneal
- skip-gram
