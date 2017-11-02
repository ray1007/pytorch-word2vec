## !!! This readme is still in progress, some links or statements may be incorrect.
## pytorch-word2vec

The aim of this project is to build a model for neural network based word embedding research. This implementation provides autograd with pytorch, and is optimized to have reasonable training speed.

### Features:
- Inherits the same command line arguments and code structure of the original word2vec EXCEPT hierachical softmax. 
- Performance optimized with Cython, multiprocessing, data batching, and CUDA GPU acceleration. 
- Easy modification of the embedding model. (Defined with pytorch) 

### Dependencies:
```
Cython 0.xx+
pytorch 0.2+
```

### Usage:
- First run cython: `python3 setup.py build_ext --inplace`
- `[CUDA_VISIBLE_DEVICES=<device_id>] ./main.py --cuda --train <your_corpus.txt> --output --iters --processes`
- `./main.py --train <your_corpus.txt>`

### Benchmarking:
- Training speed
- GPU Memory Usage

### Related works:
- The original word2vec is really fast. In addition to the fact that it is written in C, the training file is split into chunks which are processed by multiple threads, and each thread asynchronously updates the model parameters. Subsampling and negative sampling are done with random number generator. The calcualtion of gradients is further optimized by using pre-calculated sigmoid tables. Instead of performing exponential arithmetics in the runtime, it looks up the sigmoid table for the computation result. 
- [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) is reported to be a faster python implementation of word2vec. The author wrote 3 articles about the techniques used in optimizing `gensim`. Version 1: [With Numpy](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim/), Version 2: [Add Cython & BLAS](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/), Version 3: [Add Parallelizing](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/). Gensim cythonized each thread. It also built a look-up table for exponential.
- There is also a Tensorflow [tutorial]() and [implementation](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py) of word2vec, along with an [optimized version](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py) which uses a compiled kernel for training ops in order to have a reasonable training speed.

- This implementation could be viewed as a variant of gensim. speed up from Cython optimized code. The double loop to traverse a sentence to get the word and context is cythonized. 

### Tracing the source code:
- `data_producer.pyx` ...
- `main.py` ...

### Development and Optimization notes:
- Cython
python variable. Providing type information.
- Pytorch Multiprocessing
When using CUDA with multiprocessing, one has to set the start method to 'spawn or 'forkserver' with `set_start_method()` method in `__main__()`.
- batching prevents GPU out-of-memory problem.
- found that adding `sparse=True` of `torch.nn.Embedding()` reduces GPU memory usage.
- found that using memeory view instead of numpy array in cython is a bit faster.

### Features to be added:
- 
