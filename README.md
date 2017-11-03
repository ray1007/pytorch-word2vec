## !!! This readme is still in progress, some links or statements may be incorrect. (Last updated: Nov. 3)
## pytorch-word2vec

The aim of this project is to build a model for neural network based word embedding research. This implementation provides autograd with pytorch, and is optimized to have reasonable training speed.

### Features:
- Inherits the same command line arguments and code structure of the original word2vec EXCEPT hierachical softmax. 
- Performance optimized with Cython, multiprocessing, data batching, and CUDA GPU acceleration. 
- Easy modification of the embedding model. (Defined with pytorch) 

### Dependencies:
```
Cython 0.xx+
pytorch 0.2
```

### Usage:
- First run cython: `python3 setup.py build_ext --inplace`
- `[CUDA_VISIBLE_DEVICES=<device_id>] ./main.py --cuda --train <your_corpus.txt> --output --iters --processes`
- `./main.py --train <your_corpus.txt>`

### Benchmarking:
- Training speed
- GPU Memory Usage


### Tracing the source code:
- `data_producer.pyx`: contains the cython code for: 
  1. generating both CBOW and skipgram training data from a sentence, 
  2. printing the trained embeddings to file.
- `main.py`: the word2vec implementation with optimized speed. Multiple processes are created to parrallelize the training load, and each subprocess (`train_process`) creates threads (`train_sent_producer`) that process the texts and generate training data. Training data is stored in a queue (this is a producer-consumer model). Supports GPU accelertion (CUDA).
- `main_simple.py`: the word2vec implementation without multiprocessing. Easier for debugging.

### Development and Optimization notes:
- The original word2vec is really fast. In addition to the fact that it is written in C, the training file is split into chunks which are processed by multiple threads, and each thread asynchronously updates the model parameters. Subsampling and negative sampling are done with random number generator. The calcualtion of gradients is further optimized by using pre-calculated sigmoid tables. Instead of performing exponential arithmetics in the runtime, it looks up the sigmoid table for the computation result. 

- word2vec is probably the most known embedding model in world. Nonetheless when I was re-inplementing word2vec, I discovered some details probably not known by most people. dynamic window, CBOW gradient, negative sampling, end-of-sentence <to-be-finished...> 

- [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) is reported to be a faster python implementation of word2vec. The author wrote 3 articles ([#1](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim/) [#2](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/) [#3](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/))about the techniques used in `gensim`, from which I learned a lot! In [#2](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/), the author pointed out that the performance bottleneck is the double loop iterating the context window. (To confirm, I did simple profiling and found the same) To reach a reasonable training speed, it's crucial to speed up the loop, and the solution is cython. To learn about cython, I read [this](https://python.g-node.org/python-summerschool-2011/_media/materials/cython/cython-slides.pdf) clear explanation written by Pauli Virtanen. There is some overhead in python since it's a dynamic typed language (the variables does not have a explicit type like C/C++/Java). When executing a python script, everytime the interpreter encounters a variable, it checks the underlying type of that variable. What I did with cython was basically providing the type information of a variable by explicitly declaring the actual type (`int`, `float`, ..etc) with `cdef`. I guess in the double loop, the overhead for accessing variables would be O(n^2), that's why providing type information offers a significant speed-up. Also, I found it convenient to use the command `cython -a <your_script>`. It generates a HTML file showing the script and highlights the line that causes performance bottleneck.

- There is also a Tensorflow [tutorial]() and [implementation](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py) of word2vec, along with an [optimized version](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py) which uses a compiled kernel for training ops in order to have a reasonable training speed.

- Pytorch Multiprocessing
When using CUDA with multiprocessing, one has to set the start method to 'spawn or 'forkserver' with `set_start_method()` method in `__main__()`.
- batching prevents GPU out-of-memory problem.
- found that adding `sparse=True` of `torch.nn.Embedding()` reduces GPU memory usage.
- found that using memeory view instead of numpy array in cython is a bit faster.

### TODO:
- Learning Context Specific Word/Character Vectors, AAAI 2017
- Multimodal Word Distribution, ACl 2017
