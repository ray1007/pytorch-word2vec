## !!! This readme is still in progress, some links or statements may be incorrect. (Last updated: Dec. 31)
## pytorch-word2vec

The aim of this project is to build a model for neural network based word embedding research. This implementation provides autograd with pytorch, and is optimized to have reasonable training speed. Please include the url of this repository if you are using this in your research! :)

### Features:
- Same command line arguments and code structure with the original word2vec **EXCEPT** hierachical softmax. 
- Performance optimized with Cython, multiprocessing, data batching, and CUDA GPU acceleration. 
- Easy modification of the embedding model. Auto-gradient supported by pytorch. 

### Dependencies:
```
python 3.5.2
Cython 0.27.1
pytorch 0.2.0_3
```

### Usage:
- First run cython: `python3 setup.py build_ext --inplace`
- Train word2vec embedding with: `[CUDA_VISIBLE_DEVICES=<device_id>] ./main.py --cuda --train <your_corpus.txt> --output <your_output.txt> --cbow 0 --size 300 --window 5 --sample 1e-4 --negative 5 --min_count 5 --processes 4 --iter 1 --batch_size 100`
- Train word2vec without multiprocessing: `[CUDA_VISIBLE_DEVICES=<device_id>] ./main_simple.py --cuda --train <your_corpus.txt> --output <your_output.txt> --cbow 0 --size 300 --window 5 --sample 1e-4 --negative 5 --iter 1 --batch_size 100`
- Train CSV single prototype with: `[CUDA_VISIBLE_DEVICES=<device_id>] ./csv.py --cuda --train <your_corpus.txt> --save <your_output.txt> --size 300 --window 5 --sample 1e-4 --negative 5 --min_count 5 --processes 4 --iter 1 --batch_size 100`
- Train CSV multi-prototype with: `[CUDA_VISIBLE_DEVICES=<device_id>] ./csv.py --cuda --train <your_corpus.txt> --save <your_output.txt> --size 300 --window 5 --sample 1e-4 --negative 5 --min_count 5 --processes 4 --iter 1 --batch_size 100 --multi-proto`

### Implemented Models:
- CBOW and Skipgram model in `word2vec`. [paper]()
- NP-MSSG model. [paper]()
-  

### Benchmarking:
#### Training speed (Word/sec):
- The following experiment runs are trained on `text8 corpus`, with the same arguments: `-size 300 -window 5 -sample 1e-4 -negative 5 -iter 1`.

| Implementation | arguments | Speed (words/sec) | ratio |
| -------------- | --------- | ----------------: | -----:|
| `original` | `-cbow 1 -threads 4 (batch_size 1)` |  | 1 |
| `pytorch` | `--cuda --cbow 1 --processes 4 --batch_size 100` |  | 0.xx |
| `pytorch` | `--cuda --cbow 1 --processes 4 --batch_size 50`  |  | 0.xx |
| `pytorch` | `--cuda --cbow 1 --processes 4 --batch_size 16`  |  | 0.xx |
| `pytorch` | `--cuda --cbow 1 --processes 4 --batch_size 1`  |  | 0.xx |
| `original` | `-cbow 0 -threads 4 -iter 1 (batch_size 1)` |  | 1 |
| pytorch-word2vec | --cuda --cbow 0 --processes 4 --batch_size 100 |  | 0.xx |
| pytorch-word2vec | --cuda --cbow 0 --processes 4 --batch_size 50  |  | 0.xx |
| pytorch-word2vec | --cuda --cbow 0 --processes 4 --batch_size 16  |  | 0.xx |
| pytorch-word2vec | --cuda --cbow 0 --processes 4 --batch_size 1   |  | 0.xx |

#### GPU Memory Usage

| Implementation | --min_count | GPU memory |
| -------------- | --------- | ----------------: |
| pytorch-word2vec |  |  |
| pytorch-word2vec |  |  |
| pytorch-word2vec |   |  |
| pytorch-word2vec | --cuda --cbow 1 --processes 4 --batch_size 1   |  |
| pytorch-word2vec | --cuda --cbow 0 --processes 4 --batch_size 100 |  |
| pytorch-word2vec | --cuda --cbow 0 --processes 4 --batch_size 50  |  |
| pytorch-word2vec | --cuda --cbow 0 --processes 4 --batch_size 16  |  |
| pytorch-word2vec | --cuda --cbow 0 --processes 4 --batch_size 1   |  |

### Tracing the source code:
- `data_producer.pyx`: contains the cython code for: 
  1. generating both CBOW and skipgram training data from a sentence, 
  2. printing the trained embeddings to file.
  3. allocating word frequency table (`unigram_table`) for negative sampling. 
- `main.py`: the word2vec implementation with optimized speed. Multiple processes are created to parrallelize the training load, and each subprocess (`train_process`) processes the texts and generates training data with functions defined in `data_producer.pyx`. Training data is stored in a queue `data_queue` (this is a producer-consumer model). GPU accelertion (CUDA) supported.
- `main_simple.py`: the word2vec implementation without multiprocessing. Easier for debugging.
- `csv.py`: Re-implemented CSV model in [Learning Context Specific Word/Character Vectors, AAAI 2017](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14601). Currently debugging. 

### TODO:
Re-implement the following models:
- Learning Context Specific Word/Character Vectors, AAAI 2017
- Multimodal Word Distribution, ACl 2017

### Development Notes:
#### word2vec
- The original word2vec is really fast. In addition to the fact that it is written in C, the training file is split into chunks which are processed by multiple threads, and each thread asynchronously updates the model parameters. Subsampling and negative sampling are done with random number generator. The calcualtion of gradients is further optimized by using pre-calculated sigmoid tables. Instead of performing exponential arithmetics in the runtime, it looks up the sigmoid table for the computation result. 

- word2vec is probably the most known embedding model in world. Nonetheless when I was re-inplementing word2vec, I discovered some details probably not known by most people. 
   - Dynamic window: The `-window` argument in fact sets the range of "dynamic window". Say we set it to 5 (left 5 + right 5), original `word2vec` would randomly choose an integer from {1,...,5} for each target word as its actual window size.
   - CBOW gradient: When training CBOW embeddings, mean of context word embeddings is used to predict the target word. The gradients of context words is propagated in the following order: `loss` -> `inner product` -> `mean of context embeddings` -> `context embedding`. However in the original code, the gradients of context embeddings are the same as the mean, which means the actual gradients are not scaled by the length of the window. Wondering waht's gonna happend when gradients are scaled, yet haven't tried :P.
   - negative sampling, end-of-sentence <to-be-finished...> 
   
- There is also a Tensorflow [tutorial](https://www.tensorflow.org/tutorials/word2vec) and [implementation](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py) of word2vec, along with an [optimized version](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py) which uses a compiled kernel for training ops in order to have a reasonable training speed.

#### Gensim
- [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) is python implementation of word2vec. The author wrote 3 articles ([#1](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim/) [#2](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/) [#3](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/)) about the techniques used in `gensim`, from which I learned a lot! In [#2](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/), the author pointed out that the performance bottleneck is the double loop iterating the context window. (To confirm, I did simple profiling and found the same) To reach a reasonable training speed, it's crucial to speed up the loop, and the solution is cython. In [#3](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/), 

#### Cython
- To learn about cython, I read [this](https://python.g-node.org/python-summerschool-2011/_media/materials/cython/cython-slides.pdf) clear explanation written by Pauli Virtanen. There is some overhead in python since it's a dynamic typed language (the variables does not have a explicit type like C/C++/Java). When executing a python script, everytime the interpreter encounters a variable, it checks the underlying type of that variable. What I did with cython was basically providing the type information of a variable by explicitly declaring the actual type (`int`, `float`, ..etc) with `cdef`. I guess in the double loop, the overhead for accessing variables would be O(n^2), that's why providing type information offers a significant speed-up. Also, I found it convenient to use the command `cython -a <your_script>`. It generates a HTML file showing the script and highlights the line that causes performance bottleneck.

#### Pytorch 
- pytorch Multiprocessing
When using CUDA with multiprocessing, one has to set the start method to 'spawn or 'forkserver' with `set_start_method()` method in `__main__()`.
- batching prevents GPU out-of-memory problem.
- found that adding `sparse=True` of `torch.nn.Embedding()` reduces GPU memory usage.
- found that using memeory view instead of numpy array in cython is a bit faster.


