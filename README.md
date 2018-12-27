# nlp_augmentation
Easy data augmentation techniques for boosting performance on text classification tasks.

We document the following data augmentation techniques:

1. Synonym replacement: replace *n* words with synonyms from WordNet
2. Random deletion: randomly delete *n* random words
3. Random swap: randomly swap two words *n* times
4. Random insertion: randomly insert a word *n* times
5. Sliding window: slide a window of size *w* with stride *s* over the text input
6. Jittering: add *c* to *n* random word embeddings or take *e^c* for *n* random word embeddings
7. Random noising: add Gaussian noise to *n* random word embeddings

## Usage

### Data format
First place the training file in the format `label\tsentence` in `datasets/dataset/train_orig.txt` and the testing file in the same format in `datasets/dataset/test.txt`.

### Word embeddings
Download [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) and place in a folder named `word2vec`.

### The config file
Take a look at the hyperparameters in `config.py` before you begin. What increments do you want? Are the file names correct?

### Augment the data and load the word2vec dictionary
```
python code/1_data_process.py
```

### Train the model and evaluate it
```
python code/2_train_eval.py
```
