# EDA-NLP
Easy data augmentation techniques for boosting performance on text classification tasks.

We propose the following data augmentation techniques:

1. **Synonym replacement (SR):** Randomly choose *n* non-stop words from the sentence, and replace those words a randomly selected synonyms.
2. **Random insertion (RI):** Retrieve *n* words that are synonyms of any non-stop word in the sentence. Randomly insert those words into the sentence.
3. **Random swap (RS):** Randomly choose two words in the sentence and swap their positions. Do this *n* times.
4. **Random deletion (RD):** Randomly choose *n* words from the sentence and remove them.

We run a grid search over n = min(1, floor(alpha/l)), where alpha = {0.05, 0.1, 0.2, 0.3, 0.5}

Also run a grid search for s augmented sentences per technique per sentence for s = {1, 2, 3, 5, 10}

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

## EDA-7?:
5. Sliding window: slide a window of size *w* with stride *s* over the text input
6. Jittering: add *c* to *n* random word embeddings or take *e^c* for *n* random word embeddings
7. Random noising: add Gaussian noise to *n* random word embeddings