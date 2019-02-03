# EDA-NLP
This is the code for [EDA: Easy Data Augmentation techniques for boosting performance on text classification tasks.](https://arxiv.org/abs/1901.11196)

We present the following data augmentation techniques:

Given a sentence consisting of *l* ordered words *[w_1, w_2, ..., w_l]*, we perform the following operations:
Synonym Replacement (SR): Randomly choose *n* words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.
Random Insertion (RI): Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this *n* times.
Random Swap (RS): Randomly choose two words in the sentence and swap their positions. Do this *n* times.
Random Deletion (RD): For each word in the sentence, randomly remove it with probability *p*.

## Usage

### Data format
First place the training file in the format `label\tsentence` in `datasets/dataset/train_orig.txt` and the testing file in the same format in `datasets/dataset/test.txt`.

### Word embeddings
Download [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) and place in a folder named `word2vec`.

### Augment the data and load the word2vec dictionary
```
python code/1_data_process.py
```

#Experiments

Dependencies: tensorflow, keras, sklearn

```
pip install tensorflow-gpu
pip install keras
pip install sklearn
pip install -U nltk
```


### Train the model and evaluate it
```
python code/2_train_eval.py
```

## EDA-7?:
5. Sliding window: slide a window of size *w* with stride *s* over the text input
6. Jittering: add *c* to *n* random word embeddings or take *e^c* for *n* random word embeddings
7. Random noising: add Gaussian noise to *n* random word embeddings
