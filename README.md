# EDA-NLP
This is the code for the paper [EDA: Easy Data Augmentation techniques for boosting performance on text classification tasks.](https://arxiv.org/abs/1901.11196)

We present **EDA**: **e**asy **d**ata **a**ugmentation techniques for boosting performance on text classification tasks. These are a generalized set of data augmentation techniques that are easy to implement and have shown improvements over 5 nlp classification tasks, which larger improvements on datasets of size *N<500*. While other techniques require you to train a language model on an external dataset just to get a small boost, we found that simple text editing operations using EDA result in substantial performance gains. Given a sentence of *l* ordered words *[w_1, w_2, ..., w_l]*, we perform the following operations:

- **Synonym Replacement (SR):** Randomly choose *n* words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.
- **Random Insertion (RI):** Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this *n* times.
- **Random Swap (RS):** Randomly choose two words in the sentence and swap their positions. Do this *n* times.
- **Random Deletion (RD):** For each word in the sentence, randomly remove it with probability *p*.

# Usage

You can run EDA any text classification dataset in less than 5 minutes. Just two steps:

## Install NLTK (if you don't have it already):

Pip install it.

```
pip install -U nltk
```

Download WordNet.
```
python
>>> import nltk; nltk.download('wordnet')
```

## Run EDA

You can easily write your own implementation, but this one takes input files in the format `label\tsentence` (note the `\t`). So for instance, your input file should look like this (example from stanford sentiment treebank):

```
1   neil burger here succeeded in making the mystery of four decades back the springboard for a more immediate mystery in the present 
0   it is a visual rorschach test and i must have failed 
0   the only way to tolerate this insipid brutally clueless film might be with a large dose of painkillers
...
```

Now place this input file into the `data` folder. Run 

```
python code/augment.py --input=<insert input filename>
```

The default output filename will append `eda_` to the input filename, but you can specify your own with `--output`. You can also specify the number of generated augmented sentences per original sentence using `--num_aug` (default is 9). So for example, if your input file is `sst2_train.txt` and you want to output to `sst2_augmented.txt` with 16 augmented sentences per original sentence, you would do:

```
python code/augment.py --input=sst2_train.txt --output=sst2_augmented.txt --num_aug=16
```

Best of luck!

# Experiments (Coming soon)

### Word embeddings
Download [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) and place in a folder named `word2vec`.
