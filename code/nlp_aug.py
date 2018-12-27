# Easy data augmentation techniques for text classification
# Jason Wei, Chengyu Huang, Yifang Wei, Fei Xing, Kai Zou
# input for all methods is a list of words

import random

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			words = [synonym if word == random_word else word for word in words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break
	sentence = ' '.join(words)
	words = sentence.split(' ')
	return words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete n words from the sentence
########################################################################

def random_deletion(words, n):
	for _ in range(n):
		delete_word(words)
	return words

def delete_word(words):
	if len(words) >= 5:
		random_idx = random.randint(0, len(words)-1)
		words.pop(random_idx)

########################################################################
# Random swap
# Randomly swap two words from the sentence n times
########################################################################

def random_swap(words, n):
	for _ in range(n):
		words = swap_word(words)
	return words

def swap_word(words):
	random_idx_1 = random.randint(0, len(words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(words)-1)
		counter += 1
		if counter > 3:
			return words
	words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1] 
	return words

########################################################################
# Random addition
# Randomly add n words into the sentence
########################################################################

def random_addition(words, n):
	for _ in range(n):
		add_word(words)
	return words

def add_word(words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = words[random.randint(0, len(words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(words)-1)
	words.insert(random_idx, random_synonym)

########################################################################
# Sliding window
# Slide a window of size w over the sentence with stride s
# Returns a list of lists of words
########################################################################

def sliding_window_sentences(words, w, s):
	windows = []
	for i in range(0, len(words)-w+1, s):
		window = words[i:i+w]
		windows.append(window)
	return windows

########################################################################
# For each sentence, generate three different sentences using each technique
# synonym replacement: n=3
# random deletion: n=2
# random swap: n=2
# random insertion: n=2
# return a list of sentences (strings)
########################################################################

def standard_augmentation(sentence, sr=3, rd=2, rs=2, ri=2, num=3):
	sentence = get_only_chars(sentence)
	augmented_sentences = [sentence]
	words = sentence.split(' ')
	for _ in range(num):
		a_words = synonym_replacement(words, sr)
		augmented_sentences.append(' '.join(a_words))
	for _ in range(num):
		a_words = random_deletion(words, rd)
		augmented_sentences.append(' '.join(a_words))
	for _ in range(num):
		a_words = random_swap(words, rs)
		augmented_sentences.append(' '.join(a_words))
	for _ in range(num):
		a_words = random_addition(words, ri)
		augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	return augmented_sentences

########################################################################
# Testing
########################################################################

if __name__ == '__main__':

	line = 'Hi. My name is Jason. I’m a third-year computer science major at Dartmouth College, interested in deep learning and computer vision. My advisor is Saeed Hassanpour. I’m currently working on deep learning for lung cancer classification.'
	a_lines = standard_augmentation(line, sr=1, rd=1, rs=1, ri=1, num=1)
	for l in a_lines:
		print(l)


