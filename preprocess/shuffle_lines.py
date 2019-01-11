import random

def shuffle_lines(text_file):
	lines = open(text_file).readlines()
	random.shuffle(lines)
	open(text_file, 'w').writelines(lines)

shuffle_lines('increment_datasets_f2/subj/train_orig.txt')