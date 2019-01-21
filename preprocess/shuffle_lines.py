import random

def shuffle_lines(text_file):
	lines = open(text_file).readlines()
	random.shuffle(lines)
	open(text_file, 'w').writelines(lines)

shuffle_lines('special_f4/pc/test_short_aug_shuffle.txt')