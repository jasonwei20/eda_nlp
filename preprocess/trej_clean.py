
from utils import *

class_name_to_num = {'DESC': 0, 'ENTY':1, 'ABBR':2, 'HUM': 3, 'LOC': 4, 'NUM': 5}

def clean(input_file, output_file):
	lines = open(input_file, 'r').readlines()
	writer = open(output_file, 'w')
	for line in lines:
		parts = line[:-1].split(' ')
		tag = parts[0].split(':')[0]
		class_num = class_name_to_num[tag]
		sentence = get_only_chars(' '.join(parts[1:]))
		print(tag, class_num, sentence)
		output_line = str(class_num) + '\t' + sentence
		writer.write(output_line + '\n')
	writer.close()


if __name__ == "__main__":

	clean('raw/trec/train_copy.txt', 'datasets/trec/train_orig.txt')
	clean('raw/trec/test_copy.txt', 'datasets/trec/test.txt')
	