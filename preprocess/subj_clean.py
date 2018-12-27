from utils import *

if __name__ == "__main__":
	subj_path = "subj/rotten_imdb/subj.txt"
	obj_path = "subj/rotten_imdb/plot.tok.gt9.5000"

	subj_lines = open(subj_path, 'r').readlines()
	obj_lines = open(obj_path, 'r').readlines()
	print(len(subj_lines), len(obj_lines))

	test_split = int(0.9*len(subj_lines))
	
	train_lines = []
	test_lines = []

	#training set
	for s_line in subj_lines[:test_split]:
		clean_line = '1\t' + get_only_chars(s_line[:-1])
		train_lines.append(clean_line)

	for o_line in obj_lines[:test_split]:
		clean_line = '0\t' + get_only_chars(o_line[:-1])
		train_lines.append(clean_line)

	#testing set
	for s_line in subj_lines[test_split:]:
		clean_line = '1\t' + get_only_chars(s_line[:-1])
		test_lines.append(clean_line)

	for o_line in obj_lines[test_split:]:
		clean_line = '0\t' + get_only_chars(o_line[:-1])
		test_lines.append(clean_line)

	print(len(test_lines), len(train_lines))

	#print training set
	writer = open('datasets/subj/train_orig.txt', 'w')
	for line in train_lines:
		writer.write(line + '\n')
	writer.close()

	#print testing set
	writer = open('datasets/subj/test.txt', 'w')
	for line in test_lines:
		writer.write(line + '\n')
	writer.close()