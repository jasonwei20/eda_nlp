
from utils import *

def get_good_stuff(line):
	idx = line.find('s>')
	good = line[idx+2:-8]

	return get_only_chars(good)

def clean_file(con_file, pro_file, output_train, output_test):

	train_writer = open(output_train, 'w')
	test_writer = open(output_test, 'w')
	con_lines = open(con_file, 'r').readlines()
	for line in con_lines[:int(len(con_lines)*0.9)]:
		content = get_good_stuff(line)
		if len(content) >= 8:
			train_writer.write('0\t' + content + '\n')
	for line in con_lines[int(len(con_lines)*0.9):]:
		content = get_good_stuff(line)
		if len(content) >= 8:
			test_writer.write('0\t' + content + '\n')

	pro_lines = open(pro_file, 'r').readlines()
	for line in pro_lines[:int(len(con_lines)*0.9)]:
		content = get_good_stuff(line)
		if len(content) >= 8:
			train_writer.write('1\t' + content + '\n')
	for line in pro_lines[int(len(con_lines)*0.9):]:
		content = get_good_stuff(line)
		if len(content) >= 8:
			test_writer.write('1\t' + content + '\n')


if __name__ == '__main__':
	
	con_file = 'raw/pros-cons/integratedCons.txt'
	pro_file = 'raw/pros-cons/integratedPros.txt'
	output_train = 'datasets/procon/train.txt'
	output_test = 'datasets/procon/test.txt'
	clean_file(con_file, pro_file, output_train, output_test)