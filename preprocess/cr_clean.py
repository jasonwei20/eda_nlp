#0 = neg, 1 = pos
from utils import *

def retrieve_reviews(line):

	reviews = set()
	chars = list(line)
	for i, char in enumerate(chars):
		if char == '[':
			if chars[i+1] == '-':
				reviews.add(0)
			elif chars[i+1] == '+':
				reviews.add(1)
	
	reviews = list(reviews)
	if len(reviews) == 2:
		return -2
	elif len(reviews) == 1:
		return reviews[0]
	else:
		return -1

def clean_files(input_files, output_file):

	writer = open(output_file, 'w')

	for input_file in input_files:
		print(input_file)
		input_lines = open(input_file, 'r').readlines()
		counter = 0
		bad_counter = 0
		for line in input_lines:
			review = retrieve_reviews(line)
			if review in {0, 1}:
				good_line = get_only_chars(re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", line))
				output_line = str(review) + '\t' + good_line
				writer.write(output_line + '\n')
				counter += 1
			elif review == -2:
				bad_counter +=1 
		print(input_file, counter, bad_counter)

	writer.close()

if __name__ == '__main__':

	input_files = ['all.txt']#['canon_power.txt', 'canon_s1.txt', 'diaper.txt', 'hitachi.txt', 'ipod.txt', 'micromp3.txt', 'nokia6600.txt', 'norton.txt', 'router.txt']
	input_files = ['raw/cr/data_new/' + f for f in input_files]
	output_file = 'datasets/cr/apex_clean.txt'

	clean_files(input_files, output_file)
