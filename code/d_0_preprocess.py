from methods import *

def generate_short(input_file, output_file, alpha):
	lines = open(input_file, 'r').readlines()
	increment = int(len(lines)/alpha)
	lines = lines[::increment]
	writer = open(output_file, 'w')
	for line in lines:
		writer.write(line)

if __name__ == "__main__":

	#global params
	huge_word2vec = 'word2vec/glove.840B.300d.txt'
	datasets = ['pc']#, 'trec']

	for dataset in datasets:

		dataset_folder = 'special_f4/' + dataset
		test_short = 'special_f4/' + dataset + '/test_short.txt'
		test_aug_short = dataset_folder + '/test_short_aug.txt'
		word2vec_pickle = dataset_folder + '/word2vec.p' 

		#augment the data
		gen_tsne_aug(test_short, test_aug_short)

		#generate the vocab dictionaries
		gen_vocab_dicts(dataset_folder, word2vec_pickle, huge_word2vec)











