from methods import *
from a_config import *

if __name__ == "__main__":

	#generate the augmented data sets

	#for each dataset
	for dataset_folder in dataset_folders:
		train_orig = dataset_folder + '/train_orig.txt'

		#for each alpha value
		for alpha in alphas:

			#generate the augmented data
			output_file = dataset_folder + '/train_sr_' + str(alpha) + '.txt'
			gen_sr_aug(train_orig, output_file, alpha)

		#generate the vocab dictionary
		word2vec_pickle = dataset_folder + '/word2vec.p'
		gen_vocab_dicts(dataset_folder, word2vec_pickle, huge_word2vec)
			
