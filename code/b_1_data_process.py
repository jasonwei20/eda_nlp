from methods import *
from b_config import *

if __name__ == "__main__":

	#generate the augmented data sets
	for dataset_folder in dataset_folders:

		#pre-existing file locations
		train_orig = dataset_folder + '/train_orig.txt'

		#file to be created
		train_aug_st = dataset_folder + '/train_aug_st.txt'

		#standard augmentation
		gen_standard_aug(train_orig, train_aug_st)

		#generate the vocab dictionary
		word2vec_pickle = dataset_folder + '/word2vec.p' # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary
		gen_vocab_dicts(dataset_folder, word2vec_pickle, huge_word2vec)
