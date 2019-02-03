from methods import *
from e_config import *

if __name__ == "__main__":

	for size_folder in size_folders:

		dataset_folders = [size_folder + '/' + s for s in datasets]
		n_aug_list = n_aug_list_dict[size_folder]

		#for each dataset
		for i, dataset_folder in enumerate(dataset_folders):

			n_aug = n_aug_list[i]

			#pre-existing file locations
			train_orig = dataset_folder + '/train_orig.txt'

			#file to be created
			train_aug_st = dataset_folder + '/train_aug_st.txt'

			#standard augmentation
			gen_standard_aug(train_orig, train_aug_st, n_aug)

			#generate the vocab dictionary
			word2vec_pickle = dataset_folder + '/word2vec.p'
			gen_vocab_dicts(dataset_folder, word2vec_pickle, huge_word2vec)
		
