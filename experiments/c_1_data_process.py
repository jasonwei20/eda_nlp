from methods import *
from c_config import *

if __name__ == "__main__":

	#generate the augmented data sets

	for size_folder in size_folders:

		dataset_folders = [size_folder + '/' + s for s in datasets]

		#for each dataset
		for dataset_folder in dataset_folders:
			train_orig = dataset_folder + '/train_orig.txt'

			#for each n_aug value
			for num_aug in num_aug_list:

				output_file = dataset_folder + '/train_' + str(num_aug) + '.txt'

				#generate the augmented data
				if num_aug > 4 and '4_full/pc' in train_orig:
					gen_standard_aug(train_orig, output_file, num_aug=4)
				else:
					gen_standard_aug(train_orig, output_file, num_aug=num_aug)

			#generate the vocab dictionary
			word2vec_pickle = dataset_folder + '/word2vec.p'
			gen_vocab_dicts(dataset_folder, word2vec_pickle, huge_word2vec)
			
