from methods import *
from a_config import *

if __name__ == "__main__":

	#for each method
	for a_method in a_methods:

		#for each data size
		for size_folder in size_folders:

			n_aug_list = n_aug_list_dict[size_folder]
			dataset_folders = [size_folder + '/' + s for s in datasets]

			#for each dataset
			for i, dataset_folder in enumerate(dataset_folders):

				train_orig = dataset_folder + '/train_orig.txt'
				n_aug = n_aug_list[i]

				#for each alpha value
				for alpha in alphas:

					output_file = dataset_folder + '/train_' + a_method + '_' + str(alpha) + '.txt'

					#generate the augmented data
					if a_method == 'sr':
						gen_sr_aug(train_orig, output_file, alpha, n_aug)
					if a_method == 'ri':
						gen_ri_aug(train_orig, output_file, alpha, n_aug)
					if a_method == 'rd':
						gen_rd_aug(train_orig, output_file, alpha, n_aug)
					if a_method == 'rs':
						gen_rs_aug(train_orig, output_file, alpha, n_aug)

				#generate the vocab dictionary
				word2vec_pickle = dataset_folder + '/word2vec.p'
				gen_vocab_dicts(dataset_folder, word2vec_pickle, huge_word2vec)
			
