#user inputs

#load hyperparameters
sizes = ['4_full']#['1_tiny', '2_small', '3_standard', '4_full']
size_folders = ['size_data_t1/' + size for size in sizes]

#datasets
datasets = ['cr', 'sst2', 'subj', 'trec', 'pc']

#number of output classes
num_classes_list = [2, 2, 2, 6, 2]

#number of augmentations per original sentence
n_aug_list_dict = {'size_data_t1/1_tiny': [32, 32, 32, 32, 32], 
					'size_data_t1/2_small': [32, 32, 32, 32, 32],
					'size_data_t1/3_standard': [16, 16, 16, 16, 4],
					'size_data_t1/4_full': [16, 16, 16, 16, 4]}

#number of words for input
input_size_list = [50, 50, 40, 25, 25]

#word2vec dictionary
huge_word2vec = 'word2vec/glove.840B.300d.txt'
word2vec_len = 300