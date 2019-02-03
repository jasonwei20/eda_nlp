#user inputs

#size folders
sizes = ['1_tiny', '2_small', '3_standard', '4_full']
size_folders = ['size_data_f1/' + size for size in sizes]

#augmentation methods
a_methods = ['sr', 'ri', 'rd', 'rs']

#dataset folder
datasets = ['cr', 'sst2', 'subj', 'trec', 'pc']

#number of output classes
num_classes_list = [2, 2, 2, 6, 2]

#number of augmentations
n_aug_list_dict = {'size_data_f1/1_tiny': [16, 16, 16, 16, 16], 
					'size_data_f1/2_small': [16, 16, 16, 16, 16],
					'size_data_f1/3_standard': [8, 8, 8, 8, 4],
					'size_data_f1/4_full': [8, 8, 8, 8, 4]}

#alpha values we care about
alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

#number of words for input
input_size_list = [50, 50, 40, 25, 25] 

#word2vec dictionary
huge_word2vec = 'word2vec/glove.840B.300d.txt'
word2vec_len = 300 # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary
