#user inputs

#size folders
sizes = ['3_standard']#, '4_full']#['1_tiny', '2_small', '3_standard', '4_full']
size_folders = ['size_data_f3/' + size for size in sizes]

#dataset folder
datasets = ['cr', 'sst2', 'subj', 'trec', 'pc']

#number of output classes
num_classes_list = [2, 2, 2, 6, 2]

#alpha values we care about
num_aug_list = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]

#number of words for input
input_size_list = [50, 50, 50, 25, 25] 

#word2vec dictionary
huge_word2vec = 'word2vec/glove.840B.300d.txt'
word2vec_len = 300 # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary
