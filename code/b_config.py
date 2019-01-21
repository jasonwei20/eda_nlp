#user inputs

#dataset folder
datasets = ['pc']#['cr', 'sst2', 'subj', 'trec', 'pc']
dataset_folders = ['increment_datasets_f2/' + dataset for dataset in datasets] 

#number of output classes
num_classes_list = [2]#[2, 2, 2, 6, 2]

#dataset increments
increments = [0.7, 0.8, 0.9, 1]#[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#number of words for input
input_size_list = [25]#[50, 50, 40, 25, 25]

#word2vec dictionary
huge_word2vec = 'word2vec/glove.840B.300d.txt'
word2vec_len = 300