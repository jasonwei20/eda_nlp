#user inputs

#dataset folder
dataset_folder = 'sized_datasets_f1/4_full' 
datasets = ['cr', 'sst1', 'sst2', 'subj', 'trec', 'pc']
dataset_folders = [dataset_folder + '/' + s for s in datasets]

#number of output classes
num_classes_list = [2, 5, 2, 2, 6, 2]

#alpha values we care about
alphas = [0.05, 0.1, 0.2, 0.3, 0.5]

#best epochs for each dataset
epochs = [30, 20, 15, 20, 50, 20]


#number of words for input
input_size = 50 

#word2vec dictionary
huge_word2vec = 'word2vec/glove.840B.300d.txt'
word2vec_len = 300 # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary

#pre-existing file locations
test_path = dataset_folder + '/test.txt'

#files to be created
train_aug_st = dataset_folder + '/train_aug_st.txt'
