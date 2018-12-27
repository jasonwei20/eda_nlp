

#user inputs
dataset_folder = 'datasets/subj'
input_size = 50 #number of words
huge_word2vec = 'word2vec/glove.840B.300d.txt'
word2vec_len = 300
word2vec_pickle = dataset_folder + '/word2vec.p' # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary
increments = [0.001, 0.003, 0.01, 0.05, 0.1, 0.25, 0.5, 1]

train_orig = dataset_folder + '/train_orig.txt'
train_aug_st = dataset_folder + '/train_aug_st.txt'
test_path = dataset_folder + '/test.txt'
