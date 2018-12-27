from methods import *
from config import *

if __name__ == "__main__":

	#generate the augmented data sets
	print("augmenting data")
	gen_standard_aug(train_orig, train_aug_st)
	print("done")

	#generate the vocab dictionary
	print("generating word 2 vec")
	gen_vocab_dicts(dataset_folder, word2vec_pickle, huge_word2vec)
	print("done")
