from config import * 
from methods import *
from numpy.random import seed
seed(0)

def run_model(train_file, test_file, percent_dataset, epochs_base):

	#initialize model
	model = build_model(input_size, word2vec_len)

	#load data
	word2vec = load_pickle(word2vec_pickle)
	train_x, train_y = get_x_y(train_file, word2vec_len, input_size, word2vec, percent_dataset)
	print("loaded data with shape:", train_x.shape, train_y.shape)
	test_x, test_y = get_x_y(test_file, word2vec_len, input_size, word2vec, 1)

	#train model
	n_epochs = min(500, int(epochs_base/percent_dataset))
	model.fit(train_x, train_y, batch_size=1024, epochs=n_epochs, validation_split=0.1, shuffle=True, verbose=0)
	#model = load_model('lol')
	#model.save('lol')

	#evaluate model
	y_pred = model.predict(test_x)
	y_pred_binary = conf_to_pred(y_pred)
	acc = accuracy_score(test_y, y_pred_binary)

	#return the accuracy
	print(train_file, test_file, percent_dataset, acc)
	return acc

if __name__ == "__main__":

	#get the accuracy at each increment
	orig_accs = []
	aug_accs = []

	for increment in increments:
		orig_acc = run_model(train_orig, test_path, increment, epochs_base=5)
		orig_accs.append(orig_acc)
		aug_acc = run_model(train_aug_st, test_path, increment, epochs_base=2)
		aug_accs.append(aug_acc)

	print(orig_accs, aug_accs)
