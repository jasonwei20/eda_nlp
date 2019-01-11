from a_config import * 
from methods import *
from numpy.random import seed
seed(0)

def run_model(train_file, test_file, num_classes, percent_dataset, epochs_base):

	print("running", train_file)

	#initialize model
	model = build_model(input_size, word2vec_len, num_classes)

	#load data
	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec, percent_dataset)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec, 1)

	#train model
	n_epochs = min(500, int(epochs_base/percent_dataset))
	model.fit(train_x, train_y, batch_size=1024, epochs=n_epochs, validation_split=0.1, shuffle=True, verbose=0)
	#model.save('checkpoints/lol')
	#model = load_model('checkpoints/lol')

	#evaluate model
	y_pred = model.predict(test_x)
	test_y_cat = one_hot_to_categorical(test_y)
	y_pred_cat = one_hot_to_categorical(y_pred)
	acc = accuracy_score(test_y_cat, y_pred_cat)

	#return the accuracy
	print("data with shape:", train_x.shape, train_y.shape, 'train=', train_file, 'test=', test_file, 'with fraction', percent_dataset, 'had acc', acc)
	return acc

if __name__ == "__main__":

	performances = {alpha:[] for alpha in alphas}
	#for each dataset value
	for i in range(len(dataset_folders)):

		dataset_folder = dataset_folders[i]
		dataset = datasets[i]
		num_classes = num_classes_list[i]
		num_epochs = epochs[i]
		word2vec_pickle = dataset_folder + '/word2vec.p'
		word2vec = load_pickle(word2vec_pickle)

		for alpha in alphas:

			train_path = dataset_folder + '/train_sr_' + str(alpha) + '.txt'
			test_path = 'sized_datasets_f1/test/' + dataset + '/test.txt'
			acc = run_model(train_path, test_path, num_classes, percent_dataset=1, epochs_base=num_epochs)
			performances[alpha].append(acc)
			


	print(performances)
