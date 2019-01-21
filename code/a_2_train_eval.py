from a_config import * 
from methods import *
from numpy.random import seed
seed(5)

###############################
#### run model and get acc ####
###############################

def run_cnn(train_file, test_file, num_classes, percent_dataset):

	#initialize model
	model = build_cnn(input_size, word2vec_len, num_classes)

	#load data
	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec, percent_dataset)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec, 1)

	#implement early stopping
	callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

	#train model
	model.fit(	train_x, 
				train_y, 
				epochs=100000, 
				callbacks=callbacks,
				validation_split=0.1, 
				batch_size=1024, 
				shuffle=True, 
				verbose=0)
	#model.save('checkpoints/lol')
	#model = load_model('checkpoints/lol')

	#evaluate model
	y_pred = model.predict(test_x)
	test_y_cat = one_hot_to_categorical(test_y)
	y_pred_cat = one_hot_to_categorical(y_pred)
	acc = accuracy_score(test_y_cat, y_pred_cat)

	#clean memory???
	train_x, train_y, test_x, test_y, model = None, None, None, None, None
	gc.collect()

	#return the accuracy
	#print("data with shape:", train_x.shape, train_y.shape, 'train=', train_file, 'test=', test_file, 'with fraction', percent_dataset, 'had acc', acc)
	return acc

###############################
############ main #############
###############################

if __name__ == "__main__":

	#for each method
	for a_method in a_methods:

		writer = open('outputs_f1/' + a_method + '_' + get_now_str() + '.txt', 'w')

		#for each size dataset
		for size_folder in size_folders:

			writer.write(size_folder + '\n')

			#get all six datasets
			dataset_folders = [size_folder + '/' + s for s in datasets]

			#for storing the performances
			performances = {alpha:[] for alpha in alphas}

			#for each dataset
			for i in range(len(dataset_folders)):

				#initialize all the variables
				dataset_folder = dataset_folders[i]
				dataset = datasets[i]
				num_classes = num_classes_list[i]
				input_size = input_size_list[i]
				word2vec_pickle = dataset_folder + '/word2vec.p'
				word2vec = load_pickle(word2vec_pickle)

				#test each alpha value
				for alpha in alphas:

					train_path = dataset_folder + '/train_' + a_method + '_' + str(alpha) + '.txt'
					test_path = 'size_data_f1/test/' + dataset + '/test.txt'
					acc = run_cnn(train_path, test_path, num_classes, percent_dataset=1)
					performances[alpha].append(acc)

			writer.write(str(performances) + '\n')
			for alpha in performances:
				line = str(alpha) + ' : ' + str(sum(performances[alpha])/len(performances[alpha]))
				writer.write(line + '\n')
				print(line)
			print(performances)

		writer.close()
