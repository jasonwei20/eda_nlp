from methods import *
from numpy.random import seed
from keras import backend as K
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
seed(0)

################################
#### get dense layer output ####
################################

def get_dense_output(model_checkpoint, file, num_classes):

	x, y = get_x_y(file, num_classes, word2vec_len, input_size, word2vec, 1)

	model = load_model(model_checkpoint)

	get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[4].output])
	layer_output = get_3rd_layer_output([x])[0]

	return layer_output, np.argmax(y, axis=1)

def get_plot_vectors(layer_output):

	tsne = TSNE(n_components=2).fit_transform(layer_output)
	return tsne

def plot_tsne(tsne, labels, output_path):

	label_to_legend_label = {	'outputs_f4/pc_tsne.png':{	0:'Con (augmented)', 
															100:'Con (original)', 
															1: 'Pro (augmented)', 
															101:'Pro (original)'},
								'outputs_f4/trec_tsne.png':{0:'Description (augmented)',
															100:'Description (original)',
															1:'Entity (augmented)',
															101:'Entity (original)',
															2:'Abbreviation (augmented)',
															102:'Abbreviation (original)',
															3:'Human (augmented)',
															103:'Human (original)',
															4:'Location (augmented)',
															104:'Location (original)',
															5:'Number (augmented)',
															105:'Number (original)'}}

	plot_to_legend_size = {'outputs_f4/pc_tsne.png':11, 'outputs_f4/trec_tsne.png':6}

	labels = labels.tolist()
	big_groups = [label for label in labels if label < 100]
	big_groups = list(sorted(set(big_groups)))

	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff1493', '#FF4500']
	fig, ax = plt.subplots()

	for big_group in big_groups:

		for group in [big_group, big_group+100]:

			x, y = [], []

			for j, label in enumerate(labels):
				if label == group:
					x.append(tsne[j][0])
					y.append(tsne[j][1])

			#params
			color = colors[int(group % 100)]
			marker = 'x' if group < 100 else 'o'
			size = 1 if group < 100 else 27
			legend_label = label_to_legend_label[output_path][group]

			ax.scatter(x, y, color=color, marker=marker, s=size, label=legend_label)
			plt.axis('off')

	legend_size = plot_to_legend_size[output_path]
	plt.legend(prop={'size': legend_size})
	plt.savefig(output_path, dpi=1000)
	plt.clf()	

if __name__ == "__main__":

	#global variables
	word2vec_len = 300
	input_size = 25

	datasets = ['pc'] #['pc', 'trec']
	num_classes_list =[2] #[2, 6]

	for i, dataset in enumerate(datasets):

		#load parameters
		model_checkpoint = 'outputs_f4/' + dataset + '.h5'
		file = 'special_f4/' + dataset + '/test_short_aug_shuffle.txt'
		num_classes = num_classes_list[i]
		word2vec_pickle = 'special_f4/' + dataset + '/word2vec.p'
		word2vec = load_pickle(word2vec_pickle)

		#do tsne
		layer_output, labels = get_dense_output(model_checkpoint, file, num_classes)
		print(layer_output.shape)
		t = get_plot_vectors(layer_output)

		#edit labels:
		for i in range(len(labels)):
			
			#mark original, unaugmented data
			if i % 10 == 9:
				labels[i] += 100


		output_path = 'outputs_f4/' + dataset + '_tsne.png'
		plot_tsne(t, labels, output_path)

		label_names = labels.tolist()
		label_writers = {}
		for label_name in label_names:
			label_writers[label_name] = open('outputs_f4/' + str(label_name) + '.txt', 'w')

		for i, label in enumerate(labels):
			line = str(t[i, 0]) + ' ' + str(t[i, 1])
			print(line)
			label_writers[label].write(line + '\n')


