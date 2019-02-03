from methods import *
from numpy.random import seed
from keras import backend as K
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
seed(0)

################################
#### get dense layer output ####
################################

#getting the x and y inputs in numpy array form from the text file
def train_x(train_txt, word2vec_len, input_size, word2vec):

	#read in lines
	train_lines = open(train_txt, 'r').readlines()
	num_lines = len(train_lines)

	x_matrix = np.zeros((num_lines, input_size, word2vec_len))

	#insert values
	for i, line in enumerate(train_lines):

		parts = line[:-1].split('\t')
		label = int(parts[0])
		sentence = parts[1]	

		#insert x
		words = sentence.split(' ')
		words = words[:x_matrix.shape[1]] #cut off if too long
		for j, word in enumerate(words):
			if word in word2vec:
				x_matrix[i, j, :] = word2vec[word]

	return x_matrix

def get_dense_output(model_checkpoint, file, num_classes):

	x = train_x(file, word2vec_len, input_size, word2vec)

	model = load_model(model_checkpoint)

	get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[4].output])
	layer_output = get_3rd_layer_output([x])[0]

	return layer_output

def get_tsne_labels(file):
	labels = []
	alphas = []
	lines = open(file, 'r').readlines()
	for i, line in enumerate(lines):
		parts = line[:-1].split('\t')
		_class = int(parts[0])
		alpha = i % 10
		labels.append(_class)
		alphas.append(alpha)
	return labels, alphas

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
		file = 'special_f4/' + dataset + '/test_short_aug.txt'
		num_classes = num_classes_list[i]
		word2vec_pickle = 'special_f4/' + dataset + '/word2vec.p'
		word2vec = load_pickle(word2vec_pickle)

		#do tsne
		layer_output = get_dense_output(model_checkpoint, file, num_classes)
		print(layer_output.shape)
		t = get_plot_vectors(layer_output)

		labels, alphas = get_tsne_labels(file)

		print(labels, alphas)

		writer = open("outputs_f4/new_tsne.txt", 'w')

		label_to_mark = {0:'x', 1:'o'}

		for i, label in enumerate(labels):
			alpha = alphas[i]
			line = str(t[i, 0]) + ' ' + str(t[i, 1]) + ' ' + str(label_to_mark[label]) + ' ' + str(alpha/10)
			writer.write(line + '\n')


