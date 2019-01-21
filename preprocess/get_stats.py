import statistics

datasets = ['sst2', 'cr', 'subj', 'trec', 'pc']

filenames = ['increment_datasets_f2/' + x + '/train_orig.txt' for x in datasets]

def get_vocab_size(filename):
	lines = open(filename, 'r').readlines()

	vocab = set()
	for line in lines:
		words = line[:-1].split(' ')
		for word in words:
			if word not in vocab:
				vocab.add(word)

	return len(vocab)

def get_mean_and_std(filename):
	lines = open(filename, 'r').readlines()

	line_lengths = []
	for line in lines:
		length = len(line[:-1].split(' ')) - 1
		line_lengths.append(length)

	print(filename, statistics.mean(line_lengths), statistics.stdev(line_lengths), max(line_lengths))


for filename in filenames:
	#print(get_vocab_size(filename))
	get_mean_and_std(filename)







