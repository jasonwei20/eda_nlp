import os

sizes = ['1_tiny', '2_small', '3_standard', '4_full']
datasets = ['sst2', 'cr', 'subj', 'trec', 'pc']

for size in sizes:
	for dataset in datasets:
		folder = 'size_data_t1/' + size + '/' + dataset
		if not os.path.exists(folder):
			os.makedirs(folder)

		origin = 'sized_datasets_f1/' + size + '/' + dataset + '/train_orig.txt'
		destination = 'size_data_t1/' + size + '/' + dataset + '/train_orig.txt'
		os.system('cp ' + origin + ' ' + destination)