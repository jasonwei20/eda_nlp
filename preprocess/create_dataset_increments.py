import os

datasets = ['cr', 'pc', 'sst1', 'sst2', 'subj', 'trec']

for dataset in datasets:
	line = 'cat increment_datasets_f2/' + dataset + '/test.txt > sized_datasets_f1/test/' + dataset + '/test.txt'
	os.system(line)