lines = open('special_f4/trec/test_orig.txt', 'r').readlines()

label_to_lines = {x:[] for x in range(0, 6)}

for line in lines:
	label = int(line[0])
	label_to_lines[label].append(line)

for label in label_to_lines:
	print(label, len(label_to_lines[label]))