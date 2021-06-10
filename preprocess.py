with open('./data/ml-1m.txt', 'r') as f:
    train_list = []
    test_list = []
    label_list = []
    userID = '1'
    row = []
    for line in f.readlines():
        data = line.strip().split()
        if userID != data[0]:
            train_list.append(' '.join(row[:-1]))
            test_list.append(' '.join(row[:-1]) + " <mask>")
            label_list.append(row[-1])
            row = []
            userID = data[0]
        row.append(data[1])
    
    train_list.append(' '.join(row[:-1]))
    test_list.append(' '.join(row[:-1]) + " <mask>")
    label_list.append(row[-1])

with open('./data/ml-1m-seq.txt', 'w') as f:
    for row in train_list:
        f.write(row + '\n')

with open('./data/ml-1m-seq-test.txt', 'w') as f:
    for test in test_list:
        f.write(test + '\n')

with open('./data/ml-1m-seq-label.txt', 'w') as f:
    for label in label_list:
        f.write(label + '\n')
