import numpy as np


def load_dataset(name='wine.data'):
    with open(name, 'r') as f:
        dataset = [line.rstrip() for line in f.readlines()]

    return dataset


def parse_dataset():
    dataset = load_dataset()
    test_dataset = np.random.choice(dataset, int(0.15 * len(dataset)), replace=False)
    train_dataset = list(set(dataset) - set(test_dataset))
    X_train = [[float(x) for x in data.split(",")][1:] for data in train_dataset]
    X_test = [[float(x) for x in data.split(",")][1:] for data in test_dataset]

    Y_train = [int(x.split(',')[0]) - 1 for x in train_dataset]
    Y_test = [int(x.split(',')[0]) - 1 for x in test_dataset]

    return X_train, Y_train, X_test, Y_test