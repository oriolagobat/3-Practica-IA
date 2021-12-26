"""
Implements the methods that evaluate a decision tree, rather than building it
"""
import random
from typing import Union, List

import pruning
import treepredict
from decision_node import DecisionNode


def train_test_split(dataset, test_size: Union[float, int], seed=None):
    """
    Given a dataset, splits these data randomly into train and test,
     making the test ones test_size % of the total dataset.
    """
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test


def get_accuracy(tree: DecisionNode, dataset):
    """
    Given a certain tree,
    returns the accuracy of this tree to predict the label of the data in dataset
    """
    correct = 0
    for row in dataset:
        if treepredict.classify(tree, row[:-1]) == row[-1]:
            correct += 1
    return correct / len(dataset)


def mean(values: List[float]):
    """
    Given a list of values, returns the mean of the values
    """
    return sum(values) / len(values)


def cross_validation(dataset=treepredict.Data, k=1, agg=mean,
                     seed=None, scoref=treepredict.entropy, beta=0, threshold=0):
    """
    Makes the cross_validation process, given a certain dataset and number of folds.
    """
    if seed:
        random.seed(seed)
    _randomize_dataset(dataset)
    partitions = _make_partitions(dataset, k)
    scores = []
    for i in range(k):
        train, test = _get_train_test(partitions, i)
        model = treepredict.buildtree(train, scoref, beta)
        if threshold == get_accuracy:
            threshold = get_accuracy(model, train)
        pruning.prune(model, threshold)
        fold_score = get_accuracy(model, test)
        scores += [fold_score]
    final_score = agg(scores)
    return final_score


def _randomize_dataset(dataset):
    """
    Randomizes a list
    """
    random.shuffle(dataset)


def _make_partitions(dataset, folds):
    """
    Given a certain dataset, returns a list with folds number of lists,
    making even partitions of the data
    """
    partitions = []
    partition_size = int(len(dataset) / folds)
    # print(str(len(dataset)) + " Elements in this dataset")
    # print(str(folds) + " Folds for this dataset")
    # print("Folds will be of size " + str(partition_size) )
    for i in range(folds):
        if i != folds - 1:  # Not the final partition
            sub_list = dataset[i * partition_size:i * partition_size + partition_size]
            partitions += [sub_list]
        else:  # Final partition, append all the remaining data
            sub_list = dataset[i * partition_size:]
            partitions += [sub_list]
    return partitions


def _get_train_test(partitions, current_index):
    """
    Given a list of partitions and an index, returns two lists:
    The train list will have all of the partitions except the one with current_index
    The test list will only have the current_index partition
    """
    train = []
    test = []
    for i, partition in enumerate(partitions):
        if i != current_index:
            train += partition
        else:
            test += partition
    return train, test
