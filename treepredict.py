#!/usr/bin/env python3
"""
File that implements the construction of a decision tree and runs the main tasks of the project
"""
from math import log
from typing import List, Tuple
from collections import Counter
import sys

import pruning
import evaluation
from util import Stack
from decision_node import DecisionNode

# Used for typing
Data = List[List]


def _parse_value(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read(file_name: str, separator: str = ",") -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    headers = None
    data = []
    with open(file_name) as file_:
        for line in file_:
            values = line.strip().split(separator)
            if not headers:
                headers = values
            else:
                data.append([_parse_value(v) for v in values])

    return headers, data


def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """
    results = Counter()
    for row in part:
        label = row[-1]
        results[label] += 1
    return results


def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0

    results = unique_counts(part)
    imp = 1
    for count in results.values():
        prob = count / total
        imp -= prob ** 2
    return imp


def _log2(value: float):
    return log(value) / log(2)


def entropy(rows: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    total = len(rows)
    results = unique_counts(rows)

    imp = 0
    for count in results.values():
        prob = count / total
        imp -= prob * _log2(prob)
    return imp


def _split_numeric(prototype: List, column: int, value: int):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value: int):
    return prototype[column] == value


def divideset(part: Data, column: int, value: int) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    set1 = []
    set2 = []

    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical
    for row in part:
        set1.append(row) if split_function(row, column, value) else set2.append(row)

    return set1, set2


def _gain(part: Data, set1: Data, set2: Data, scoref):
    prob1 = len(set1) / len(part)

    prob2 = len(set2) / len(part)
    return scoref(part) - prob1 * scoref(set1) - prob2 * scoref(set2)


def buildtree(part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s(Delta i(s,t) < beta)
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)

    if current_score == 0:
        # The partition is pure
        return DecisionNode(results=unique_counts(part), goodness=0)

    best_gain, best_criteria, best_sets = _search_best_params(part, scoref)

    if best_gain < beta:
        return DecisionNode(results=unique_counts(part), goodness=best_gain)

    return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                        true_branch=buildtree(best_sets[0]),
                        false_branch=buildtree(best_sets[1]), goodness=best_gain)


def _search_best_params(part: Data, scoref=entropy):
    # Set up some variables to track the best criteria
    best_gain = 0
    best_criteria = None
    best_sets = None

    n_cols = len(part[0]) - 1  # Skip the label

    for i in range(n_cols):
        possibles_cut_values = set()
        for row in part:
            possibles_cut_values.add(row[i])

        for value in possibles_cut_values:
            set1, set2 = divideset(part, i, value)
            gain = _gain(part, set1, set2, scoref)
            if gain > best_gain:
                best_gain = gain
                best_criteria = (i, value)
                best_sets = set1, set2

    return best_gain, best_criteria, best_sets


def iterative_buildtree(part: Data, scoref=entropy, beta=0):
    """
    t10: Define the iterative version of the function buildtree
    """

    if len(part) == 0:
        return DecisionNode(results=unique_counts(part), goodness=0)

    stack = Stack()
    node_stack = Stack()
    stack.push((0, part, None, 0))

    while not stack.is_empty():
        context, data, criteria, goodness = stack.pop()
        if context == 0:  # No built sub-trees
            current_score = scoref(data)
            if current_score == 0:
                node_stack.push(DecisionNode(results=unique_counts(data), goodness=goodness))

            else:
                best_gain, best_criteria, best_sets, = _search_best_params(data, scoref)
                if best_gain < beta:
                    node_stack.push(DecisionNode(results=unique_counts(data)))
                else:
                    stack.push((1, data, best_criteria, best_gain))
                    stack.push((0, best_sets[0], best_criteria, best_gain))
                    stack.push((0, best_sets[1], best_criteria, best_gain))

        elif context == 1:
            true_branch = node_stack.pop()  # True tree
            false_branch = node_stack.pop()  # False tree
            node_stack.push(DecisionNode(col=criteria[0], value=criteria[1],
                                         true_branch=true_branch, false_branch=false_branch, goodness=goodness))

            if len(data) == len(part):  # If root node
                return node_stack.pop()


def classify(tree: DecisionNode, row):
    """
    Returns the prediction for a certain row given a tree.
    """

    # RECURSIVE WAY

    # if tree.results is not None:
    #     return tree.results.most_common()[0][0]
    # return classify(tree.true_branch, row) if _classify_function(tree, row) else classify(tree.false_branch, row)

    # ITERATIVE WAY

    while tree.results is None:
        tree = tree.true_branch if _classify_function(tree, row) else tree.false_branch
    return tree.results.most_common()[0][0]


def _classify_function(tree: DecisionNode, row):
    if isinstance(tree.value, (int, float)):
        return _split_numeric(row, tree.col, tree.value)
    return _split_categorical(row, tree.col, tree.value)


def print_tree(tree, headers=None, indent=""):
    """
    t11: Include the following function
    """
    # Is this a leaf node?
    if tree.results is not None:
        print(indent, tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.true_branch, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.false_branch, headers, indent + "  ")


def print_data(headers, data):
    """
    Prints tree data.
    """
    colsize = 15
    print('-' * ((colsize + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(colsize), end="|")
    print("")
    print('-' * ((colsize + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(colsize), end="|")
            else:
                print(value.ljust(colsize), end="|")
        print("")
    print('-' * ((colsize + 1) * len(headers) + 1))


def main():
    """
    Executes the main course of the program
    Comment any call to any task you don't want to execute
    """
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"

    headers, data = read(filename)

    """ APARTAT 1 """
    _build_tree_iterative(data, headers)

    """ APARTAT 2, we get the rows to predict by dividing dataset into train and test """
    _function_classify(data)

    """ APARTAT 3 """
    _tree_pruning(data, headers)

    """ APARTAT 4 """
    _test_performance(data)

    """ APARTAT 5 """
    _cross_validation(data)

    """ APARTAT 6, run with iris.csv """
    _find_optimal_threshold(data)


""" SCHOOL TASKS, IGNORE AND RUN FROM MAIN """


def _build_tree_iterative(data, headers):
    print("******* APARTAT 1 ******")
    tree = buildtree(data)
    print("**** RECURSIVE ****")
    print_tree(tree, headers)
    print("\n\n")
    print("**** ITERATIVE ****")
    it_tree = iterative_buildtree(data)
    print_tree(it_tree, headers)


def _function_classify(data):
    print("\n\n******* APARTAT 2 ******")
    train, test = evaluation.train_test_split(data, 0.2)
    tree = buildtree(train)
    for row in test:
        prediction = classify(tree, row[:-1])
        print("Prediction for row: ", row, "is label", prediction)


def _tree_pruning(data, headers):
    print("\n\n******* APARTAT 3 ******")
    tree = buildtree(data)
    print("**** NO PRUNING ****")
    print_tree(tree, headers)
    pruning.prune(tree, 0.85)
    print("\n\n")
    print("**** PRUNING ****")
    print_tree(tree, headers)


def _test_performance(data):
    print("\n\n******* APARTAT 4 ******")
    train, test = evaluation.train_test_split(data, 0.2)
    tree = buildtree(train)
    print("Data split between train and test with 0.2 test size")
    train_accuracy = evaluation.get_accuracy(tree, train)
    print("Accuracy with training data is " + "{:.2f}".format(train_accuracy * 100) + " %")
    test_accuracy = evaluation.get_accuracy(tree, test)
    print("Accuracy with testing data is " + "{:.2f}".format(test_accuracy * 100) + " %")


def _cross_validation(data):
    print("\n\n******* APARTAT 5 ******")
    score = evaluation.cross_validation(dataset=data, k=5)
    print("Score obtained with 5 folds, no beta and no threshold is: " + "{:.2f}".format(score))


def _find_optimal_threshold(data):
    print("\n\n******* APARTAT 6 ******")
    best_threshold = (None, -1)
    train, test = evaluation.train_test_split(data, 0.2)

    """ Change if you want to try to find the optimal threshold with more iterations"""
    """ There will always be one more iteration than the one detailed here"""
    iterations_minus_one = 10

    segment_division = 1 / iterations_minus_one
    for i in range(iterations_minus_one + 1):
        threshold = segment_division * i
        result = evaluation.cross_validation(dataset=train, k=5, threshold=threshold)
        if result > best_threshold[1]:
            best_threshold = (threshold, result)

    print(
        "Best threshold found is: " + str(best_threshold[0]) + ", with an accuracy of " + "{:.2f}".format(
            best_threshold[1]))

    best_threshold_model = buildtree(train)

    not_prunned_accuracy = evaluation.get_accuracy(best_threshold_model, test)
    print(
        "Accuracy found with test dataset on tree trained with training dataset and not prunned yet is: " + "{:.2f}".format(
            not_prunned_accuracy * 100) + " %")

    pruning.prune(best_threshold_model, best_threshold[0])
    best_threshold_accuracy = evaluation.get_accuracy(best_threshold_model, test)
    print(
        "Accuracy found with test dataset, "
        "on tree trained with training dataset and pruned with the best threshold "
        "found is: "
        + "{:.2f}".format(best_threshold_accuracy * 100) + " %")

if __name__ == "__main__":
    main()
