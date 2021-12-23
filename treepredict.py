#!/usr/bin/env python3
from math import log
from typing import List, Tuple
from collections import Counter
import sys

from util import Stack

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
    data = list()
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
        p = count / total
        imp -= p ** 2
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
        p = count / total
        imp -= p * _log2(p)
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


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        t8: We have 5 member variables:
        - col is the column index which represents the
          attribute we use to split the node
        - value corresponds to the answer that satisfies
          the question
        - tb and fb are internal nodes representing the
          positive and negative answers, respectively
        - results is a dictionary that stores the result
          for this branch. Is None except for the leaves
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def _gain(part: Data, set1: Data, set2: Data, scoref):
    p1 = len(set1) / len(part)
    p2 = len(set2) / len(part)
    return scoref(part) - p1 * scoref(set1) - p2 * scoref(set2)


def buildtree(part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)

    if current_score == 0:
        # The partition is pure
        return DecisionNode(results=unique_counts(part))

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

    if best_gain < beta:
        return DecisionNode(results=unique_counts(part))

    return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                        tb=buildtree(best_sets[0]), fb=buildtree(best_sets[1]))


def iterative_buildtree(part: Data, scoref=entropy, beta=0):
    """
    t10: Define the iterative version of the function buildtree
    """
    stack = Stack


def classify(tree, values):
    raise NotImplementedError


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
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")


def print_data(headers, data):
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
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"

    headers, data = read(filename)
    tree = buildtree(data)
    print_tree(tree, headers)


if __name__ == "__main__":
    main()
