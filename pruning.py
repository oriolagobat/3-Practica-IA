"""
File to implement the pruning method.
"""
import sys
from collections import Counter

from decision_node import DecisionNode


def prune(tree: DecisionNode, threshold: float):
    """
    Makes the pruning given a certain tree and threshold.
    """
    if _non_leaf(tree.true_branch):
        prune(tree.true_branch, threshold)
    if _non_leaf(tree.false_branch):
        prune(tree.false_branch, threshold)
    elif _both_children_leaf(tree):
        if tree.goodness < threshold:
            _merge_leaves(tree)


def _both_children_leaf(tree: DecisionNode):
    """
    Returns true if both of this node children are leaves (Non-None results)
    """
    return tree.true_branch.results is not None and tree.false_branch.results is not None


def _non_leaf(tree: DecisionNode):
    return tree.results is None


def _merge_leaves(tree: DecisionNode):
    # tree = DecisionNode(results=tree.tb.results + tree.fb.results)
    tree.col = -1
    tree.value = None
    tree.results = tree.true_branch.results + tree.false_branch.results
    tree.true_branch = None
    tree.false_branch = None
    tree.goodness = 0


if __name__ == '__main__':
    sys.exit(-1)
