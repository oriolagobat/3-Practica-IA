import sys
from collections import Counter

from decisionNode import DecisionNode


def prune(tree: DecisionNode, threshold: float):
    if _non_leaf(tree.tb):
        prune(tree.tb, threshold)
    if _non_leaf(tree.fb):
        prune(tree.fb, threshold)
    elif _both_children_leaf(tree):
        if tree.goodness < threshold:
            _merge_leaves(tree)


def _both_children_leaf(tree: DecisionNode):
    """
    Returns true if both of this node children are leaves (Non-None results)
    """
    return tree.tb.results is not None and tree.fb.results is not None


def _non_leaf(tree: DecisionNode):
    return tree.results is None


def _merge_leaves(tree: DecisionNode):
    # tree = DecisionNode(results=tree.tb.results + tree.fb.results)
    tree.col = -1
    tree.value = None
    tree.results = _merge_results(tree)
    tree.tb = None
    tree.fb = None
    tree.goodness = 0


def _merge_results(tree: DecisionNode):
    new_results = Counter()
    merged = tree.tb.results + tree.fb.results
    new_label = merged.most_common()[0][0]
    new_count = sum(merged.values())
    new_results[new_label] = new_count
    return new_results


if __name__ == '__main__':
    sys.exit(-1)
