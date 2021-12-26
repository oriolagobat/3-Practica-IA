"""
These file will help by abstracting the DecisionNode class
"""


class DecisionNode:
    """
    Represents a DecisionNode, a tree, that compresses all the trees below him.
    """
    def __init__(self, col=-1, value=None, results=None,
                 true_branch=None, false_branch=None, goodness=0):
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
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.goodness = goodness
