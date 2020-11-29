"""#####################################################################################################################
Contains skeleton code for a DecisionTree class. The constructor expects a dataset to be learned.
Some of the methods are written for you. Do not change the names of any of the predefined variables or interfaces,
they will be examined in the unit test code and can cause your execution to fail.
The class contains methods to predict the class of examples, estimation entropy, estimate χ 2 statistics,
etc. Read through the code to determine what is provided and what you need to write.
The bulk of your assignment is modifying this module.
#####################################################################################################################"""
import math
from collections import namedtuple

import numpy as np
import scipy.stats

from ml_lib.utils import scalar_vector_product
from ml_lib.ml_util import argmax_random_tie, normalize, remove_all, best_index
from ml_lib.decision_tree_support import DecisionLeaf, DecisionFork
import ml_lib.utils


class DecisionTreeLearner:
    """DecisionTreeLearner - Class to learn decision trees and predict classes
    on novel exmaples
    """

    # Typedef for method chi2test result value (see chi2test for details)
    chi2_result = namedtuple("chi2_result", ('value', 'similar'))

    def __init__(self, dataset, debug=False, p_value=None):
        """
        DecisionTreeLearner(dataset)
        dataset is an instance of ml_lib.ml_util.DataSet.
        """

        # Hints: Be sure to read and understand the DataSet class
        # as you will use it throughout.

        # ---------------------------------------------------------------
        # Do not modify these lines, the unit tests will expect these fields
        # to be populated correctly.
        self.dataset = dataset

        # degrees of freedom for Chi^2 tests is number of categories minus 1
        self.dof = len(self.dataset.values[self.dataset.target]) - 1

        # Learn the decison tree
        self.tree = self.decision_tree_learning(dataset.examples, dataset.inputs)
        # -----------------------------------------------------------------

        self.debug = debug

    def __str__(self):
        "str - Create a string representation of the tree"
        if self.tree is None:
            result = "untrained decision tree"
        else:
            result = str(self.tree)  # string representation of tree
        return result

    def decision_tree_learning(self, examples, attrs, parent=None, parent_examples=()):
        """
        decision_tree_learning(examples, attrs, parent_examples)
        Recursively learn a decision tree
        examples - Set of examples (see DataSet for format)
        attrs - List of attribute indices that are available for decisions
        parent - When called recursively, this is the parent of any node that
           we create.
        parent_examples - When not invoked as root, these are the examples
           of the prior level.
        """

        # Hints:  See pseudocode from class and leverage classes
        # DecisionFork and DecisionLeaf
        if len(examples) == 0:
            # If there are no more examples create leaf with the most popular target based on parent examples
            popular_target = self.plurality_value(parent_examples)
            leaf = DecisionLeaf(popular_target, self.count_targets(examples), parent)  # Create leaf
            return leaf
        elif self.all_same_class(examples):
            # If all examples are from the same class then we are done, hence we will return the result in a leaf
            target = self.dataset.target  # Get target index
            # Returns the class from the first example since all the remaining examples have the same class
            result = examples[0][target]
            leaf = DecisionLeaf(result, self.count_targets(examples), parent)  # Create leaf
            return leaf
        elif len(attrs) == 0:
            # If there are no more questions to ask then pick the most popular target based on the examples passed
            popular_target = self.plurality_value(examples)
            leaf = DecisionLeaf(popular_target, self.count_targets(examples), parent)  # Create leaf
            return leaf
        else:

            a = self.choose_attribute(attrs, examples)  # Choose the most important attribute based on info gained
            # Create new tree rooted on the most important question such as: if a..?
            t = DecisionFork(a, self.count_targets(examples), self.dataset.attr_names[a], parent=parent)

            # Get values associated with attribute a and its examples
            valuesWithAList = self.split_by(a, examples)  # e.g. [(val_1,listOfExamplesWithVal_1),...]
            for tupleValue in valuesWithAList:  # For each value and its examples associated with attribute a
                v, vexamples = tupleValue  # value, value_examples
                subtree = self.decision_tree_learning(vexamples, np.setdiff1d(attrs, [a]), t, examples)
                t.add(v, subtree)  # Add a subtree to our tree with v as branch level
            return t  # Return the current tree

    def plurality_value(self, examples):
        """
        Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality).
        """
        popular = argmax_random_tie(self.dataset.values[self.dataset.target],
                                    key=lambda v: self.count(self.dataset.target, v, examples))
        return popular

    def count(self, attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def count_targets(self, examples):
        """count_targets: Given a set of examples, count the number of examples
        belonging to each target.  Returns list of counts in the same order
        as the DataSet values associated with the target
        (self.dataset.values[self.dataset.target])
        """

        tidx = self.dataset.target  # index of target attribute
        target_values = self.dataset.values[tidx]  # Class labels across dataset

        # Count the examples associated with each target
        counts = [0 for i in target_values]
        for e in examples:
            target = e[tidx]
            position = target_values.index(target)
            counts[position] += 1

        return counts

    def all_same_class(self, examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][self.dataset.target]
        return all(e[self.dataset.target] == class0 for e in examples)

    def choose_attribute(self, attrs, examples):
        """Choose the attribute with the highest information gain."""
        maxAttr = attrs[0]
        maxGain = -math.inf

        for attr in attrs:
            attrGain = self.information_gain(attr, examples)
            if attrGain > maxGain:
                maxGain = attrGain
                maxAttr = attr
        return maxAttr
        # Returns the attribute index
        # raise NotImplementedError

    def entropy(self, examples):
        return self.information_content([self.count(self.dataset.target, v, examples) for v in self.dataset.values[self.dataset.target]])

    def information_gain(self, attr, examples):
        """Return the expected reduction in entropy for examples from splitting by attr."""
        # information gain is Gain = entropy - remainder
        # only use information_per_class used in information_gain
        # use split_by in remainder
        # use information_per_class, split_by, information_content
        remainder = 0.0
        total = float(len(examples))
        valueswithAList = self.split_by(attr, examples)
        for tupleValue in valueswithAList:
            v, vexamples = tupleValue
            v_entropy = self.entropy(vexamples)
            remainderSum = (len(vexamples) / total) * v_entropy
            remainder = remainder + remainderSum
        gain = self.entropy(examples) - remainder
        return gain
        # raise NotImplementedError

    def split_by(self, attr, examples):
        """split_by(attr, examples)
        Return a list of (val, examples) pairs for each val of attr.
        """
        return [(v, [e for e in examples if e[attr] == v]) for v in self.dataset.values[attr]]

    def predict(self, x):
        "predict - Determine the class, returns class index"
        return self.tree(x)  # Evaluate the tree on example x

    def __repr__(self):
        return repr(self.tree)

    @classmethod
    def information_content(cls, class_counts):
        """Given an iterable of counts associated with classes
        compute the empirical entropy.

        Example:  3 class problem where we have 3 examples of class 0,
        2 examples of class 1, and 0 examples of class 2:
        information_content((3, 2, 0)) returns ~ .971
        """
        # information_content says to compute entropy [H(x)]
        # normalize class_counts to get ratios/fractions for propability then use entropy equation to find summation

        entropy = 0.0
        class_probability = normalize(remove_all(0, class_counts)) #np.setdiff1d(class_counts, [0]))
        for ratio in class_probability:  # looping through the probabilities from normalizing
            entropySum = -1 * ratio * np.log2(ratio)
            entropy = entropy + entropySum
        return entropy

        # returning entropy [H(x)] as a float but maybe it should be a list because remainder needs to get every individual entropies

        # Hints:
        #  Remember discrete values use log2 when computing probability
        #  Function normalize might be helpful here...
        #  Python treats logarithms of 0 as a domain error, whereas numpy
        #    will compute them correctly.  Be careful.

        # raise NotImplementedError

    def information_per_class(self, examples):
        """information_per_class(examples)
        Given a set of examples, use the target attribute of the dataset
        to determine the information associated with each target class
        Returns information content per class.
        """
        # Hint:  list of classes can be obtained from
        # self.data.set.values[self.dataset.target]
        # targetClasses = self.dataset.values[self.dataset.target]
        # only use information_per_class to get the list of target classes count

        class_count = self.count_targets(examples)
        return class_count

    ########################################################################################################################
    def prune(self, p_value):
        """Prune leaves of a tree when the hypothesis that the distribution
        in the leaves is not the same as in the parents as measured by
        a chi-squared test with a significance of the specified p-value.

        Pruning is only applied to the last DecisionFork in a tree.
        If that fork is merged (DecisionFork and child leaves (DecisionLeaf),
        the DecisionFork is replaced with a DecisionLeaf.  If a parent of
        and DecisionFork only contains DecisionLeaf children, after
        pruning, it is examined for pruning as well.
        """
        self.prune_aux(p_value, self.tree, None)
        # Hint - Easiest to do with a recursive auxiliary function, that takes
        # a parent argument, but you are free to implement as you see fit.
        # e.g. self.prune_aux(p_value, self.tree, None)
        return 1.0
        # raise NotImplementedError

    ########################################################################################################################
    def prune_aux(self, p_value, branch, parent):
        branches = []
        if isinstance(branch, DecisionLeaf):
            return
        for child in branch.branches.values():
            if isinstance(child, DecisionFork):
                branches.append(child)
        if len(branches) == 0:
            chi_test_tuple = self.chi2test(p_value, branch)
            similarity = chi_test_tuple.similar
            #print(similarity)
            if similarity == True:
                #prune
                branch_best_index = best_index(branch.distribution)
                update_key = 0
                result = self.dataset.values[self.dataset.target][branch_best_index]
                new_leaf = DecisionLeaf(result, branch.distribution, branch)
                for tupleValue in branch.branches.items():
                    key, name = tupleValue
                    if name.result == new_leaf.result:
                        update_key = key
                #print("I am pruned: ", new_leaf)
                #print("Previous Tree: ", parent)
                #print("\n")
                #print("{0} has been pruned", branch)
                #print("\n")
                parent.branches.update({update_key: new_leaf})
                #print("Pruning Happened")
                #print("New tree: ", parent)
                #print("\n")
                #print("\n")

        else:
            for b in branches:
                self.prune_aux(p_value, b, branch)


    def chi_annotate(self, p_value):
        """chi_annotate(p_value)
        Annotate each DecisionFork with the tuple returned by chi2test
        in attribute chi2.  When present, these values will be printed along
        with the tree.  Calling this on an unpruned tree can significantly aid
        with developing pruning routines and verifying that the chi^2 statistic
        is being correctly computed.
        """
        # Call recursive helper function
        self.__chi_annotate_aux(self.tree, p_value)

    def __chi_annotate_aux(self, branch, p_value):
        """chi_annotate(branch, p_value)
        Add the chi squared value to a DecisionFork.  This is only used
        for debugging.  The decision tree helper functions will look for a
        chi2 attribute.  If there is one, they will display chi-squared
        test information when the tree is printed.
        """

        if isinstance(branch, DecisionLeaf):
            return  # base case
        else:
            # Compute chi^2 value of this branch
            branch.chi2 = self.chi2test(p_value, branch)
            # Check its children
            for child in branch.branches.values():
                self.__chi_annotate_aux(child, p_value)

    def chi2test(self, p_value, fork):
        """chi2test - Helper function for prune
        Given a DecisionFork and a p_value, determine if the children
        of the decision have significantly different distributions than
        the parent.

        Returns named tuple of type chi2result:
        chi2result.value - Chi^2 statistic
        chi2result.similar - True if the distribution in the children of the
           specified fork are similar to the the distribution before the
           question is asked.  False indicates that they are not similar and
           that there is a significant difference between the fork and its
           children
        """
        """ Define an acceptable level of error called a p-value
            look up threshold from chi-squared inverse cdf at 1 - p-value
            computer change for each leaf"""

        if not isinstance(fork, DecisionFork):
            raise ValueError("fork is not a DecisionFork")

        # look up threshold from chi-squared inverse cdf at 1 - p-value
        threshold_inverse_cdf = scipy.stats.chi2.ppf(1 - p_value, self.dof)
        # Hint:  You need to extend the 2 case chi^2 test that we covered
        # in class to an n-case chi^2 test.  This part is straight forward.
        # Whereas in class we had positive and negative samples, now there
        # are more than two, but they are all handled similarly.
        # Don't forget, scipy has an inverse cdf for chi^2
        # scipy.stats.chi2.ppf

        delta = 0.0
        p_list = fork.distribution                      # Get p
        p_plus_n = sum(p_list)                           # Get p + n
        n_list = np.subtract(p_plus_n, p_list)           # Get n
        size = len(p_list)
        child_nodes = fork.branches
        p_k_list = []
        n_k_list = []
        pk_plus_nk = []                             # For p_k + n_k
        # Get p_k and n_k
        for child in child_nodes.values():
            dist = child.distribution                   # Get p_k
            total = sum(dist)                           # Get items in split
            pk_plus_nk.append(total)                    # Get p_k + n_k
            p_k_list.append(dist)
            n_k_list.append(np.subtract(total, dist))   # Get n_k

        num_of_children = len(p_k_list)
        # Get delta
        delta = 0.0
        for index in range(0, size):                                # For each distribution value
            p = p_list[index]                                       # a value in the distribution
            # For each child node
            for k in range(0, num_of_children):
                p_k = p_k_list[k]                                   # Get p_k value
                fraction = pk_plus_nk[k] / p_plus_n                 # (p_k + n_k) / ( p + n)
                p_hat = p * fraction                                # p * (p_k + n_k) / ( p + n)
                p_k_dev = 0                                         # Get p_k deviation
                if p_hat != 0:
                    p_k_dev = ((p_k[index] - p_hat) ** 2) / p_hat   # ((p_k - p_hat)^2)/p_hat
                delta += p_k_dev                                    # Increment delta

        # Set our delta value in the chi2 result
        self.chi2_result.value = delta
        # Determine if leaves are irrelevant
        similar = False
        if self.chi2_result.value < threshold_inverse_cdf:
            similar = True
        self.chi2_result.similar = similar

        return self.chi2_result

    def neg_dist(self, list_dist):
        size = sum(list_dist)
        return [size - x for x in list_dist]

    def __str__(self):
        """str - String representation of the tree"""
        return str(self.tree)
