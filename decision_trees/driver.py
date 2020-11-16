"""#####################################################################################################################
Entry point into your program. You will be classifying the mushroom data set and the zoo data set, both of which are
provided to you. Use the provided cross validation class to conduct two 10-fold crossvalidation decision tree
experiments. One should be conducted without pruning and the other with pruned trees at a p-value of 0.05.
Two miniscule datasets that correspond to things we did in class are available as well: restaurant, and tiny_animal_set.
You do not need to test these in your program, but you may find them very useful for debugging as you know the correct
behavior of many of the operations you need to implement as we discussed these in class.

The driver should print the mean error and standard deviation of the zoo and mushroom datasets using both unpruned
and pruned decision trees. In addition, you should print out one unpruned decision tree and one pruned decision tree
for each class. Call method chi_annotate on the tree before you print it so that you can see the χχ 2 statistic
for each decision node.
#####################################################################################################################"""

"""
Machine learning
decision trees
"""
import time


from ml_lib.ml_util import DataSet

from decision_tree import  DecisionTreeLearner

from ml_lib.crossval import cross_validation

from statistics import mean, stdev
    


    
def main():
    """
    Machine learning with decision trees.
    Runs cross validation on data sets and reports results/trees
    """
    raise NotImplementedError

if __name__ == '__main__':
    main()