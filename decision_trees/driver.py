"""#####################################################################################################################
Pair Programming Equitable Participation & Honesty Affidavit
We the undersigned promise that we have in good faith attempted to follow the principles of pair programming.
Although we were free to discuss ideas with others, the implementation is our own.
We have shared a common workspace and taken turns at the keyboard for the majority of the work that we are submitting.
Furthermore, any non programming portions of the assignment were done independently.
We recognize that should this not be the case, we will be subject to penalties as outlined in the course syllabus.

Pair Programmer 1 (print & sign your name, then date it)
Lilian Vu 11/1/2020

Pair Programmer 2 (print & sign your name, then date it)
Amiel Nava 11/1/2020
#####################################################################################################################"""

"""
Machine learning
decision trees
"""
import time

from ml_lib.ml_util import DataSet

from decision_tree import DecisionTreeLearner

from ml_lib.crossval import cross_validation
import numpy as np

from ml_lib.ml_util import mean_error

from statistics import mean, stdev


def main():
    """
    Machine learning with decision trees.
    Runs cross validation on data sets and reports results/trees
    """
    # Constant Divider string when we print results
    DIVIDER = '-------------------------------------------------------------------------------------------------------------'

    # Create data sets
    mushroomData = DataSet(attr_names=True, target=0, name='mushrooms')
    zooData = DataSet(attr_names=True, name='zoo', exclude=[0])

    # Call cross validation with the DecisionTreeLearner class for 10 folds and 2 trials
    # Unpruned results
    mushroomResults = cross_validation(DecisionTreeLearner, mushroomData)
    zooResults = cross_validation(DecisionTreeLearner, zooData)
    # Pruned results
    mushroomResults_p = cross_validation(DecisionTreeLearner, mushroomData, debug=True, p_value=0.05)
    zooResults_p = cross_validation(DecisionTreeLearner, zooData, p_value=0.05)

    """Get Mushroom Results"""
    # Unpruned mushroom
    mushroom_err = mushroomResults[0]  # Get error lists from the mushroom training sets
    mushroom_model = mushroomResults[1]  # Get the list of decision trees from the mushroom training sets
    # Pruned mushroom
    mushroom_err_p = mushroomResults_p[0]  # Get error lists from the pruned mushroom training sets
    mushroom_model_p = mushroomResults_p[1]  # Get the list of decision trees from the pruned mushroom training sets

    """Get Zoo Results"""
    # Unpruned zoo
    zoo_err = zooResults[0]  # Get error lists from the zoo training sets
    zoo_model = zooResults[1]  # Get the list of decision trees from the zoo training sets
    # Pruned zoo
    zoo_err_p = zooResults_p[0]  # Get error lists from the pruned zoo training sets
    zoo_model_p = zooResults_p[1]  # Get the list of decision trees from the pruned zoo training sets

    """Print Mushroom statistics"""
    print(DIVIDER)
    print("MUSHROOM")
    print(DIVIDER)
    print('Mushroom err:', mushroom_err)
    print('Mushroom Model:', mushroom_model)
    print('Mushroom with pruning err:', mushroom_err_p)
    print('Mushroom with pruning Model:', mushroom_model_p)

    # Mean error and Standard Deviation of Mushroom Data Without Pruning
    mushroom_mean_error = np.mean(mushroom_err, 0)
    mushroom_std_error = np.std(mushroom_err, 0)
    print("Mushroom Mean Error Without Pruning: ", mushroom_mean_error)
    print("Mushroom STD Error Without Pruning: ", mushroom_std_error)

    print(DIVIDER)
    print("I am pruned")
    print(DIVIDER)
    # Mean error and Standard Deviation of Mushroom Data With Pruning
    mushroom_mean_error_prune = np.mean(mushroom_err_p, 0)
    mushroom_std_error_prune = np.std(mushroom_err_p, 0)
    print("Mushroom Mean Error With Pruning: ", mushroom_mean_error_prune)
    print("Mushroom STD Error With Pruning: ", mushroom_std_error_prune)

    print(DIVIDER)
    print("ZOO")
    print(DIVIDER)

    print('Zoo err:', zoo_err)
    print('Zoo Model:', zoo_model)
    print('Zoo with pruning err:', zoo_err_p)
    print('Zoo with pruning Model:', zoo_model_p)

    # Mean error and Standard Deviation of Zoo Data Without Pruning
    zoo_mean_error = np.mean(zoo_err, 0)
    zoo_std_error = np.std(zoo_err, 0)
    print("Zoo Mean Error Without Pruning: ", zoo_mean_error)
    print("Zoo STD Error Without Pruning: ", zoo_std_error)

    print(DIVIDER)
    print("I am pruned")
    print(DIVIDER)

    # Mean error and Standard Deviation of Zoo Data With Pruning
    zoo_mean_error_prune = np.mean(zoo_err_p, 0)
    zoo_std_error_prune = np.std(zoo_err_p, 0)
    print("Zoo Mean Error With Pruning: ", zoo_mean_error_prune)
    print("Zoo STD Error With Pruning: ", zoo_std_error_prune)


if __name__ == '__main__':
    main()
