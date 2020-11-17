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
    mushroomData = DataSet(target=0, name='mushrooms')
    zooData = DataSet(name='zoo', exclude=[0])

    zooExamples = zooData.examples
    zooInputs = zooData.inputs
    #DecisionTreeLearner(zooData)

    print(mushroomData)
    print(zooData)
    print(zooExamples)
    print(zooInputs)

if __name__ == '__main__':
    main()