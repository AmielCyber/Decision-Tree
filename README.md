# Decision-Tree

## Description
A Python implementation of a Decision Tree Classifier with pruning. 

### Input  
The program trains the Decision Tree Classifier using ten fold cross validation. 
The two data sets used are the mushroom and the zoo data set. For each class(mushroom/animal species), 
each attribute contains 2 or more classification. For example an animal in the zoo may have 2, 4 or 6(insect) legs. 
Hence, our tree classifier has to learn to classify a class with attributes with 2 or more values. 

The Decision Tree Classifier also takes in an input called p_value where it is used to calculate chi squared in ordered list 
to decide if branch needs to pruned. The chi squared is used to calculate the amount of information gained or lost when we 
branch out. If the information loss hits a threshold using the p_value, then that branch is pruned. This helps with
preventing over fitting and unnecessary recursive calls. 

The tree is also made to be used with other data in the `aima-data/` directory, given the right parameters are called when
the function `crossvalidation` is called.

### Output
The program will display the decision choices it has made along with the prediction of the class for the last choice it makes.
The program will also print a tree with pruning and without pruning along with its statistical data giving its prediction accuracy.

## Repository Overview
```
Decision-Tree
├── README.md
└── decision_trees
    ├── decision_tree.py
    ├── driver.py
    ├── ml_lib
    │   ├── aima-data
    │   │   ├── README.md
    │   │   ├── abalone.csv
    │   │   ├── abalone.txt
    │   │   ├── iris.csv
    │   │   ├── iris.txt
    │   │   ├── mushrooms.csv
    │   │   ├── mushrooms.txt
    │   │   ├── orings.csv
    │   │   ├── orings.txt
    │   │   ├── restaurant.csv
    │   │   ├── tiny_animal_set.csv
    │   │   ├── zoo.csv
    │   │   ├── zoo.txt
    │   │   └── zoo_label.txt
    │   ├── crossval.py
    │   ├── decision_tree_support.py
    │   ├── ml_util.py
    │   └── utils.py
    └── output.txt
```

## Running Instructions
1. Download this repository in to your preferred system's directory
2. Go to the `decision_tree` directory
3. Run the following command in your terminal `python3 driver.py`

## Authors
The following authors for the files `decision_tree.py` and `driver.py` only:
* [Lilan Vu](https://github.com/lilianvu99)
* [Amiel Nava](https://github.com/AmielCyber)

The writing of `decision_tree.py` and `driver.py` was written by pair programming from the authors mentioned above.

The rest of files are from the book companion code [*Artificial Intelligence: A Modern Approach*](https://github.com/aimacode/aima-python)
also know as [aimacode](https://github.com/aimacode) in GitHub.

## Resources
* [*Decision Tree Crash Course by Machine Learning* @ Berkeley](https://medium.com/@ml.at.berkeley/machine-learning-crash-course-part-5-decision-trees-and-ensemble-models-dcc5a36af8cd)
## Sources
* Textbook [*Artificial Intelligence: A Modern Approach* Pseudocode](http://aima.cs.berkeley.edu/algorithms.pdf)
