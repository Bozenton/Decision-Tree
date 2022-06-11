import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from copy import deepcopy

from decision_tree import Node, DecisionTree

if __name__ == '__main__':
    
    data_path = '.\data\iris.data'
    iris_raw_data = pd.read_csv(data_path, 
            names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'], 
            delimiter=',')

    train_mask = np.random.rand(len(iris_raw_data)) < 0.8
    temp_train_iris_data = iris_raw_data[train_mask]
    test_iris_data = iris_raw_data[~train_mask]
    print(f'Train: {temp_train_iris_data.shape[0]} samples, Test: {test_iris_data.shape[0]}')

    train_iris_data = temp_train_iris_data.copy()
    train_iris_data['class'], mapping_dict= temp_train_iris_data['class'].factorize()
    train_iris_data.attrs =dict(zip(train_iris_data.columns.values, [False, False, False, False, False]))
    tree = DecisionTree(train_iris_data, mapping_dict)
    print('\n')
    print(tree.root.mermaid())

    tree.pruning(train_iris_data, alp=3)
    print(tree.root_cut.mermaid())

    pred_data = tree.predict(test_iris_data)
    
    print("Predict in test data")
    print(pred_data)