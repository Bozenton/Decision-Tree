import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from copy import deepcopy

from entropy import empiricalEntropy, empiricalConditionalEntropy
from kmeans import Kmeans1d

class Node():
    def __init__(self, label=None, parent=None, children=None):
        self.parent = parent
        self.children = children
        self.label = label  
        self.leaf_label = None
        self.test_cond = None   # test condition of each internal node is here
                                # list of str (used in pd.DataFrame.query)
    
    def __str__(self, level=0):
        """generate the tree structure in string

        Args:
            level (int, optional): the level of this node. Defaults to 0.

        Returns:
            _type_: string of tree structure
        """
        if not self.children:
            ret = "\t"*level + repr(self.leaf_label) + "\n"
        else:
            ret = "\t"*level + repr(self.label) + "\n"
        if self.children:
            for child in self.children:
                ret += child.__str__(level+1)
        return ret
    
    def mermaid(self):
        """generate the mermaid code for displaying tree structure

        Returns:
            string: mermaid code
        """
        ss = self.gen_mermaid()
        ss = 'graph TB\nroot' + ss
        return ss
    
    def gen_mermaid(self, prefix='a'):
        """recursion to generate mermaid code

        Args:
            prefix (str, optional): . Defaults to 'a'.

        Returns:
            string: main part of mermaid code
        """
        if not self.children:
            ret = "--" + self.label + "-->" + prefix + "(" + self.leaf_label + ")\n"
        else:
            cond = " ".join(re.findall("[a-zA-Z]+", self.test_cond[0]))
            ret = "--" + self.label + "-->" + prefix + "(" + cond + ")\n" 
        if self.children:
            for j, child in enumerate(self.children):
                ret += prefix + "("+cond+")"
                ret += child.gen_mermaid(prefix+str(j))
        return ret
    
    def get_leaf_nodes(self):
        """get all the leafs of this node (not exactly its direct children)

        Returns:
            list of Node: all the leafs
        """
        leafs = []
        self._collect_leaf_nodes(self,leafs)
        return leafs

    def _collect_leaf_nodes(self, node, leafs):
        """recursion to get leafs

        Args:
            node (Node): node
            leafs (list of Node): list of leafs
        """
        if node is not None:
            if not node.children:
                leafs.append(node)
            else:
                for n in node.children:
                    self._collect_leaf_nodes(n, leafs)
    
    def get_parents_cond(self):
        """get all its ancestors' test condition, which can be used to 
            reach this node from root

        Returns:
            string: conditions to reach this node from root
        """
        return self._get_parents_cond(self)
    
    def _get_parents_cond(self, node):
        """recursion to get ancestors' test condition

        Args:
            node (Node): node

        Returns:
            string: conditions to reach this node from root
        """
        if not node.parent:
            return 'True'  # for pd.DataFrame.query
        else:
            parent_label = self._get_parents_cond(node.parent)
            return node.label + '&' + parent_label
    
    
class DecisionTree():
    def __init__(self, train_data: pd.DataFrame, 
                    mapping_dict = None,
                    stop_threshold=1e-2):
        """initialize the decision tree

        Args:
            train_data (pd.DataFrame): train data
            mapping_dict (dict, optional): dictionary to map classes to number. Defaults to None.
            stop_threshold (int, optional): the threshold of info gain to stop extend tree. Defaults to 0.
        """
        self.stop_threshold = stop_threshold
        self.mapping_dict = mapping_dict
        # pd.DataFrame.attrs should contain whether each column is discrete or not
        for c in train_data.columns.values:
            if c not in train_data.attrs.keys():
                raise ValueError("Input data (pd.DataFrame) should contain whether each column is discrete or not")
        # discretize the columns with continuous data
        self.dividing_points_dict = self.preprocess(train_data)
        
        self.root = self.grow(train_data)
        self.root.label = 'root'
        self.root_cut = deepcopy(self.root)

    def grow(self, data:pd.DataFrame):
        """recursion to grow tree

        Args:
            data (pd.DataFrame): dataset

        Returns:
            Node: root of the decision tree
        """
        if self.stopping_condition(data):
            # finnally we reach the leaf
            leaf = Node()
            leaf.leaf_label = self.classify(data)
            leaf.leaf_label = self.mapping_dict[leaf.leaf_label]
            return leaf
        else:
            root = Node(label=None, children=list())
            root.test_cond, attr = self.find_best_split(data)
            for lb in root.test_cond:
                assert type(lb) is str
                sub_data = data.query(lb)
                if sub_data.empty:
                    continue
                sub_data = sub_data.drop(columns=[attr])  # this attribute has been used this time
                child = self.grow(sub_data)
                child.parent = root
                child.label = lb
                root.children.append(child)
        return root

    def check_discrete(self, data: pd.DataFrame, col: str):
        """check if the attribute is discrete

        Args:
            data (pd.DataFrame): dataset
            col (str): name of the attribute

        Returns:
            Bool: discrete or not
        """
        return data.attrs[col]

    def stopping_condition(self, data:pd.DataFrame):
        """check whether to stop extending decision tree

        Args:
            data (pd.DataFrame): dataset

        Returns:
            Bool: stop or not
        """
        assert len(data.columns.values) > 0, "Cannot find the label column"
        if len(data.columns.values) == 1:   # the last one column is the label of each sample
            # all attributes have been used, thus stop
            return True
        if data.iloc[:,-1].nunique() == 1:
            return True
        max_info_gain, _, _ = self.find_largest_info_gain(data)
        # print(max_info_gain)
        if max_info_gain < self.stop_threshold:
            return True
        
        return False

    def find_best_split(self, data:pd.DataFrame):
        """find the best test_condition (attribute and dividing point) for this dataset

        Args:
            data (pd.DataFrame): dataset

        Returns:
            str: test_condition
            str: attribute with max infomation gain
        """
        # select the attribute with largest information gain
        _, attr_with_max_info_gain, attr_discrete = self.find_largest_info_gain(data)
        if attr_discrete:
            # for example, if available values for attr 'A' is [0, 1 ,2], then cond = ['A==0', 'A==1', 'A==2']
            cond = [attr_with_max_info_gain+'=='+str(i) for i in data[attr_with_max_info_gain].unique()]
        else:
            # raise ValueError("Continuous data is not available now")
            dividing_points = self.dividing_points_dict[attr_with_max_info_gain]
            cond = []
            for i in range(len(dividing_points)-1):
                if dividing_points[i] == -np.inf:
                    ss = attr_with_max_info_gain + "<%.4f"%dividing_points[i+1]
                    cond.append(ss)
                elif dividing_points[i+1] == np.inf:
                    ss = attr_with_max_info_gain + ">=%.4f"%dividing_points[i]
                    cond.append(ss)
                else:
                    ss = "%.4f<="%dividing_points[i] +  attr_with_max_info_gain + "<%.4f"%dividing_points[i+1]
                    cond.append(ss)

        return cond, attr_with_max_info_gain
    
    def find_largest_info_gain(self, data:pd.DataFrame):
        """ select the attribute with largest information gain in the input dataset

        Args:
            data (pd.DataFrame): dataset (sub dataset) with label in the last column

        Returns:
            float: attr_with_max_info_gain, max_info_gain, attr_discrete
        """
        max_info_gain = -np.inf
        for col in data.columns.values[:-1]: # the last column is the label, thus ignore it
            discrete = self.check_discrete(data, col)
            if discrete:
                info_gain = empiricalEntropy(data.iloc[:, -1]) - \
                            empiricalConditionalEntropy(label=data.iloc[:, -1], feature=data[col])
            else:
                col_feature = np.digitize(data[col].values, self.dividing_points_dict[col])-1
                col_feature = pd.Series(col_feature)
                info_gain = empiricalEntropy(data.iloc[:, -1]) - \
                            empiricalConditionalEntropy(label=data.iloc[:, -1], feature=col_feature)
                
            # get largest info gain
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                attr_with_max_info_gain = col
                attr_discrete = discrete
        return max_info_gain, attr_with_max_info_gain, attr_discrete
        
    
    def classify(self, data:pd.DataFrame):
        """classify the leaf node, depend on the type that appears most frequent

        Args:
            data (pd.DataFrame): dataset (sub dataset)

        Returns:
            int: the class of leaf according to the input dataset
        """
        return data.iloc[:,-1].mode().values[0]
    
    
    def preprocess(self, data:pd.DataFrame, max_k=5, max_each=8):
        """discretize the attribute with continuous values

        Args:
            data (pd.DataFrame): dataset
            max_k (int, optional): max k to try. Defaults to 5.
            max_each (int, optional): max number of trial for each k. Defaults to 8.

        Returns:
            dict: dividing points for every continuous attribute
        """
        cols = data.columns.values[:-1] # except the last one
        dividing_points_dict = {}
        kmeans = Kmeans1d()
        for col in cols:
            discrete = self.check_discrete(data, col)
            if not discrete:
                # apply kmeans to continuous data
                best_score, best_cps, best_rcd, all_score = \
                    kmeans.auto_fit(data[col].values, max_k=max_k, max_each=max_each)
                dividing_points = kmeans.get_dividing_points(best_cps)
                dividing_points_dict[col] = dividing_points
                print(f"Attribute {col} has been discretized")
        return dividing_points_dict
    
    def pruning(self, data:pd.DataFrame, alp=1):
        """pruning the tree

        Args:
            data (pd.DataFrame): dataset
            alp (int, optional): the coefficient to balance the predict loss and 
                                the regularization term. Defaults to 1.

        Returns:
            Node: root of the tree after pruning
        """
        self._pruning(self.root_cut, data, alp)
        return self.root_cut
    
    def _pruning(self, node:Node, data:pd.DataFrame, alp=1):
        """recursion to pruning

        Args:
            node (Node): node
            data (pd.DataFrame): dataset
            alp (int, optional): coefficient to balance. Defaults to 1.
        """
        leafs = node.get_leaf_nodes()
        new_leafs = node.get_leaf_nodes()
        cannot_cut_any_more = True
        for leaf in leafs:
            if leaf not in new_leafs:
                # it has already been cut because of another leaf
                continue
            err, _ = self._predict_err(new_leafs, data)
            leafs_cut = []
            for lf in new_leafs:
                if lf not in leaf.parent.get_leaf_nodes():
                    leafs_cut.append(lf)
            leafs_cut.append(leaf.parent)
            err_cut, sub_data = self._predict_err(leafs_cut, data)
            num_leafs = len(new_leafs)
            num_leafs_cut = len(leafs_cut)
            
            loss = err + alp*num_leafs
            loss_cut = err_cut + alp*num_leafs_cut
            # print(len(new_leafs), len(leafs_cut), loss, loss_cut)
            if loss_cut <= loss:
                # cut, leaf's parent becomes new leaf
                leaf.parent.children = list()
                leaf.parent.leaf_label = sub_data.iloc[:,-1].mode().values[0]
                leaf.parent.leaf_label = self.mapping_dict[leaf.parent.leaf_label]
                new_leafs = node.get_leaf_nodes()
                cannot_cut_any_more = False
        if cannot_cut_any_more:
            return 
        else:
            self._pruning(node, data, alp)

    @staticmethod
    def _predict_err(leafs, data:pd.DataFrame):
        """compute predict error for decision with given leafs

        Args:
            leafs (list of Node): list of leafs
            data (pd.DataFrame): dataset

        Returns:
            float: predict error
        """
        err = 0
        for leaf in leafs:
            cond = leaf.get_parents_cond()
            # print(cond)
            if cond == 'True':
                sub_data = data.copy()
            else:
                sub_data = data.query(cond)
            empirical_entropy = empiricalEntropy(sub_data.iloc[:, -1])
            err += empirical_entropy*sub_data.iloc[:, -1].count()
        return err, sub_data
    
    def predict(self, data:pd.DataFrame):
        """classify the given data by the tree pruned

        Args:
            data (pd.DataFrame): dataset

        Returns:
            pd.DataFrame: classified dataset with predict class added in the last column
        """
        leafs = self.root_cut.get_leaf_nodes()
        new_data = []
        for lf in leafs:
            cond = lf.get_parents_cond()
            sub_data = data.query(cond)
            pred_class = lf.leaf_label
            pred_class = [pred_class]*sub_data.shape[0]
            sub_data.insert(len(sub_data.columns), 'pred_class', pred_class)
            new_data.append(sub_data)
        return pd.concat(new_data)

            