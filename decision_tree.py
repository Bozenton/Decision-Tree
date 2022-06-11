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
        if not self.children:
            ret = "\t"*level + repr(self.leaf_label) + "\n"
        else:
            ret = "\t"*level + repr(self.label) + "\n"
        if self.children:
            for child in self.children:
                ret += child.__str__(level+1)
        return ret
    
    def mermaid(self):
        ss = self.gen_mermaid()
        ss = 'graph TB\nroot' + ss
        return ss
    
    def gen_mermaid(self, prefix='a'):
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
        leafs = []
        self._collect_leaf_nodes(self,leafs)
        return leafs

    def _collect_leaf_nodes(self, node, leafs):
        if node is not None:
            if not node.children:
                leafs.append(node)
            else:
                for n in node.children:
                    self._collect_leaf_nodes(n, leafs)
    
    def get_parents_cond(self):
        return self._get_parents_cond(self)
    
    def _get_parents_cond(self, node):
        if not node.parent:
            return 'True'  # for pd.DataFrame.query
        else:
            parent_label = self._get_parents_cond(node.parent)
            return node.label + '&' + parent_label
    
    
class DecisionTree():
    def __init__(self, train_data: pd.DataFrame, 
                    mapping_dict = None,
                    stop_threshold=0):
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
        return data.attrs[col]

    def stopping_condition(self, data:pd.DataFrame):
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
        """ select the attribute with largest information gain

        Args:
            data (pd.DataFrame): data with label in the last column

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
        return data.iloc[:,-1].mode().values[0]
    
    
    def preprocess(self, data:pd.DataFrame, max_k=5, max_each=8):
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
        self._pruning(self.root_cut, data, alp)
        return self.root_cut
    
    def _pruning(self, node:Node, data:pd.DataFrame, alp=1):
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

            