import torch
from sklearn.datasets import load_breast_cancer



# load dataset and create tensor
X, y = load_breast_cancer(return_X_y=True)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)
data = load_breast_cancer()
print(data.feature_names)

def find_best_split(X, y):
    _, num_features = X.shape # already aware about samples thus not utilizing that parameter
    print(X.shape)

    best_gini = float('inf')
    best_feature = None
    best_threshold = None # starting with infinity just to ensure the first split is valid

    for feature_idx in range(num_features):
        feature_values = X[:, feature_idx] # returned values of each column, sorted and then just storing the unique values

        # get candidate thresholds from sorted unique values
        sorted_vals, _ = torch.sort(feature_values)
        unique_vals = torch.unique(sorted_vals)
        if len(unique_vals) <= 1:
            continue 

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

        for threshold in thresholds:
            left_mask = feature_values <= threshold # creating a boolean mask to generate nodes from root aka left and right.
            y_left = y[left_mask]
            y_right = y[~left_mask]

            # setting up gini impurity index for finding the best split (more homo = better), can use mse here to make this a regression tree rather than classification.

            # gini for left child
            counts_left = torch.bincount(y_left, minlength=2)
            probs_left = counts_left / counts_left.sum()
            gini_left = 1 - torch.sum(probs_left ** 2)

            # gini for right child
            counts_right = torch.bincount(y_right, minlength=2)
            probs_right = counts_right / counts_right.sum()
            gini_right = 1 - torch.sum(probs_right ** 2)

            # weighted impurity (using current node size)
            weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_idx
                best_threshold = threshold.item()  

    return best_feature, best_threshold, best_gini # feature, threshold, gini


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def make_leaf(y):
    # return a node with the mean as value
    return Node(value=torch.mode(y).values.item())
    

def build_tree(X, y, depth=0, max_depth=10, min_samples_split=2):
        if (depth >= max_depth or len(y) < min_samples_split or len(torch.unique(y)) == 1):
            return make_leaf(y)
        
        best_feature, best_threshold, _ = find_best_split(X, y) # gini isn't required here
        if best_feature is None:   
            return make_leaf(y)
        
        left_mask = X[:, best_feature] <= best_threshold
        left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_samples_split)
        right = build_tree(X[~left_mask], y[~left_mask], depth+1, max_depth, min_samples_split)
        
        return Node(feature_idx=best_feature, threshold=best_threshold, left=left, right=right)

tree = build_tree(X_t, y_t, max_depth=5, min_samples_split=5)

def predict_sample(node, x):
    if node.value is not None:      
        return node.value
    if x[node.feature_idx] <= node.threshold:
        return predict_sample(node.left, x)
    else:
        return predict_sample(node.right, x)

def predict(tree, X):
    return [predict_sample(tree, x) for x in X]





    
        
       

        
