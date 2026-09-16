import torch
from sklearn.datasets import fetch_california_housing

# load dataset and create tensor
X, y = fetch_california_housing(return_X_y=True)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
data = fetch_california_housing()
print(data.feature_names)

def find_best_split(X, y):
    _, num_features = X.shape # already aware about samples thus not utilizing that parameter
   
    best_feature = None
    best_threshold = None
    best_mse = float('inf')   # starting with infinity just to ensure the first split is valid

    for feature_idx in range(num_features):
        feature_values = X[:, feature_idx] 
        sorted_vals, _ = torch.sort(feature_values)
        unique_vals = torch.unique(sorted_vals) #returned values of each column, sorted and then just storing the unique values

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2 

        for threshold in thresholds: 
            left_mask = feature_values <= threshold # creating a boolean mask to generate nodes from root aka left and right.
            y_left = y[left_mask]
            
            y_right = y[~left_mask]
            if len(y_left) < 10 or len(y_right) < 10:
                continue
            # setting up mse for finding the best split (more homo = better), can use gini impurity index here to make this a classification tree rather than regression.
            mse_left = torch.mean((y_left - torch.mean(y_left))**2) 
            mse_right = torch.mean((y_right - torch.mean(y_right))**2)

            weighted_mse = (len(y_left) * mse_left + len(y_right) * mse_right) / len(y)

            if weighted_mse < best_mse:
                best_mse = weighted_mse
                best_feature = feature_idx
                best_threshold = threshold.item()

    return best_mse, best_feature, best_threshold   # mse, feature, threshold

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def make_leaf(y):
    # return a node with the mean as value
    return Node(value=torch.mean(y).item())

def build_tree(X, y, depth=0, max_depth=10, min_samples_split=2, min_variance=(10**(-5))):
    # stopping conditions are max depth, min samples, or very low variance
 
    if (depth >= max_depth or len(y) < min_samples_split or torch.var(y, unbiased=False) < min_variance):
        return make_leaf(y)

    _, best_feature, best_threshold = find_best_split(X, y) # mse isn't required here
    if best_feature is None:
        return make_leaf(y)

    left_mask = X[:, best_feature] <= best_threshold
    left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_samples_split)
    right = build_tree(X[~left_mask], y[~left_mask], depth+1, max_depth, min_samples_split)

    return Node(feature_idx=best_feature, threshold=best_threshold, left=left, right=right)

tree = build_tree(X_t, y_t, max_depth=3, min_samples_split=50)

def predict_sample(node, x):
    if node.value is not None:      
        return node.value
    if x[node.feature_idx] <= node.threshold:
        return predict_sample(node.left, x)
    else:
        return predict_sample(node.right, x)

def predict(tree, X):
    return torch.tensor([predict_sample(tree, x) for x in X])

def print_tree(node, feature_names, depth=0):
    if node.value is not None:
        print("  " * depth + f"-> Predict: {node.value:.2f}")
    else:
        print("  " * depth + f"{feature_names[node.feature_idx]} <= {node.threshold:.2f}")
        print_tree(node.left, feature_names, depth + 1)
        print_tree(node.right, feature_names, depth + 1)

print_tree(tree,data.feature_names)

