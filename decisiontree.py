import numpy as np
import pandas as pd
from collections import Counter

# Define the dataset (tennis playing example)
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast',
                'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = -np.sum([(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy_val

# Function to calculate information gain
def info_gain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    info_gain_val = total_entropy - weighted_entropy
    return info_gain_val

# Function to get the attribute with the highest information gain
def find_best_attribute(data, target_name, attributes):
    info_gains = [info_gain(data, attribute, target_name) for attribute in attributes]
    best_attribute_index = np.argmax(info_gains)
    return attributes[best_attribute_index]

# Function to build the decision tree
def build_tree(data, target_name, attributes):
    # Get unique classes in target column
    unique_classes = np.unique(data[target_name])
    
    # If all instances belong to the same class, return that class
    if len(unique_classes) == 1:
        return unique_classes[0]
    
    # If no attributes are left, return the majority class
    elif len(attributes) == 0:
        return Counter(data[target_name]).most_common(1)[0][0]
    
    else:
        # Find the attribute with the highest information gain
        best_attr = find_best_attribute(data, target_name, attributes)
        
        # Create tree structure
        tree = {best_attr: {}}
        remaining_attributes = [attr for attr in attributes if attr != best_attr]
        
        # Get unique values of best attribute
        unique_vals = np.unique(data[best_attr])
        
        # Recursively build the tree
        for val in unique_vals:
            subset = data.where(data[best_attr] == val).dropna()
            subtree = build_tree(subset, target_name, remaining_attributes)
            tree[best_attr][val] = subtree
        
        return tree

# Function to classify a new sample using the decision tree
def classify(tree, sample):
    # Traverse the tree until a leaf node is reached
    for node in tree.keys():
        value = sample[node]
        tree = tree[node][value]
        prediction = 0
        
        if type(tree) is dict:
            prediction = classify(tree, sample)
        else:
            prediction = tree
            break
    
    return prediction

# Example usage:
attributes = list(df.columns[:-1])  # All columns except the last one (target)
target_name = df.columns[-1]        # Last column (target)

# Build the decision tree
tree = build_tree(df, target_name, attributes)
print("Constructed Decision Tree:\n", tree)

# Example new sample to classify
new_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'Normal',
    'Windy': False
}

# Classify the new sample
prediction = classify(tree, new_sample)
print("\nPrediction for new sample:", prediction)
