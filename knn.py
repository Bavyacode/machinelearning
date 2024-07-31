"""import numpy as np
from collections import Counter

# Provided data
data = [
    ["Ajay", 32, 0, "Football"],
    ["Mark", 40, 0, "Neither"],
    ["Sara", 16, 1, "Cricket"],
    ["Zaira", 34, 1, "Cricket"],
    ["Sachin", 55, 0, "Neither"],
    ["Rahul", 40, 0, "Cricket"],
    ["Pooja", 20, 1, "Neither"],
    ["Smith", 15, 0, "Cricket"],
    ["Laxmi", 55, 1, "Football"],
    ["Michael", 15, 0, "Football"]
]

# Convert data to numpy array for easier manipulation
X = np.array([[age, gender] for _, age, gender, _ in data])
y = np.array([sport for _, _, _, sport in data])

# New instance
new_instance = np.array([9, 0])

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return the indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Choose the value of K
k = 3
model = KNN(k=k)
model.fit(X, y)

# Predict the class of the new instance
prediction = model._predict(new_instance)
print(f'The class of sports for Hari is: {prediction}')"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
species_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    sample = scaler.transform(sample)
    prediction = knn.predict(sample)
    return species_names[prediction][0]

sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

print("Sample Data: ")
print("Sepal_length: ",sepal_length)
print("sepal_width: ",sepal_width)
print("petal_length: ",petal_length)
print("petal_width: ",petal_width)
print("\n")
predicted_species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
print(f"Predicted species: {predicted_species}")
