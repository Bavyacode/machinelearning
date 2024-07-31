import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            model = np.dot(X, self.theta) + self.bias
            predictions = sigmoid(model)

            # Gradient calculation
            error = predictions - y
            gradient_theta = np.dot(X.T, error) / self.m
            gradient_bias = np.sum(error) / self.m

            # Update parameters
            self.theta -= self.learning_rate * gradient_theta
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X):
        model = np.dot(X, self.theta) + self.bias
        predictions = sigmoid(model)
        return [1 if i > 0.5 else 0 for i in predictions]

# Generate synthetic data with 2 features for visualization
X, y = make_classification(
    n_samples=1000, 
    n_features=2, 
    n_informative=2,  # Number of informative features
    n_redundant=0,    # Number of redundant features
    n_clusters_per_class=1, 
    flip_y=0,         # No label noise
    random_state=42
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Visualization
def plot_decision_boundary(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# Plot decision boundary
plot_decision_boundary(X_test, y_test, model)
