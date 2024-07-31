import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression class
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            # Linear model
            model = np.dot(X, self.theta) + self.bias
            predictions = model

            # Compute gradients
            error = predictions - y
            gradient_theta = np.dot(X.T, error) / self.m
            gradient_bias = np.sum(error) / self.m

            # Update parameters
            self.theta -= self.learning_rate * gradient_theta
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X):
        return np.dot(X, self.theta) + self.bias

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=1, noise=0.1, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Visualization
def plot_regression_line(X, y, model):
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

# Plot regression line
plot_regression_line(X_test, y_test, model)
