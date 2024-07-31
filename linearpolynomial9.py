import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data with a non-linear relationship
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X = X**2  # Transforming X to introduce non-linearity

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)

# Polynomial Regression (degree 2 for comparison)
polynomial_features = PolynomialFeatures(degree=2)
polynomial_regressor = make_pipeline(polynomial_features, LinearRegression())
polynomial_regressor.fit(X_train, y_train)
y_pred_poly = polynomial_regressor.predict(X_test)

# Evaluate models
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Linear Regression - Mean Squared Error: {mse_linear:.4f}, R^2 Score: {r2_linear:.4f}")
print(f"Polynomial Regression - Mean Squared Error: {mse_poly:.4f}, R^2 Score: {r2_poly:.4f}")

# Visualization
plt.figure(figsize=(14, 6))

# Plot Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred_linear, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()

# Plot Polynomial Regression
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Data points')
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = polynomial_regressor.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial Regression (degree=2)')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Regression')
plt.legend()

plt.tight_layout()
plt.show()
