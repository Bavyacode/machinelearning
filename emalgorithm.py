import numpy as np
from scipy.stats import multivariate_normal

# Function to initialize parameters
def initialize_parameters(data, num_clusters):
    np.random.seed(0)
    mu = np.random.rand(num_clusters, data.shape[1])
    sigma = [np.eye(data.shape[1])] * num_clusters
    pi = np.array([1.0 / num_clusters] * num_clusters)
    return mu, sigma, pi

# Expectation step
def expectation_step(data, mu, sigma, pi):
    num_data_points = data.shape[0]
    num_clusters = len(mu)
    gamma = np.zeros((num_data_points, num_clusters))
    for i in range(num_data_points):
        for j in range(num_clusters):
            gamma[i, j] = pi[j] * multivariate_normal.pdf(data[i], mean=mu[j], cov=sigma[j])
        gamma[i] /= np.sum(gamma[i])
    return gamma

# Maximization step
def maximization_step(data, gamma):
    num_data_points = data.shape[0]
    num_clusters = gamma.shape[1]
    mu = np.zeros((num_clusters, data.shape[1]))
    sigma = [np.zeros((data.shape[1], data.shape[1]))] * num_clusters
    pi = np.zeros(num_clusters)
    for j in range(num_clusters):
        N_j = np.sum(gamma[:, j])
        pi[j] = N_j / num_data_points
        mu[j] = np.sum(gamma[:, j].reshape(-1, 1) * data, axis=0) / N_j
        for i in range(num_data_points):
            sigma[j] += gamma[i, j] * (data[i] - mu[j]).reshape(-1, 1) @ (data[i] - mu[j]).reshape(1, -1)
        sigma[j] /= N_j
    return mu, sigma, pi

# EM algorithm
def em_algorithm(data, num_clusters, max_iterations=1000):
    mu, sigma, pi = initialize_parameters(data, num_clusters)
    for _ in range(max_iterations):
        gamma = expectation_step(data, mu, sigma, pi)
        mu, sigma, pi = maximization_step(data, gamma)
    return mu, sigma, pi

# Example usage
np.random.seed(0)
data = np.concatenate([np.random.normal(0, 1, (100, 2)), np.random.normal(5, 2, (100, 2))])
num_clusters = 2
mu, sigma, pi = em_algorithm(data, num_clusters)
print('Mu:', mu)
print('Sigma:', sigma)
print('Pi:', pi)
