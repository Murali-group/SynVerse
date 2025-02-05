import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Given samples of X and Y
X_values = np.random.normal(0, 1, 1000)  # Example: Replace with actual X values
Y_values = np.random.normal(2, 1.5, 1000)  # Example: Replace with actual Y values

# Compute probability distributions using histogram (for discrete values)
hist_X, bin_edges_X = np.histogram(X_values, bins=30, density=True)
hist_Y, bin_edges_Y = np.histogram(Y_values, bins=30, density=True)

# Compute bin centers
X_bins = (bin_edges_X[:-1] + bin_edges_X[1:]) / 2
Y_bins = (bin_edges_Y[:-1] + bin_edges_Y[1:]) / 2

# Normalize to ensure probabilities sum to 1 (if needed)
f_X = hist_X / np.sum(hist_X)
f_Y = hist_Y / np.sum(hist_Y)

# Compute the independent joint distribution
g_XY = np.outer(f_X, f_Y)  # g(X, Y) = f(X) * f(Y)

# Plot distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(X_bins, f_X, width=np.diff(bin_edges_X), alpha=0.7, label="Estimated f(X)")
plt.xlabel("X values")
plt.ylabel("Probability Density")
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(Y_bins, f_Y, width=np.diff(bin_edges_Y), alpha=0.7, label="Estimated f(Y)", color="orange")
plt.xlabel("Y values")
plt.ylabel("Probability Density")
plt.legend()

plt.show()

# Plot heatmap of the joint distribution
plt.figure(figsize=(8, 6))
plt.imshow(g_XY, cmap="Blues", aspect="auto",
           extent=[Y_bins.min(), Y_bins.max(), X_bins.max(), X_bins.min()])
plt.colorbar(label="Probability Density")
plt.xlabel("Y values")
plt.ylabel("X values")
plt.title("Independent Joint Distribution g(X, Y)")

plt.xticks(ticks=np.linspace(Y_bins.min(), Y_bins.max(), num=len(Y_bins)),
           labels=np.round(Y_bins, 2),  rotation=90)
plt.yticks(ticks=np.linspace(X_bins.min(), X_bins.max(), num=len(X_bins)),
           labels=np.round(X_bins, 2))

plt.show()
