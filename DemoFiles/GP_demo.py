import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Training data
X = np.linspace(0, 10, 10).reshape(-1, 1)
y_true = np.sqrt(X).ravel()

# Kernel: amplitude * RBF(length_scale)
kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
# -------------------------------------------------
# CASE 1: Exact data (no noise)
# -------------------------------------------------

# GP: assume exact data (no noise)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-12, optimizer=None, normalize_y=False)
gp.fit(X, y_true)

# Predict on training and test points
X_pred = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True)
y_train_pred = gp.predict(X)
print(max(sigma))
# Check that the GP interpolates
print("Max absolute error at training points:", np.max(np.abs(y_train_pred - y_true)))

# Plot
plt.figure(figsize=(9, 6))
plt.plot(X, y_true, 'ro', label="Data")
plt.plot(X_pred, np.sqrt(X_pred), 'k--', label="True function: √x")
plt.plot(X_pred, y_pred, 'b', label="GP mean")
plt.fill_between(X_pred.ravel(), y_pred - 2*sigma, y_pred + 2*sigma,
                 color='blue', alpha=0.2, label="±2std")
plt.title("GP with exact data")
plt.legend()
plt.show()
# -------------------------------------------------
# CASE 2: Noisy data
# -------------------------------------------------
noise_std = 0.1
X = np.linspace(0, 10, 30).reshape(-1, 1)
y_true = np.sqrt(X).ravel()
y_noisy = y_true + noise_std * np.random.randn(len(y_true))

gp_noisy = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=False)
gp_noisy.fit(X, y_noisy)

y_pred_noisy, sigma_noisy = gp_noisy.predict(X_pred, return_std=True)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(X, y_noisy, 'r.', markersize=10, label="Data")
plt.plot(X_pred, np.sqrt(X_pred), 'k--', label="True function: √x")
plt.plot(X_pred, y_pred_noisy, 'b', label="GP mean")
plt.fill_between(
    X_pred.ravel(),
    y_pred_noisy - 2*sigma_noisy,
    y_pred_noisy + 2*sigma_noisy,
    alpha=0.2, color='blue',
    label="±2std"
)
plt.title("GP with noisy data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
