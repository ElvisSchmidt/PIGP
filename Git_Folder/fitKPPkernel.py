import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
Normalize = True
# --- 1. Load your data ---
data = pd.read_csv("CellData24-48.csv")  # header included
#data = pd.read_csv("SimulatedDataKPP.csv")
y = data["y"].values # times 10^3 if we take the data from the dataset
noise = np.random.normal(loc=0.0, scale=0.1, size=y.shape)
y= y+noise

x= data["x"].values
t=data["t"].values
X = data[["x", "t"]].values  # inputs: 2D (x, t)
N = X.shape[0]
if Normalize:
     #If we want to normalize
    x_norm = (x - min(x)) / (max(x) - min(x))
    t_norm = (t - min(t)) / (max(t) - min(t))
    X = np.column_stack((x_norm,t_norm))
else:
    X = np.column_stack((x,t))

# --- 2. Train the original GP ---
kernel = (
    C(1, (1e-3, 1e3)) *
    RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-3, 1e5)) +
    WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 1e5))
)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)
gp.fit(X, y)

# --- 3. Output original GP hyperparameters ---
print("=== Original GP ===")
print("Kernel:", gp.kernel_)
print("Learned lengthscales (x, t):", gp.kernel_.k1.k2.length_scale)
print("Estimated output scale σ²:", gp.kernel_.k1.k1.constant_value)
print("Estimated noise variance:", gp.kernel_.k2.noise_level)

# --- 4. Compute posterior mean of original GP ---
K1 = gp.kernel_.k1.k2(X, X)           # RBF part only
sigma2 = gp.kernel_.k2.noise_level
l1x, l1t = gp.kernel_.k1.k2.length_scale

K1_noise = K1 + sigma2 * np.eye(N)

C1 = K1 @ np.linalg.solve(K1_noise, y)
# --- 5. Compute cov(∂²U/∂x², U) for derivative GP ---
dx = X[:, 0][:, None] - X[:, 0][None, :]
LKL = ((dx**2) / l1x**4 - 1 / l1x**2) * K1

# --- 6. Posterior mean of derivative GP ---
C2 = LKL @ np.linalg.solve(K1, C1)
# --- 7. Fit a new GP for C2 ---
kernel_deriv = (
    C(0.01, (1e-8,1e10)) *
    RBF(length_scale=[0.5, 0.5],length_scale_bounds=[(0.01,10),(0.01,10)]) +
    WhiteKernel(noise_level=0.05,noise_level_bounds=(1e-30,1))
)

gp_deriv = GaussianProcessRegressor(kernel=kernel_deriv, n_restarts_optimizer=10, normalize_y=False)
gp_deriv.fit(X, C2)

# --- 8. Output derivative GP hyperparameters ---
print("\n=== Derivative GP (∂²U/∂x²) ===")
print("Kernel:", gp_deriv.kernel_)
print("Learned lengthscales (x, t):", gp_deriv.kernel_.k1.k2.length_scale)
print("Estimated output scale σ²:", gp_deriv.kernel_.k1.k1.constant_value)
print("Estimated noise variance:", gp_deriv.kernel_.k2.noise_level)

# --- 9. Plot original GP at a fixed t ---
t_fixed = 0.5
Xplot = np.array([[xi, t_fixed] for xi in np.linspace(X[:,0].min(), X[:,0].max(), 200)])
y_mean, y_std = gp.predict(Xplot, return_std=True)
y_mean = np.clip(y_mean,0,None)
y_std = np.clip(y_std,0,None)

#Plot
plt.figure(figsize=(8,4))
plt.fill_between(Xplot[:,0], y_mean - y_std, y_mean + y_std, color='lightblue', alpha=0.5, label="±1 std")
plt.plot(Xplot[:,0], y_mean, 'b-', label="GP mean")
plt.scatter(X[X[:,1]==t_fixed,0], y[X[:,1]==t_fixed], c='r', label="data")
plt.xlabel("x"); plt.ylabel("y")
plt.title(f"Original GP at t={t_fixed}")
plt.legend()
plt.show()

# --- 10. Plot derivative GP at the same fixed t ---
C2_mean, C2_std = gp_deriv.predict(Xplot, return_std=True)

plt.figure(figsize=(8,4))
plt.fill_between(Xplot[:,0], C2_mean - C2_std, C2_mean + C2_std, color='lightgreen', alpha=0.5, label="±1 std")
plt.plot(Xplot[:,0], C2_mean, 'g-', label="Derivative GP mean")
plt.scatter(X[X[:,1]==t_fixed,0], C2[X[:,1]==t_fixed], c='r', label="C2 (posterior derivative)")
plt.xlabel("x"); plt.ylabel("∂²U/∂x²")
plt.title(f"Derivative GP (∂²U/∂x²) at t={t_fixed}")
plt.legend()
plt.show()

t_init = 24
# Create a dense grid in x
x_init = np.linspace(X[:,0].min(), X[:,0].max(), 76*3)
# Build the input array for GP: (x, t)
X_init = np.column_stack([x_init, np.full_like(x_init, t_init)])

#import pandas as pd
#
#df_init = pd.DataFrame({
#    "x": x_init,
#    "y0": y_mean
#})
#
#df_init.to_csv("InitialCondition.csv", index=False)
