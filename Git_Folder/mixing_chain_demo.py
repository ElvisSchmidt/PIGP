import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
n_chains = 7          # number of Markov chains
n_steps = 30        # number of time steps
alpha = 0.05          # pull strength toward 0
sigma = 0.5           # noise standard deviation
initial_positions = np.linspace(-10, 10, n_chains)  # initial states spread out

# --- Simulation ---
X = np.zeros((n_chains, n_steps))
X[:, 0] = initial_positions

for t in range(1, n_steps):
    noise = np.random.normal(0, sigma, size=n_chains)
    X[:, t] = X[:, t-1] - alpha * X[:, t-1] + noise

# --- Color palette (distinct and readable) ---
cmap = plt.cm.get_cmap("tab20", n_chains)
colors = [cmap(i) for i in range(n_chains)]

# --- Compute variances ---
variances = np.var(X, axis=1)

# --- Plot ---
fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# (1) Time-series plot
for i in range(n_chains):
    axs[0].plot(X[i, :], label=f'Chain {i+1}', color=colors[i], linewidth=1.6)

axs[0].axhline(0, color='k', linestyle='--', linewidth=1)
axs[0].set_title( 'Not Mixed Chains')
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('State value')
axs[0].legend(ncol=2, fontsize=9,loc='upper right')
axs[0].grid(True, linestyle=':', alpha=0.6)

# (2) Histograms + Variance info
all_samples = X.flatten()

for i in range(n_chains):
    axs[1].hist(X[i, :], bins=40, density=True, alpha=0.5,
                color=colors[i],
                label=f'Var = {variances[i]:.2f}')

# Combined histogram (all chains)
axs[1].hist(all_samples, bins=40, density=True, alpha=0.25,
            color='black', label=f'Total Var = {np.var(all_samples):.2f}',
            edgecolor='none')

axs[1].set_title('State Distributions')
axs[1].set_xlabel('State value')
axs[1].set_ylabel('Density')
axs[1].legend(ncol=2, fontsize=9,loc='upper right')
axs[1].grid(True, linestyle=':', alpha=0.6)

plt.show()
