"""Generate Lasso and Ridge coefficient-path figures for the chapter.

Steps:
- Build synthetic data with some correlated features.
- Standardize features and center y (crucial for L1/L2).
- Compute Lasso coefficient paths across alphas.
- Sweep Ridge over alphas and record coefficients.
- Save figures into chapter-local figures/ for LaTeX inclusion.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, Ridge

# Reproducibility
np.random.seed(7)

# Data shape: n samples, d features
n, d = 120, 12

# 1) Synthetic features, with correlation between feature 0 and 1
X_raw = np.random.randn(n, d)
X_raw[:, 1] = 0.7 * X_raw[:, 0] + 0.3 * np.random.randn(n)

# Sparse ground-truth weights
true_w = np.zeros(d)
true_w[[0, 3, 7]] = [2.0, -3.0, 1.5]

# Generate target with noise
y = X_raw @ true_w + 0.8 * np.random.randn(n)

# 2) Standardize X (per-column) and center y
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
y = y - y.mean()

# 3) Lasso path (alphas descending); sklearn uses alpha=lambda/n_samples
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, alphas=None)

# Plot Lasso paths
fig, ax = plt.subplots(figsize=(7, 4.5))
for j in range(d):
    ax.plot(alphas_lasso, coefs_lasso[j, :], lw=1.6, label=f"w{j}")
ax.set_xscale('log')
ax.invert_xaxis()  # larger alphas on the left
ax.set_xlabel('alpha (log)')
ax.set_ylabel('coefficient')
ax.set_title('Lasso coefficient paths')
handles, labels = ax.get_legend_handles_labels()
ax.legend(loc='best', fontsize=8)

# Figures directory (relative to repo root) used by LaTeX \graphicspath
fig_dir = os.path.join(
    "0_Machine Learning", "0_Supervised Learning",
    "1_Regularization Methods in Regression", "figures"
)
os.makedirs(fig_dir, exist_ok=True)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'lasso_path.png'), dpi=160)

# 4) Ridge path across a grid of alphas
alphas_ridge = np.logspace(-3, 2, 40)
coefs_ridge = []
for a in alphas_ridge:
    model = Ridge(alpha=a, fit_intercept=False)
    model.fit(X, y)
    coefs_ridge.append(model.coef_)
coefs_ridge = np.array(coefs_ridge)

# Plot Ridge paths
fig, ax = plt.subplots(figsize=(7, 4.5))
for j in range(d):
    ax.plot(alphas_ridge, coefs_ridge[:, j], lw=1.6, label=f"w{j}")
ax.set_xscale('log')
ax.invert_xaxis()
ax.set_xlabel('alpha (log)')
ax.set_ylabel('coefficient')
ax.set_title('Ridge coefficient paths')
handles, labels = ax.get_legend_handles_labels()
ax.legend(loc='best', fontsize=8)

plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'ridge_path.png'), dpi=160)
print('saved to', os.path.join(fig_dir, 'lasso_path.png'), 'and ridge_path.png')
