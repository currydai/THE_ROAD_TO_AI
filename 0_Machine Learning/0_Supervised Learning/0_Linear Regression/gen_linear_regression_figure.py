import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 80
X = np.linspace(-3, 3, n).reshape(-1, 1)
true_w, true_b = 3.0, 2.0
y = true_w * X[:, 0] + true_b + np.random.normal(0, 1.0, size=n)

X_aug = np.hstack([X, np.ones((n, 1))])
theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
w_hat, b_hat = theta[0], theta[1]

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X[:, 0], y, s=18, alpha=0.7, label='data')
xx = np.linspace(X.min(), X.max(), 200)
yy = w_hat * xx + b_hat
ax.plot(xx, yy, color='crimson', lw=2.0, label=f'fit: y={w_hat:.2f}x+{b_hat:.2f}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('Linear Regression (Closed-form OLS)')

fig_dir = os.path.join("0_Machine Learning", "0_Supervised Learning", "0_Linear Regression", "figures")
os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(fig_dir, "linear_regression_fit.png")
plt.tight_layout()
plt.savefig(out_path, dpi=160)
print("saved to", out_path)
plt.show()
