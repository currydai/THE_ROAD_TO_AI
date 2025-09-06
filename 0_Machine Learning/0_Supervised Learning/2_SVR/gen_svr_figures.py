"""
Generate figures for the SVR chapter:
1) Fit an RBF-SVR to synthetic non-linear data; highlight support vectors.
2) Visualize the effects of C, epsilon, and gamma on the regression curve.

Saves:
- figures/svr_rbf_fit.png
- figures/svr_params_effect.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def main() -> None:
    np.random.seed(7)

    # 1) Synthetic non-linear data: y = sin(1.5x) + 0.5x + noise
    n = 200
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 0] + np.random.normal(0, 0.2, size=n)

    # Standardize features (common for SVR). Keep y in original scale.
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 2) Fit RBF-SVR
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    svr.fit(Xs, y)

    # Predictions on a dense grid (transform with same scaler)
    xx = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)
    xg = scaler.transform(xx)
    yy = svr.predict(xg)

    # 3) Plot fit and support vectors
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(X[:, 0], y, s=15, alpha=0.6, label="data")
    ax.plot(xx[:, 0], yy, color="crimson", lw=2.0, label="SVR (RBF)")
    ax.scatter(
        X[svr.support_, 0],
        y[svr.support_],
        s=35,
        facecolors="none",
        edgecolors="k",
        label="support vectors",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SVR (RBF) fit and support vectors")
    ax.legend(loc="best", fontsize=8)

    fig_dir = os.path.join(
        "0_Machine Learning", "0_Supervised Learning", "2_SVR", "figures"
    )
    os.makedirs(fig_dir, exist_ok=True)
    out_fit = os.path.join(fig_dir, "svr_rbf_fit.png")
    plt.tight_layout()
    plt.savefig(out_fit, dpi=160)

    # 4) Hyperparameter effects: vary C, epsilon, gamma
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)

    # (a) vary C
    for C in [0.3, 1.0, 10.0]:
        m = SVR(kernel="rbf", C=C, epsilon=0.1, gamma="scale").fit(Xs, y)
        axes[0].plot(xx[:, 0], m.predict(xg), label=f"C={C}")
    axes[0].scatter(X[:, 0], y, s=8, alpha=0.3, color="gray")
    axes[0].set_title("Effect of C")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8)

    # (b) vary epsilon
    for e in [0.05, 0.2, 0.5]:
        m = SVR(kernel="rbf", C=10.0, epsilon=e, gamma="scale").fit(Xs, y)
        axes[1].plot(xx[:, 0], m.predict(xg), label=f"eps={e}")
    axes[1].scatter(X[:, 0], y, s=8, alpha=0.3, color="gray")
    axes[1].set_title("Effect of epsilon")
    axes[1].set_xlabel("x")
    axes[1].legend(fontsize=8)

    # (c) vary gamma (kernel width)
    for g in [0.3, 1.0, 3.0]:
        m = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma=g).fit(Xs, y)
        axes[2].plot(xx[:, 0], m.predict(xg), label=f"gamma={g}")
    axes[2].scatter(X[:, 0], y, s=8, alpha=0.3, color="gray")
    axes[2].set_title("Effect of gamma")
    axes[2].set_xlabel("x")
    axes[2].legend(fontsize=8)

    out_params = os.path.join(fig_dir, "svr_params_effect.png")
    plt.tight_layout()
    plt.savefig(out_params, dpi=160)

    print("saved to", out_fit, "and", out_params)


if __name__ == "__main__":
    main()

