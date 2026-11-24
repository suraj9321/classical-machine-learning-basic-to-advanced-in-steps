"""Small plotting helpers shared across modules.

Import from a notebook inside modules/XX_name/ with:

    import sys; sys.path.append("../..")   # make the repo root importable
    from common.plotting import plot_decision_boundary
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y, ax=None, resolution=300, title=None):
    """Shade the regions a fitted 2-feature classifier assigns to each class.

    Works for any object with a `.predict(X)` method (sklearn models or your own).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Build a dense grid covering the data, with a small margin around it.
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    # ravel() flattens each grid to 1-D; column_stack turns them into (n_points, 2) rows.
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = np.asarray(model.predict(grid)).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    if title:
        ax.set_title(title)
    return ax


def plot_regression_fit(x, y, y_pred, ax=None, title=None):
    """Scatter 1-D data and draw the model's predictions as a line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(x)  # sort so the line is drawn left-to-right, not zig-zagging
    ax.scatter(x, y, alpha=0.6, label="data")
    ax.plot(x[order], y_pred[order], color="crimson", linewidth=2, label="model")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


def plot_learning_curve(train_sizes, train_scores, val_scores, ax=None, title=None):
    """Plot output of sklearn.model_selection.learning_curve (mean +/- std bands)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for scores, label in [(train_scores, "train"), (val_scores, "validation")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)  # axis=1 -> across CV folds
        ax.plot(train_sizes, mean, marker="o", label=label)
        ax.fill_between(train_sizes, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("training examples")
    ax.set_ylabel("score")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax
