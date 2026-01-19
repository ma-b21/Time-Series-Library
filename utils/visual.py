import matplotlib.pyplot as plt
import numpy as np
import os

def plot_predictions_vs_targets(predictions: np.ndarray,
                                targets: np.ndarray,
                                title: str = "Predictions vs Targets",
                                folder_path: str = "results") -> None:
    """
    Plot model predictions vs targets from sliding-window forecasts.

    Args:
        predictions: np.ndarray, shape [N, pred_len, 1] or [N, pred_len]
        targets: np.ndarray, shape [N, pred_len, 1] or [N, pred_len]
        title: Plot title
        folder_path: Save directory
    """

    # -------- 1. Sanity check & squeeze --------
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.ndim == 3:
        predictions = predictions[:, :, 0]
    if targets.ndim == 3:
        targets = targets[:, :, 0]

    # -------- 2. One-step aligned reconstruction --------
    # Use the first prediction step of each window
    pred_ts = predictions[:, 0]
    true_ts = targets[:, 0]

    assert pred_ts.shape == true_ts.shape, "Prediction and target shape mismatch"

    # -------- 3. Plot --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time series plot
    ax1.plot(pred_ts, label='Predictions', alpha=0.7, linewidth=1)
    ax1.plot(true_ts, label='Targets', alpha=0.7, linewidth=1)
    ax1.set_title(f"{title} - One-step Forecast")
    ax1.set_ylabel("Value")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2.scatter(true_ts, pred_ts, alpha=0.5, s=20)

    min_val = min(true_ts.min(), pred_ts.min())
    max_val = max(true_ts.max(), pred_ts.max())
    ax2.plot([min_val, max_val], [min_val, max_val],
             'r--', label='Perfect Prediction')

    ax2.set_xlabel("Target")
    ax2.set_ylabel("Prediction")
    ax2.set_title("Scatter Plot")
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(folder_path, exist_ok=True)
    path = os.path.join(folder_path, "predictions_vs_targets.png")
    plt.savefig(path, dpi=150)
    plt.close()
