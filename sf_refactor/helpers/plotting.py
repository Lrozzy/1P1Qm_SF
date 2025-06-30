import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import mplhep as hep
import numpy as np
hep.style.use("CMS")
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

def plot_roc_curve(labels, scores, save_path):
    """Calculates and plots the ROC curve, saving it to a file.
    Also annotates TPR at FPR=1% and 10%."""
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    # Find TPR at FPR=1% and 10%
    tpr_at_1 = np.interp(0.01, fpr, tpr)*100
    tpr_at_10 = np.interp(0.10, fpr, tpr)*100

    plt.figure()
    plt.plot(fpr, tpr, color='cornflowerblue', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Annotate TPR at FPR=1% and 10% as text on the plot (bottom right, above AUC)
    textstr = (
        f'TPR@1%FPR = {tpr_at_1:.1f}%\n'
        f'TPR@10%FPR = {tpr_at_10:.1f}%'
    )
    plt.text(
        0.98, 0.20, textstr,
        transform=plt.gca().transAxes,
        fontsize=20,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: t -> bqq vs q/g jets')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Create a fake dataset
    np.random.seed(42)
    n_signal = 1000
    n_background = 1000
    # Signal: scores mostly high, background: scores mostly low
    signal_scores = np.random.normal(loc=1.5, scale=0.3, size=n_signal)
    background_scores = np.random.normal(loc=0.5, scale=0.3, size=n_background)
    # Clip scores to [0, 2]
    signal_scores = np.clip(signal_scores, 0, 2)
    background_scores = np.clip(background_scores, 0, 2)
    scores = np.concatenate([signal_scores, background_scores])
    labels = np.concatenate([np.ones(n_signal), np.zeros(n_background)])

    plot_roc_curve(labels, scores, "roc_curve_fake.png")

def plot_score_histogram(labels, scores, save_path):
    """Plots and saves a histogram of classifier scores for signal vs background."""
    scores_signal = scores[labels == 1]
    scores_background = scores[labels == 0]
    
    plt.figure()
    plt.hist(scores_background, bins=40, range=(0, 2), color='cornflowerblue', alpha=0.7, label='q/g jets')
    plt.hist(scores_signal, bins=40, range=(0, 2), histtype='step', color='darkorange', lw=2, label='t -> bqq')
    plt.xlabel('Classifier Score')
    plt.ylabel('No. of events')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()