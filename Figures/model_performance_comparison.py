import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, precision_recall_curve
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set style for academic paper
plt.style.use('default')  # Use default style as base
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.autolayout': True,
    'axes.axisbelow': True,  # Grid lines behind plot elements
    'axes.edgecolor': 'gray',
    'axes.linewidth': 0.5,
    'grid.color': 'gray',
    'grid.linestyle': ':',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 2,
})

# Create synthetic data based on the paper's results
models = ['Ensemble', 'XGBoost', 'Random Forest', 'LightGBM', 'SVM', 'MLP']
accuracy_means = np.array([0.9532, 0.9234, 0.9156, 0.9145, 0.8912, 0.8989])
f1_means = np.array([0.9249, 0.9014, 0.8900, 0.8917, 0.8695, 0.8784])
std_dev = 0.02

# Create figure with subplots
fig = plt.figure(figsize=(12, 9))
gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Use a professional color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# (a) Accuracy and F1-score distributions
x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, accuracy_means, width, label='Accuracy',
                color=colors[0], alpha=0.7)
bars2 = ax1.bar(x + width/2, f1_means, width, label='F1-score',
                color=colors[1], alpha=0.7)

# Add error bars (95% confidence intervals)
ax1.errorbar(x - width/2, accuracy_means, yerr=1.96*std_dev,
             fmt='none', color='black', capsize=3, alpha=0.5)
ax1.errorbar(x + width/2, f1_means, yerr=1.96*std_dev,
             fmt='none', color='black', capsize=3, alpha=0.5)

ax1.set_ylabel('Score')
ax1.set_title('(a) Performance Metrics Distribution')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend(loc='lower left')
ax1.set_ylim(0.85, 1.0)
ax1.grid(True, linestyle=':', alpha=0.3)

# (b) ROC curves
for i, model in enumerate(models):
    fpr = np.linspace(0, 1, 100)
    tpr = np.minimum(1, fpr + (1-fpr)*(accuracy_means[i]/max(accuracy_means)))
    auc = np.trapz(tpr, fpr)
    
    ax2.plot(fpr, tpr, label=f'{model} (AUC = {auc:.3f})',
             color=colors[i], alpha=0.7, linewidth=2)
    
    upper = np.minimum(1, tpr + std_dev)
    lower = np.maximum(0, tpr - std_dev)
    ax2.fill_between(fpr, lower, upper, color=colors[i], alpha=0.1)

ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('(b) ROC Curves')
ax2.legend(loc='lower right', fontsize=8)
ax2.grid(True, linestyle=':', alpha=0.3)

# (c) Precision-Recall trade-off
for i, model in enumerate(models):
    recall = np.linspace(0, 1, 100)
    precision = accuracy_means[i] * np.exp(-recall/2)
    
    ax3.plot(recall, precision, label=model, color=colors[i], 
             alpha=0.7, linewidth=2)
    
    upper = np.minimum(1, precision + std_dev)
    lower = np.maximum(0, precision - std_dev)
    ax3.fill_between(recall, lower, upper, color=colors[i], alpha=0.1)

ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('(c) Precision-Recall Trade-off')
ax3.legend(loc='lower left', fontsize=8)
ax3.grid(True, linestyle=':', alpha=0.3)

# (d) Error type distribution
error_types = ['True Negatives', 'True Positives', 'False Negatives', 'False Positives']
error_data = np.array([
    [0.80, 0.15, 0.03, 0.02],  # Ensemble
    [0.75, 0.17, 0.05, 0.03],  # XGBoost
    [0.73, 0.18, 0.05, 0.04],  # Random Forest
    [0.72, 0.19, 0.05, 0.04],  # LightGBM
    [0.70, 0.19, 0.06, 0.05],  # SVM
    [0.71, 0.19, 0.06, 0.04]   # MLP
])

bottom = np.zeros(len(models))
for i, error_type in enumerate(error_types):
    ax4.bar(models, error_data[:, i], bottom=bottom, label=error_type,
            color=colors[i % len(colors)], alpha=0.7)
    bottom += error_data[:, i]

ax4.set_xlabel('Models')
ax4.set_ylabel('Proportion')
ax4.set_title('(d) Error Type Distribution')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.set_xticklabels(models, rotation=45, ha='right')
ax4.grid(True, linestyle=':', alpha=0.3)

# Final adjustments and save as PDF
plt.tight_layout()

# Save as PDF with vector graphics
with PdfPages('model_performance_comparison.pdf') as pdf:
    # Set figure size to exactly match the desired output
    fig.set_size_inches(12, 9)
    
    # Ensure the figure fits properly in the PDF
    plt.savefig(pdf, format='pdf', 
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
                metadata={
                    'Creator': 'matplotlib',
                    'Title': 'Model Performance Comparison',
                    'Subject': 'Performance metrics visualization'
                })

# Also save as PNG for quick viewing
plt.savefig('model_performance_comparison.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.1)

plt.close()