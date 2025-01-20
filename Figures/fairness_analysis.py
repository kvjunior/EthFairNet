import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# Set style for academic paper
plt.style.use('default')
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
    'axes.axisbelow': True,
    'axes.edgecolor': 'gray',
    'axes.linewidth': 0.5,
    'grid.color': 'gray',
    'grid.linestyle': ':',
    'lines.linewidth': 2,
})

# Create synthetic data
np.random.seed(42)

# Create figure
fig = plt.figure(figsize=(12, 9))
gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Colors
main_color = '#1f77b4'
threshold_color = '#d62728'
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# (a) Disparity scores across user groups
user_groups = ['Volume\nBased', 'Token\nDiversity', 'Network\nPosition', 'Temporal\nActivity']
disparity_scores = np.array([0.0123, 0.0122, 0.0256, 0.0144])
confidence_intervals = np.array([0.0018, 0.0017, 0.0021, 0.0019])

bars = ax1.bar(user_groups, disparity_scores, color=main_color, alpha=0.7)
ax1.errorbar(user_groups, disparity_scores, yerr=1.96*confidence_intervals,
             fmt='none', color='black', capsize=5)

# Add regulatory threshold
ax1.axhline(y=0.05, color=threshold_color, linestyle='--', alpha=0.7,
            label='Regulatory Threshold')

ax1.set_ylabel('Disparity Score')
ax1.set_title('(a) Disparity Scores Across Groups')
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.legend()

# (b) Performance distribution across transaction volumes
volume_ranges = np.linspace(0, 100, 50)
performance = 0.92 + 0.05 * np.exp(-volume_ranges/20) - 0.02 * np.exp(-volume_ranges/50)
performance_std = 0.02 * np.ones_like(volume_ranges)

ax2.plot(volume_ranges, performance, color=main_color, alpha=0.7,
         label='Performance')
ax2.fill_between(volume_ranges,
                 performance - 1.96*performance_std,
                 performance + 1.96*performance_std,
                 color=main_color, alpha=0.1)

# Add fairness threshold
ax2.axhline(y=0.90, color=threshold_color, linestyle='--', alpha=0.7,
            label='Fairness Threshold')

ax2.set_xlabel('Transaction Volume Percentile')
ax2.set_ylabel('Performance')
ax2.set_title('(b) Performance vs Transaction Volume')
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend()

# (c) Token diversity impact
diversity_levels = np.array([1, 2, 3, 4, 5, 6])
fairness_scores = np.array([0.91, 0.93, 0.94, 0.93, 0.92, 0.91])
fairness_std = np.array([0.02, 0.015, 0.01, 0.015, 0.02, 0.025])

ax3.plot(diversity_levels, fairness_scores, 'o-', color=main_color,
         alpha=0.7, label='Fairness Score')
ax3.fill_between(diversity_levels,
                 fairness_scores - 1.96*fairness_std,
                 fairness_scores + 1.96*fairness_std,
                 color=main_color, alpha=0.1)

# Add threshold
ax3.axhline(y=0.90, color=threshold_color, linestyle='--', alpha=0.7,
            label='Minimum Required')

ax3.set_xlabel('Token Diversity Level')
ax3.set_ylabel('Fairness Score')
ax3.set_title('(c) Token Diversity Impact')
ax3.grid(True, linestyle=':', alpha=0.3)
ax3.legend()

# (d) Network position bias
positions = ['Core', 'Semi-\nPeripheral', 'Peripheral']
bias_metrics = {
    'Accuracy': [0.95, 0.93, 0.91],
    'Precision': [0.94, 0.92, 0.90],
    'Recall': [0.93, 0.91, 0.89]
}

x = np.arange(len(positions))
width = 0.25
multiplier = 0

for attribute, measurement in bias_metrics.items():
    offset = width * multiplier
    rects = ax4.bar(x + offset, measurement, width, label=attribute,
                    color=colors[multiplier], alpha=0.7)
    multiplier += 1

# Add threshold
ax4.axhline(y=0.90, color=threshold_color, linestyle='--', alpha=0.7,
            label='Fairness Threshold')

ax4.set_ylabel('Score')
ax4.set_title('(d) Network Position Bias')
ax4.set_xticks(x + width)
ax4.set_xticklabels(positions)
ax4.grid(True, linestyle=':', alpha=0.3)
ax4.legend()

# Final adjustments
plt.tight_layout()

# Save as PDF
with PdfPages('fairness_analysis.pdf') as pdf:
    fig.set_size_inches(12, 9)
    plt.savefig(pdf, format='pdf', 
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
                metadata={
                    'Creator': 'matplotlib',
                    'Title': 'Multi-dimensional Fairness Analysis',
                    'Subject': 'Fairness metrics visualization'
                })

# Also save as PNG
plt.savefig('fairness_analysis.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.1)

plt.close()