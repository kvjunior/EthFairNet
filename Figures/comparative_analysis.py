import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

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

# Colors and methods
our_color = '#d62728'  # Red for our approach
other_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
methods = ['Ours', 'Method A', 'Method B', 'Method C', 'Method D', 'Method E']
colors = [our_color if method == 'Ours' else other_colors[i] 
          for i, method in enumerate(methods[1:])]
colors.insert(0, our_color)

# (a) Performance metrics across approaches
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']
our_scores = np.array([0.9532, 0.9249, 0.9312, 0.9187, 0.9621])
other_scores = np.array([
    [0.89, 0.87, 0.88, 0.86, 0.90],  # Method A
    [0.87, 0.85, 0.86, 0.84, 0.88],  # Method B
    [0.85, 0.83, 0.84, 0.82, 0.86],  # Method C
    [0.83, 0.81, 0.82, 0.80, 0.84],  # Method D
    [0.81, 0.79, 0.80, 0.78, 0.82],  # Method E
])

x = np.arange(len(metrics))
width = 0.15
for i, scores in enumerate([our_scores] + [row for row in other_scores]):
    offset = width * (i - len(methods)/2)
    ax1.bar(x + offset, scores, width, label=methods[i], 
            color=colors[i], alpha=0.7)

ax1.set_ylabel('Score')
ax1.set_title('(a) Performance Metrics Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.3)

# (b) Fairness-accuracy trade-off
accuracy = np.linspace(0.8, 0.96, 100)
for i, method in enumerate(methods):
    if method == 'Ours':
        fairness = 0.95 - 0.1 * (accuracy - 0.95)**2
        ax2.scatter(0.9532, fairness[80], color=our_color, s=100, 
                   zorder=5, label=method)
    else:
        fairness = 0.85 - 0.2 * (accuracy - 0.85)**2 + i*0.02
        ax2.scatter(accuracy[i*15], fairness[i*15], color=colors[i], s=80,
                   label=method)
        ax2.plot(accuracy, fairness, color=colors[i], alpha=0.5)

ax2.set_xlabel('Accuracy')
ax2.set_ylabel('Fairness Score')
ax2.set_title('(b) Fairness-Accuracy Trade-off')
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend()

# (c) Privacy-utility balance
privacy_levels = np.linspace(0, 1, 100)
for i, method in enumerate(methods):
    if method == 'Ours':
        utility = 0.95 - 0.1 * privacy_levels**0.5
        ax3.plot(privacy_levels, utility, color=our_color, 
                linewidth=3, label=method)
    else:
        utility = 0.85 - (0.2 + i*0.05) * privacy_levels**0.5
        ax3.plot(privacy_levels, utility, color=colors[i], 
                alpha=0.7, label=method)

ax3.set_xlabel('Privacy Level')
ax3.set_ylabel('Utility Score')
ax3.set_title('(c) Privacy-Utility Balance')
ax3.grid(True, linestyle=':', alpha=0.3)
ax3.legend()

# (d) Computational efficiency
compute_times = [1.0, 2.5, 3.2, 4.1, 3.8, 4.5]  # Normalized times
memory_usage = [1.0, 2.2, 2.8, 3.5, 3.2, 3.8]   # Normalized memory

ax4.scatter(compute_times, memory_usage, c=colors, s=100)
for i, method in enumerate(methods):
    ax4.annotate(method, (compute_times[i], memory_usage[i]), 
                xytext=(5, 5), textcoords='offset points')

ax4.set_xlabel('Normalized Computation Time')
ax4.set_ylabel('Normalized Memory Usage')
ax4.set_title('(d) Computational Efficiency')
ax4.grid(True, linestyle=':', alpha=0.3)

# Mark our approach with bold red point
ax4.scatter(compute_times[0], memory_usage[0], c=our_color, 
           s=150, linewidth=2, edgecolor='black', zorder=5)

# Final adjustments
plt.tight_layout()

# Save as PDF
with PdfPages('comparative_analysis.pdf') as pdf:
    fig.set_size_inches(12, 9)
    plt.savefig(pdf, format='pdf', 
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
                metadata={
                    'Creator': 'matplotlib',
                    'Title': 'Comparative Analysis Results',
                    'Subject': 'Performance comparison visualization'
                })

# Also save as PNG
plt.savefig('comparative_analysis.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.1)

plt.close()