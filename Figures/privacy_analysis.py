import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.spatial import ConvexHull

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
pareto_color = '#ff7f0e'
attack_color = '#d62728'
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# (a) Privacy-utility trade-off curves
epsilon_values = np.linspace(0.01, 1, 100)
utility = 0.95 - 0.1 * np.exp(-5 * epsilon_values)
privacy = 1 - (0.8 * epsilon_values ** 0.5)

# Generate Pareto points
n_points = 20
pareto_points = np.column_stack((
    np.random.uniform(0.85, 0.95, n_points),
    np.random.uniform(0.85, 0.95, n_points)
))

# Find Pareto frontier
hull = ConvexHull(pareto_points)
hull_points = pareto_points[hull.vertices]
hull_points = hull_points[np.argsort(hull_points[:, 0])]

ax1.scatter(pareto_points[:, 0], pareto_points[:, 1], 
           color=main_color, alpha=0.5, label='Operating Points')
ax1.plot(hull_points[:, 0], hull_points[:, 1], 
         color=pareto_color, linestyle='--', label='Pareto Frontier')

ax1.set_xlabel('Utility Score')
ax1.set_ylabel('Privacy Score')
ax1.set_title('(a) Privacy-Utility Trade-off')
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.legend()

# (b) Membership inference attack success rates
privacy_budgets = np.linspace(0.05, 1, 50)
base_success_rate = 0.5 + 0.3 * (1 - np.exp(-2 * privacy_budgets))
success_std = 0.02 * np.ones_like(privacy_budgets)

ax2.plot(privacy_budgets, base_success_rate, 
         color=attack_color, alpha=0.7, label='Attack Success')
ax2.fill_between(privacy_budgets,
                 base_success_rate - 1.96*success_std,
                 base_success_rate + 1.96*success_std,
                 color=attack_color, alpha=0.1)

# Add random guess line
ax2.axhline(y=0.5, color='gray', linestyle='--', 
            alpha=0.7, label='Random Guess')

ax2.set_xlabel('Privacy Budget (ε)')
ax2.set_ylabel('Attack Success Rate')
ax2.set_title('(b) Membership Inference Attack Success')
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend()

# (c) Information leakage quantification
n_features = 6
feature_names = ['Network\nMetrics', 'Token\nDiversity', 'Temporal\nPatterns',
                'Volume', 'Centrality', 'Clustering']
leakage_scores = np.array([0.15, 0.12, 0.10, 0.08, 0.06, 0.05])
leakage_std = np.array([0.02, 0.015, 0.012, 0.01, 0.008, 0.005])

bars = ax3.bar(feature_names, leakage_scores, 
               color=main_color, alpha=0.7)
ax3.errorbar(feature_names, leakage_scores, 
             yerr=1.96*leakage_std,
             fmt='none', color='black', capsize=5)

# Add critical leakage threshold
ax3.axhline(y=0.2, color=attack_color, linestyle='--', 
            alpha=0.7, label='Critical Threshold')

ax3.set_ylabel('Information Leakage (bits)')
ax3.set_title('(c) Information Leakage Quantification')
ax3.grid(True, linestyle=':', alpha=0.3)
ax3.legend()

# (d) Feature importance privacy impact
importance_levels = np.linspace(0, 1, 100)
privacy_impact = np.zeros((3, len(importance_levels)))
labels = ['High Risk', 'Medium Risk', 'Low Risk']

for i, (label, base) in enumerate(zip(labels, [0.8, 0.6, 0.4])):
    impact = base * (1 - importance_levels**2) + 0.2
    noise = np.random.normal(0, 0.01, len(importance_levels))
    privacy_impact[i] = impact + noise
    
    ax4.plot(importance_levels, impact, 
             label=label, color=colors[i], alpha=0.7)
    ax4.fill_between(importance_levels,
                     impact - 0.02,
                     impact + 0.02,
                     color=colors[i], alpha=0.1)

ax4.set_xlabel('Feature Importance')
ax4.set_ylabel('Privacy Score')
ax4.set_title('(d) Feature Importance Privacy Impact')
ax4.grid(True, linestyle=':', alpha=0.3)
ax4.legend()

# Final adjustments
plt.tight_layout()

# Save as PDF
with PdfPages('privacy_analysis.pdf') as pdf:
    fig.set_size_inches(12, 9)
    plt.savefig(pdf, format='pdf', 
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
                metadata={
                    'Creator': 'matplotlib',
                    'Title': 'Privacy Analysis Results',
                    'Subject': 'Privacy metrics visualization'
                })

# Also save as PNG
plt.savefig('privacy_analysis.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.1)

plt.close()