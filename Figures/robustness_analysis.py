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

# Perturbation levels
perturbation_levels = np.linspace(0, 0.5, 100)
models = ['Ensemble', 'XGBoost', 'Random Forest', 'LightGBM', 'SVM', 'MLP']

# Colors for different perturbation types and models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Create figure
fig = plt.figure(figsize=(12, 9))
gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# (a) Performance degradation curves
perturbation_types = ['Network', 'Value', 'Temporal']
base_performance = 0.95

for i, p_type in enumerate(perturbation_types):
    # Generate degradation curve with different characteristics
    if p_type == 'Network':
        degradation = base_performance * np.exp(-2 * perturbation_levels)
    elif p_type == 'Value':
        degradation = base_performance * (1 - perturbation_levels**1.5)
    else:
        degradation = base_performance * (1 - 0.8 * perturbation_levels)
    
    # Add noise
    noise = np.random.normal(0, 0.01, len(perturbation_levels))
    degradation += noise
    
    # Plot with confidence bands
    ax1.plot(perturbation_levels, degradation, label=p_type,
             color=colors[i], alpha=0.7)
    ax1.fill_between(perturbation_levels,
                     degradation - 0.02,
                     degradation + 0.02,
                     color=colors[i], alpha=0.1)

ax1.set_xlabel('Perturbation Level')
ax1.set_ylabel('Performance (F1-Score)')
ax1.set_title('(a) Performance Degradation Curves')
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.legend()

# (b) Recovery patterns
time_points = np.linspace(0, 100, 200)
perturbation_strengths = ['Low', 'Medium', 'High']

for i, strength in enumerate(perturbation_strengths):
    # Generate recovery pattern
    if strength == 'Low':
        recovery = 0.9 - 0.2 * np.exp(-time_points/10)
    elif strength == 'Medium':
        recovery = 0.85 - 0.3 * np.exp(-time_points/20)
    else:
        recovery = 0.8 - 0.4 * np.exp(-time_points/30)
    
    # Add noise
    noise = np.random.normal(0, 0.01, len(time_points))
    recovery += noise
    
    ax2.plot(time_points, recovery, label=f'{strength} Impact',
             color=colors[i], alpha=0.7)
    ax2.fill_between(time_points,
                     recovery - 0.02,
                     recovery + 0.02,
                     color=colors[i], alpha=0.1)

ax2.set_xlabel('Recovery Time (blocks)')
ax2.set_ylabel('Performance Recovery')
ax2.set_title('(b) Recovery Patterns')
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend()

# (c) Comparative stability
stability_scores = np.array([0.92, 0.89, 0.87, 0.86, 0.84, 0.83])
stability_std = np.array([0.02, 0.025, 0.03, 0.03, 0.035, 0.035])

ax3.bar(np.arange(len(models)), stability_scores, 
        yerr=1.96*stability_std, capsize=5,
        color=colors, alpha=0.7)
ax3.set_xticks(np.arange(len(models)))
ax3.set_xticklabels(models, rotation=45, ha='right')
ax3.set_xlabel('Models')
ax3.set_ylabel('Stability Score')
ax3.set_title('(c) Comparative Stability')
ax3.grid(True, linestyle=':', alpha=0.3)

# (d) Critical point analysis
critical_points = np.linspace(0.1, 0.5, 50)
performance_impact = np.zeros((len(models), len(critical_points)))

for i, model in enumerate(models):
    base = 0.95 - i*0.02
    impact = base * (1 - critical_points**1.5)
    noise = np.random.normal(0, 0.01, len(critical_points))
    performance_impact[i] = impact + noise
    
    ax4.plot(critical_points, impact, label=model,
             color=colors[i], alpha=0.7)
    ax4.fill_between(critical_points,
                     impact - 0.02,
                     impact + 0.02,
                     color=colors[i], alpha=0.1)

# Add critical threshold line
ax4.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, 
            label='Critical Threshold')

ax4.set_xlabel('Critical Point')
ax4.set_ylabel('Performance Impact')
ax4.set_title('(d) Critical Point Analysis')
ax4.grid(True, linestyle=':', alpha=0.3)
ax4.legend(fontsize=8)

# Final adjustments
plt.tight_layout()

# Save as PDF
with PdfPages('robustness_analysis.pdf') as pdf:
    fig.set_size_inches(12, 9)
    plt.savefig(pdf, format='pdf', 
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
                metadata={
                    'Creator': 'matplotlib',
                    'Title': 'Robustness Analysis',
                    'Subject': 'Performance robustness visualization'
                })

# Also save as PNG
plt.savefig('robustness_analysis.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.1)

plt.close()