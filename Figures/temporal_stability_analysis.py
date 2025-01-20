import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import seaborn as sns

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

# Time points
time_points = np.linspace(0, 100, 200)
window_sizes = [100, 500, 1000, 5000]

# Generate F1-score evolution data
base_f1 = 0.92
f1_trend = base_f1 + 0.03 * np.sin(time_points/20) + 0.02 * np.sin(time_points/10)
noise = np.random.normal(0, 0.01, len(time_points))
f1_scores = f1_trend + noise

# Generate autocorrelation data
lags = np.arange(0, 50)
autocorr = np.array([1.0 * np.exp(-lag/10) for lag in lags])

# Generate recovery time data
recovery_times = np.random.lognormal(3.5, 0.3, 1000)

# Generate stability scores
window_stability = np.array([0.94, 0.92, 0.89, 0.87])
stability_std = np.array([0.01, 0.015, 0.02, 0.025])

# Create figure
fig = plt.figure(figsize=(12, 9))
gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Colors
main_color = '#1f77b4'
band_color = '#1f77b4'
theoretical_color = '#ff7f0e'

# (a) F1-score evolution
ax1.plot(time_points, f1_scores, color=main_color, alpha=0.7, label='F1-score')

# Add confidence bands
std_dev = 0.02
ax1.fill_between(time_points, 
                 f1_scores - 1.96*std_dev,
                 f1_scores + 1.96*std_dev,
                 color=band_color, alpha=0.2, label='95% CI')

ax1.set_xlabel('Time (blocks ×1000)')
ax1.set_ylabel('F1-Score')
ax1.set_title('(a) F1-score Evolution Over Time')
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.legend()

# Theoretical upper bound
ax1.axhline(y=base_f1 + 0.05, color=theoretical_color, linestyle='--', 
            alpha=0.5, label='Theoretical bound')

# (b) Autocorrelation
ax2.plot(lags, autocorr, color=main_color, alpha=0.7, label='Autocorrelation')
theoretical_autocorr = np.exp(-lags/8)
ax2.plot(lags, theoretical_autocorr, color=theoretical_color, linestyle='--',
         alpha=0.5, label='Theoretical decay')

ax2.set_xlabel('Time Lag (blocks ×1000)')
ax2.set_ylabel('Autocorrelation')
ax2.set_title('(b) Performance Auto-correlation')
ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend()

# (c) Recovery time distribution
ax3.hist(recovery_times, bins=30, density=True, alpha=0.7, color=main_color,
         label='Observed')

# Add theoretical distribution
x_range = np.linspace(0, max(recovery_times), 100)
theoretical_dist = stats.lognorm.pdf(x_range, s=0.3, scale=np.exp(3.5))
ax3.plot(x_range, theoretical_dist, color=theoretical_color, linestyle='--',
         alpha=0.7, label='Theoretical')

ax3.set_xlabel('Recovery Time (blocks)')
ax3.set_ylabel('Density')
ax3.set_title('(c) Recovery Time Distribution')
ax3.grid(True, linestyle=':', alpha=0.3)
ax3.legend()

# (d) Stability scores across window sizes
ax4.errorbar(window_sizes, window_stability, yerr=1.96*stability_std,
             fmt='o-', color=main_color, capsize=5, alpha=0.7,
             label='Observed stability')

# Add theoretical stability bound
theoretical_stability = 0.95 * np.exp(-np.array(window_sizes)/10000)
ax4.plot(window_sizes, theoretical_stability, color=theoretical_color,
         linestyle='--', alpha=0.5, label='Theoretical bound')

ax4.set_xlabel('Window Size (blocks)')
ax4.set_ylabel('Stability Score')
ax4.set_title('(d) Stability Score Comparison')
ax4.grid(True, linestyle=':', alpha=0.3)
ax4.legend()

# Final adjustments
plt.tight_layout()

# Save as PDF
with PdfPages('temporal_stability_analysis.pdf') as pdf:
    fig.set_size_inches(12, 9)
    plt.savefig(pdf, format='pdf', 
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
                metadata={
                    'Creator': 'matplotlib',
                    'Title': 'Temporal Stability Analysis',
                    'Subject': 'Performance stability visualization'
                })

# Also save as PNG
plt.savefig('temporal_stability_analysis.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.1)

plt.close()