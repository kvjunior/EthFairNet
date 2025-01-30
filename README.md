# EthFairNet: A Knowledge-Based Framework for Ethereum Exchange Detection

## Overview

EthFairNet is a robust knowledge-based framework for detecting cryptocurrency exchanges on the Ethereum blockchain. It features a dynamic-weighted ensemble architecture, fairness-aware feature extraction, and adaptive feedback mechanisms to achieve high accuracy while maintaining fairness across different exchange types.

## Key Features

- Dynamic-weighted ensemble architecture with five specialized models
- Fairness-aware feature extraction pipeline
- Adaptive feedback mechanism for continuous optimization
- Privacy-preserving mechanisms with differential privacy guarantees
- Comprehensive evaluation metrics and visualization tools

## Repository Structure

```
├── Figures/                    # Visualization outputs and figures
├── datainfos.txt              # Dataset information and statistics
├── eth_analysis_enhanced.py   # Enhanced analysis implementation
├── eth_ethical.py            # Fairness and ethical analysis
├── eth_robustness.py         # Robustness testing implementation
├── ethereum_detector.py       # Core detector implementation
└── main.py                   # Main execution script

Dataset (https://snap.stanford.edu/data/ethereum-exchanges.html)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EthFairNet.git
cd EthFairNet
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main detection pipeline:
```bash
python main.py --data_file path/to/transactions.csv --label_file path/to/labels.csv
```

2. Run specific analysis:
```bash
# Enhanced analysis
python eth_analysis_enhanced.py --file_path path/to/data

# Ethical analysis
python eth_ethical.py --model_path path/to/model

# Robustness testing
python eth_robustness.py --model_path path/to/model
```

## Required Dependencies

- Python 3.8+
- PyTorch 1.12.0
- scikit-learn
- networkx
- pandas
- numpy
- matplotlib
- seaborn
- wandb (for experiment tracking)

## Performance

EthFairNet achieves:
- 95.32% ± 0.31% accuracy
- 92.49% ± 0.30% F1 score
- Disparity scores ≤ 0.0256
- Privacy guarantee (ε = 0.1)

## Model Architecture

The framework combines five specialized models:
1. Random Forest: Feature interaction handling
2. XGBoost: Imbalanced data management
3. SVM: Boundary definition
4. MLP: Non-linear pattern recognition
5. LightGBM: High-dimensional feature processing


## Acknowledgments

- Stanford SNAP for the Ethereum dataset (https://snap.stanford.edu/data/ethereum-exchanges.html)
