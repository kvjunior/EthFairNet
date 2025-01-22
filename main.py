import argparse
import logging
from datetime import datetime
import os
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import gc
import pandas as pd

# Import our components
from ethereum_detector import EnhancedEthereumDetector
from eth_analysis_enhanced import EthereumExchangeDetectorEnhanced
from eth_robustness import EthereumRobustnessAnalysis
from eth_ethical import EthereumEthicalAnalysis

def setup_experiment_directory():
    """Create directory structure for experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"experiments/eth_run_{timestamp}")
    
    # Create all necessary directories
    dirs = [
        base_dir / "models",
        base_dir / "results" / "base",
        base_dir / "results" / "enhanced",
        base_dir / "results" / "robustness",
        base_dir / "results" / "ethical",
        base_dir / "figures" / "analysis",
        base_dir / "figures" / "robustness",
        base_dir / "figures" / "ethical",
        base_dir / "logs",
        base_dir / "data"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir

def setup_logging(experiment_dir):
    """Configure logging"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(experiment_dir / "logs/experiment.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_config(args, experiment_dir):
    """Save experiment configuration"""
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=4)

def load_exchange_labels(label_file):
    """Load known exchange addresses from file"""
    try:
        labels_df = pd.read_csv(label_file)
        return set(labels_df[labels_df['type'] == 'exchange']['address'])
    except Exception as e:
        logging.error(f"Error loading exchange labels: {e}")
        return set()

def analyze_dataset(df):
    """Analyze the token transfers dataset"""
    analysis = {
        'total_transactions': len(df),
        'unique_addresses': len(set(df['from_address']).union(set(df['to_address']))),
        'unique_tokens': len(df['token_address'].unique()),
        'date_range': {
            'min_block': df['block_number'].min(),
            'max_block': df['block_number'].max()
        },
        'transaction_stats': {
            'mean_value': df['value'].mean(),
            'median_value': df['value'].median(),
            'total_value': df['value'].sum()
        }
    }
    return analysis

def visualize_data_statistics(analysis, experiment_dir):
    """Create visualizations of dataset statistics"""
    plt.figure(figsize=(15, 10))
    
    # Transaction values histogram
    plt.subplot(2, 2, 1)
    plt.hist(analysis['transaction_stats']['value_distribution'], bins=50)
    plt.title('Transaction Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Token activity
    plt.subplot(2, 2, 2)
    plt.bar(range(len(analysis['token_stats'])), 
            list(analysis['token_stats'].values()))
    plt.title('Token Activity')
    plt.xlabel('Token')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(experiment_dir / "figures/analysis/data_statistics.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Ethereum Exchange Detection Analysis'
    )
    
    # Dataset arguments
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to the token transfers CSV file')
    parser.add_argument('--label_file', type=str, required=True,
                       help='Path to the exchange labels file')
    
    # Experiment arguments
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of experimental runs')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for processing')
    parser.add_argument('--min_transactions', type=int, default=5,
                       help='Minimum number of transactions for address analysis')
    parser.add_argument('--time_window', type=int, default=1000,
                       help='Block window size for temporal analysis')
    
    # W&B settings
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, 
                       default='ethereum-exchange-detection',
                       help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for tracking')

    args = parser.parse_args()

    # Setup experiment
    experiment_dir = setup_experiment_directory()
    logger = setup_logging(experiment_dir)
    save_config(args, experiment_dir)

    try:
        # Only initialize wandb if explicitly enabled
        if args.use_wandb:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=args.experiment_name,
                    config=vars(args),
                    dir=str(experiment_dir)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                args.use_wandb = False
        
        # Main experimental pipeline
        try:
            # 1. Load and analyze data
            logger.info("Loading dataset...")
            df = pd.read_csv(args.data_file)
            
            logger.info("Analyzing dataset...")
            data_analysis = analyze_dataset(df)
            if args.use_wandb:
                wandb.log({"dataset_analysis": data_analysis})
            
            logger.info("Loading exchange labels...")
            exchange_addresses = load_exchange_labels(args.label_file)
            
            # 2. Initialize detector with enhanced enhancements
            logger.info("Initializing enhanced detector...")
            detector = EthereumExchangeDetectorEnhanced(
                args.data_file,  # Changed from df to data_file
                n_runs=args.n_runs,
                n_folds=args.n_folds,
                use_wandb=args.use_wandb
            )
            
            # 3. Data preprocessing and feature engineering
            logger.info("Preprocessing data...")
            detector.load_and_preprocess()
            
            logger.info("Setting labels...")
            detector.set_labels(exchange_addresses)
            
            # Save preprocessed data
            torch.save(detector.X, experiment_dir / "data/processed_features.pt")
            
            # 4. Model optimization and training
            logger.info("Optimizing hyperparameters...")
            hyperparams = detector.hyperparameter_optimization()
            
            logger.info("Training models...")
            detector.train_models()
            
            # 5. Validation and analysis
            logger.info("Performing statistical validation...")
            statistical_results = detector.statistical_validation()
            
            logger.info("Interpreting models...")
            detector.model_interpretation()
            
            logger.info("Conducting ablation studies...")
            ablation_results = detector.ablation_study()
            
            # 6. Robustness analysis
            logger.info("Analyzing model robustness...")
            robustness_analyzer = EthereumRobustnessAnalysis(
                model=detector.ensemble_model,
                graph=detector.graph,
                X=detector.X,
                y=detector.y,
                address_to_id=detector.address_to_id,
                id_to_address=detector.id_to_address
            )
            
            robustness_results = robustness_analyzer.generate_robustness_report()
            
            # 7. Ethical analysis
            logger.info("Conducting ethical analysis...")
            ethical_analyzer = EthereumEthicalAnalysis(
                model=detector.ensemble_model,
                graph=detector.graph,
                X=detector.X,
                y=detector.y,
                address_to_id=detector.address_to_id,
                id_to_address=detector.id_to_address
            )
            
            ethical_results = ethical_analyzer.generate_ethical_report()
            
            # 8. Save all results
            logger.info("Saving results...")
            all_results = {
                'data_analysis': data_analysis,
                'statistical_results': statistical_results,
                'ablation_results': ablation_results,
                'robustness_results': robustness_results,
                'ethical_results': ethical_results,
                'hyperparameters': hyperparams
            }
            
            # Save in multiple formats
            torch.save(all_results, experiment_dir / "results/all_results.pt")
            np.save(experiment_dir / "results/all_results.npy", all_results)
            
            with open(experiment_dir / "results/all_results.json", 'w') as f:
                json.dump(str(all_results), f, indent=4)
            
            # 9. Generate final report
            logger.info("Generating final report...")
            with open(experiment_dir / "results/final_report.txt", 'w') as f:
                f.write("Ethereum Exchange Detection Analysis Report\n")
                f.write("=====================================\n\n")
                
                sections = [
                    ("Dataset Analysis", data_analysis),
                    ("Statistical Analysis", statistical_results),
                    ("Ablation Study", ablation_results),
                    ("Robustness Analysis", robustness_results),
                    ("Ethical Analysis", ethical_results),
                    ("Hyperparameters", hyperparams)
                ]
                
                for title, results in sections:
                    f.write(f"\n{title}\n")
                    f.write("-" * len(title) + "\n")
                    f.write(str(results) + "\n\n")
            
            logger.info(f"Experiment completed successfully. "
                       f"Results saved in {experiment_dir}")
            
        except Exception as e:
            logger.error(f"Error during experiment: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error initializing W&B: {str(e)}")
        raise
        
    finally:
        # Cleanup
        try:
            wandb.finish()
            plt.close('all')
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("Experiment finished. Cleaning up resources...")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()
