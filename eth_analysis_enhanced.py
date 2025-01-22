import numpy as np
import logging
from tqdm import tqdm
import os
import gc
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import wandb
import networkx as nx
from collections import defaultdict
from ethereum_detector import EnhancedEthereumDetector

# Try to import tensorboard, but don't fail if it's not available
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("Tensorboard not available. Some logging features will be disabled.")

class EthereumExchangeDetectorEnhanced(EnhancedEthereumDetector):
    def __init__(self, file_path, n_runs=10, n_folds=5, use_wandb=False):
        super().__init__(file_path)
        self.n_runs = n_runs
        self.n_folds = n_folds
        self.use_wandb = use_wandb
        
        # Only initialize tensorboard if available
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter('runs/ethereum_exchange_detection')
        else:
            self.writer = None
        
        self.statistical_results = {}
        
        # Create necessary directories
        os.makedirs('figures/Enhanced', exist_ok=True)
        os.makedirs('results/Enhanced', exist_ok=True)
        
        # Initialize W&B only if explicitly enabled
        if self.use_wandb:
            try:
                wandb.init(
                    project="ethereum-exchange-detection",
                    config={
                        "architecture": "ensemble",
                        "n_runs": n_runs,
                        "n_folds": n_folds,
                        "dataset": file_path
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('results/Enhanced/Enhanced_analysis.log'),
                logging.StreamHandler()
            ]
        )
    def hyperparameter_optimization(self):
        """Systematic hyperparameter optimization for Ethereum exchange detection"""
        logging.info("Starting hyperparameter optimization...")
        
        param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'class_weight': ['balanced', None]
            },
            'mlp': {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100, 50)],
                'learning_rate': ['constant', 'adaptive'],
                'early_stopping': [True]
            },
            'xgboost': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3, 5]
            },
            'lightgbm': {
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'min_child_samples': [20, 50]
            }
        }
        
        optimization_results = {}
        
        for name, model in self.models.items():
            logging.info(f"Optimizing {name}...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring={
                    'f1': make_scorer(f1_score, average='binary'),
                    'mcc': make_scorer(matthews_corrcoef)
                },
                refit='f1',
                cv=RepeatedStratifiedKFold(n_splits=self.n_folds, n_repeats=3),
                n_jobs=-1,
                verbose=1
            )
            
            try:
                grid_search.fit(self.X_train, self.y_train)
                self.models[name] = grid_search.best_estimator_
                
                # Store and log results
                optimization_results[name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                # Log to W&B
                wandb.log({
                    f"{name}_best_params": grid_search.best_params_,
                    f"{name}_best_score": grid_search.best_score_
                })
                
                # Optimization curves
                self._plot_optimization_curves(name, grid_search.cv_results_)
                
            except Exception as e:
                logging.error(f"Error optimizing {name}: {str(e)}")
            
            finally:
                gc.collect()
        
        return optimization_results

    def _plot_optimization_curves(self, model_name, cv_results):
        """Plot optimization curves"""
        plt.figure(figsize=(12, 6))
        
        # Plot F1 scores
        plt.subplot(1, 2, 1)
        plt.plot(cv_results['mean_test_f1'], label='Mean F1')
        plt.fill_between(
            range(len(cv_results['mean_test_f1'])),
            cv_results['mean_test_f1'] - cv_results['std_test_f1'],
            cv_results['mean_test_f1'] + cv_results['std_test_f1'],
            alpha=0.3
        )
        plt.title(f'{model_name} F1 Score Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('F1 Score')
        plt.legend()
        
        # Plot MCC scores
        plt.subplot(1, 2, 2)
        plt.plot(cv_results['mean_test_mcc'], label='Mean MCC')
        plt.fill_between(
            range(len(cv_results['mean_test_mcc'])),
            cv_results['mean_test_mcc'] - cv_results['std_test_mcc'],
            cv_results['mean_test_mcc'] + cv_results['std_test_mcc'],
            alpha=0.3
        )
        plt.title(f'{model_name} MCC Score Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('MCC Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'figures/Enhanced/{model_name}_optimization.png')
        plt.close()

    def statistical_validation(self):
        """Perform statistical validation focused on exchange detection"""
        logging.info("Performing statistical validation...")
        model_scores = {name: [] for name in self.models.keys()}
        model_scores['ensemble'] = []
        
        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_folds,
            n_repeats=self.n_runs,
            random_state=42
        )
        
        for run_idx, (train_idx, test_idx) in enumerate(rskf.split(self.X, self.y)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = {
                        'f1': f1_score(y_test, y_pred, average='binary'),
                        'mcc': matthews_corrcoef(y_test, y_pred)
                    }
                    model_scores[name].append(score)
                    
                    # Log to W&B
                    wandb.log({
                        f"{name}_f1_run_{run_idx}": score['f1'],
                        f"{name}_mcc_run_{run_idx}": score['mcc'],
                        "run": run_idx
                    })
                    
                except Exception as e:
                    logging.error(f"Error in statistical validation for {name}: {str(e)}")
            
            # Evaluate ensemble
            try:
                self.ensemble_model.fit(X_train, y_train)
                y_pred = self.ensemble_model.predict(X_test)
                score = {
                    'f1': f1_score(y_test, y_pred, average='binary'),
                    'mcc': matthews_corrcoef(y_test, y_pred)
                }
                model_scores['ensemble'].append(score)
                
            except Exception as e:
                logging.error(f"Error in ensemble validation: {str(e)}")
            
            gc.collect()
        
        # Statistical tests
        self.statistical_results = self._perform_statistical_tests(model_scores)
        return self.statistical_results

    def _perform_statistical_tests(self, model_scores):
        """Perform statistical tests on model performance"""
        results = {}
        
        for name1 in model_scores:
            for name2 in model_scores:
                if name1 < name2:
                    try:
                        # F1 score comparison
                        f1_scores1 = [s['f1'] for s in model_scores[name1]]
                        f1_scores2 = [s['f1'] for s in model_scores[name2]]
                        f1_stat, f1_pval = stats.wilcoxon(f1_scores1, f1_scores2)
                        
                        # MCC comparison
                        mcc_scores1 = [s['mcc'] for s in model_scores[name1]]
                        mcc_scores2 = [s['mcc'] for s in model_scores[name2]]
                        mcc_stat, mcc_pval = stats.wilcoxon(mcc_scores1, mcc_scores2)
                        
                        results[f"{name1}_vs_{name2}"] = {
                            'f1_statistic': f1_stat,
                            'f1_p_value': f1_pval,
                            'f1_significant': f1_pval < 0.05,
                            'mcc_statistic': mcc_stat,
                            'mcc_p_value': mcc_pval,
                            'mcc_significant': mcc_pval < 0.05
                        }
                        
                    except Exception as e:
                        logging.error(f"Error in statistical test: {str(e)}")
        
        return results

    def model_interpretation(self):
        """Advanced model interpretation for exchange detection"""
        logging.info("Starting model interpretation...")
        try:
            # Use subset for SHAP analysis if dataset is large
            max_shap_samples = 1000
            if len(self.X_test) > max_shap_samples:
                indices = np.random.choice(len(self.X_test), max_shap_samples, replace=False)
                X_shap = self.X_test[indices]
            else:
                X_shap = self.X_test

            # SHAP analysis for tree models
            for model_name in ['rf', 'xgboost', 'lightgbm']:
                self._analyze_feature_importance(model_name, X_shap)
            
            # Network structure analysis
            self._analyze_network_structure()
            
            # Token flow patterns
            self._analyze_token_flows()
            
        except Exception as e:
            logging.error(f"Error in model interpretation: {str(e)}")

    def _analyze_feature_importance(self, model_name, X_shap):
        """Analyze feature importance using SHAP"""
        try:
            explainer = shap.TreeExplainer(self.models[model_name])
            shap_values = explainer.shap_values(X_shap)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values[1] if isinstance(shap_values, list) else shap_values,
                X_shap,
                show=False
            )
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'figures/Enhanced/shap_{model_name}.png')
            plt.close()
            
            wandb.log({f"shap_plot_{model_name}": 
                      wandb.Image(f'figures/Enhanced/shap_{model_name}.png')})
            
        except Exception as e:
            logging.error(f"Error in feature importance analysis: {str(e)}")

    def _analyze_network_structure(self):
        """Analyze network structure of exchanges vs non-exchanges"""
        try:
            # Calculate network metrics for each group
            exchange_metrics = defaultdict(list)
            non_exchange_metrics = defaultdict(list)
            
            for node in self.graph.nodes():
                metrics = {
                    'degree': self.graph.degree(node),
                    'in_degree': self.graph.in_degree(node),
                    'out_degree': self.graph.out_degree(node),
                    'clustering': nx.clustering(self.graph, node)
                }
                
                if self.y[node] == 1:
                    for k, v in metrics.items():
                        exchange_metrics[k].append(v)
                else:
                    for k, v in metrics.items():
                        non_exchange_metrics[k].append(v)
            
            # Plot comparisons
            self._plot_network_metrics(exchange_metrics, non_exchange_metrics)
            
        except Exception as e:
            logging.error(f"Error in network structure analysis: {str(e)}")

    def _plot_network_metrics(self, exchange_metrics, non_exchange_metrics):
        """Plot comparison of network metrics"""
        plt.figure(figsize=(15, 10))
        metrics = list(exchange_metrics.keys())
        
        for idx, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, idx)
            plt.boxplot([
                exchange_metrics[metric],
                non_exchange_metrics[metric]
            ], labels=['Exchanges', 'Non-Exchanges'])
            plt.title(f'{metric.title()} Distribution')
            plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('figures/Enhanced/network_metrics.png')
        plt.close()

    def _analyze_token_flows(self):
        """Analyze token flow patterns"""
        try:
            exchange_patterns = defaultdict(list)
            non_exchange_patterns = defaultdict(list)
            
            for node in self.graph.nodes():
                # Calculate token flow metrics
                in_tokens = set(self.graph[pred][node]['token_address'] 
                              for pred in self.graph.predecessors(node))
                out_tokens = set(self.graph[node][succ]['token_address'] 
                               for succ in self.graph.successors(node))
                
                metrics = {
                    'unique_tokens': len(in_tokens.union(out_tokens)),
                    'token_overlap': len(in_tokens.intersection(out_tokens)),
                    'token_ratio': len(in_tokens) / (len(out_tokens) + 1),
                    'flow_volume': sum(self.graph[pred][node]['weight']
                                     for pred in self.graph.predecessors(node))
                }
                
                if self.y[node] == 1:
                    for k, v in metrics.items():
                        exchange_patterns[k].append(v)
                else:
                    for k, v in metrics.items():
                        non_exchange_patterns[k].append(v)
            
            # Plot token flow patterns
            self._plot_token_patterns(exchange_patterns, non_exchange_patterns)
            
        except Exception as e:
            logging.error(f"Error in token flow analysis: {str(e)}")

    def _plot_token_patterns(self, exchange_patterns, non_exchange_patterns):
        """Plot token flow patterns comparison"""
        plt.figure(figsize=(15, 10))
        metrics = list(exchange_patterns.keys())
        
        for idx, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, idx)
            plt.boxplot([
                exchange_patterns[metric],
                non_exchange_patterns[metric]
            ], labels=['Exchanges', 'Non-Exchanges'])
            plt.title(f'{metric.replace("_", " ").title()} Distribution')
            plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('figures/Enhanced/token_patterns.png')
        plt.close()
        
        # Log to W&B
        wandb.log({
            "token_patterns": wandb.Image('figures/Enhanced/token_patterns.png')
        })

    def ablation_study(self):
        """Perform ablation studies on feature groups"""
        logging.info("Starting ablation study...")
        feature_groups = {
            'network': slice(0, 4),     # Basic network metrics
            'token': slice(4, 8),       # Token-related features
            'temporal': slice(8, 12)    # Temporal features
        }
        
        results = {'base': self.evaluate_models()['ensemble']['f1_score']}
        
        for group_name, feature_slice in feature_groups.items():
            logging.info(f"Testing without {group_name} features...")
            try:
                # Create copy of data without current feature group
                X_temp = self.X.copy()
                X_temp[:, feature_slice] = 0
                
                # Train and evaluate
                self.X = X_temp
                score = self.evaluate_models()['ensemble']['f1_score']
                results[f'without_{group_name}'] = score
                
                # Log to W&B
                wandb.log({
                    f'ablation_{group_name}': score,
                    'feature_group': group_name
                })
                
            except Exception as e:
                logging.error(f"Error in ablation study for {group_name}: {str(e)}")
            
            finally:
                gc.collect()
        
        # Restore original features
        self.X = self.features
        return results

    def save_experimental_results(self):
        """Save comprehensive experimental results"""
        logging.info("Saving experimental results...")
        try:
            results = {
                'statistical_tests': self.statistical_results,
                'ablation_study': self.ablation_study(),
                'cross_validation': self.cross_validate_models(),
                'hyperparameters': {name: model.get_params() 
                                  for name, model in self.models.items()},
                'evaluation_metrics': self.evaluate_models()
            }
            
            # Save to file
            torch.save(results, 'results/Enhanced/experimental_results.pt')
            wandb.save('results/Enhanced/experimental_results.pt')
            
            # Generate report
            with open('results/Enhanced/experimental_report.txt', 'w') as f:
                f.write(f"Ethereum Exchange Detection Experiment Results\n{'='*50}\n")
                for key, value in results.items():
                    f.write(f"\n{key}:\n{'-'*30}\n{value}\n")
            
            logging.info("Results saved successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error saving experimental results: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup resources"""
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb:
            try:
                wandb.finish()
            except:
                pass
        gc.collect()
        plt.close('all')