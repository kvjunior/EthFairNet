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

class EthereumEthicalAnalysis:
    def __init__(self, model, graph, X, y, address_to_id, id_to_address):
        self.model = model
        self.graph = graph
        self.X = X
        self.y = y
        self.address_to_id = address_to_id
        self.id_to_address = id_to_address
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs('results/ethical', exist_ok=True)
        os.makedirs('figures/ethical', exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('results/ethical/ethical_analysis.log'),
                logging.StreamHandler()
            ]
        )

    def fairness_metrics(self, batch_size=1000):
        """Calculate fairness metrics specific to exchange detection"""
        try:
            metrics = {}
            total_samples = len(self.X)
            predictions = self.model.predict(self.X)
            
            # Calculate metrics across different groups
            metrics.update(self._analyze_transaction_volume_fairness())
            metrics.update(self._analyze_token_diversity_fairness())
            metrics.update(self._analyze_temporal_fairness())
            metrics.update(self._analyze_network_position_fairness())
            
            # Visualize fairness metrics
            self._plot_fairness_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in fairness metrics calculation: {str(e)}")
            return None

    def _analyze_transaction_volume_fairness(self):
        """Analyze fairness across transaction volume groups"""
        metrics = {}
        try:
            # Calculate transaction volumes
            volumes = []
            for node in self.graph.nodes():
                in_vol = sum(self.graph[pred][node].get('weight', 0) 
                           for pred in self.graph.predecessors(node))
                out_vol = sum(self.graph[node][succ].get('weight', 0) 
                            for succ in self.graph.successors(node))
                volumes.append(in_vol + out_vol)
            
            # Split into high/low volume groups
            volume_median = np.median(volumes)
            high_volume_mask = np.array(volumes) > volume_median
            
            # Get predictions
            predictions = self.model.predict(self.X)
            
            # Calculate metrics for each group
            metrics['volume_fairness'] = {
                'high_volume_accuracy': np.mean(predictions[high_volume_mask] == 
                                              self.y[high_volume_mask]),
                'low_volume_accuracy': np.mean(predictions[~high_volume_mask] == 
                                             self.y[~high_volume_mask]),
                'disparity': abs(np.mean(predictions[high_volume_mask]) - 
                               np.mean(predictions[~high_volume_mask]))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in volume fairness analysis: {str(e)}")
            return {}

    def _analyze_token_diversity_fairness(self):
        """Analyze fairness across token diversity groups"""
        metrics = {}
        try:
            # Calculate token diversity
            diversity_scores = []
            for node in self.graph.nodes():
                unique_tokens = set()
                for pred in self.graph.predecessors(node):
                    unique_tokens.add(self.graph[pred][node]['token_address'])
                for succ in self.graph.successors(node):
                    unique_tokens.add(self.graph[node][succ]['token_address'])
                diversity_scores.append(len(unique_tokens))
            
            # Split into high/low diversity groups
            diversity_median = np.median(diversity_scores)
            high_diversity_mask = np.array(diversity_scores) > diversity_median
            
            # Get predictions
            predictions = self.model.predict(self.X)
            
            # Calculate metrics
            metrics['token_diversity_fairness'] = {
                'high_diversity_accuracy': np.mean(predictions[high_diversity_mask] == 
                                                 self.y[high_diversity_mask]),
                'low_diversity_accuracy': np.mean(predictions[~high_diversity_mask] == 
                                                self.y[~high_diversity_mask]),
                'disparity': abs(np.mean(predictions[high_diversity_mask]) - 
                               np.mean(predictions[~high_diversity_mask]))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in token diversity fairness analysis: {str(e)}")
            return {}

    def _analyze_temporal_fairness(self):
        """Analyze fairness across temporal activity patterns"""
        metrics = {}
        try:
            # Calculate activity frequency
            activity_scores = []
            for node in self.graph.nodes():
                block_numbers = []
                for pred in self.graph.predecessors(node):
                    block_numbers.append(self.graph[pred][node]['block_number'])
                for succ in self.graph.successors(node):
                    block_numbers.append(self.graph[node][succ]['block_number'])
                
                activity_scores.append(len(set(block_numbers)))
            
            # Split into high/low activity groups
            activity_median = np.median(activity_scores)
            high_activity_mask = np.array(activity_scores) > activity_median
            
            # Get predictions
            predictions = self.model.predict(self.X)
            
            # Calculate metrics
            metrics['temporal_fairness'] = {
                'high_activity_accuracy': np.mean(predictions[high_activity_mask] == 
                                                self.y[high_activity_mask]),
                'low_activity_accuracy': np.mean(predictions[~high_activity_mask] == 
                                               self.y[~high_activity_mask]),
                'disparity': abs(np.mean(predictions[high_activity_mask]) - 
                               np.mean(predictions[~high_activity_mask]))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in temporal fairness analysis: {str(e)}")
            return {}

    def _analyze_network_position_fairness(self):
        """Analyze fairness across network positions"""
        metrics = {}
        try:
            # Calculate centrality measures
            centrality = nx.degree_centrality(self.graph)
            centrality_scores = [centrality[node] for node in self.graph.nodes()]
            
            # Split into central/peripheral groups
            centrality_median = np.median(centrality_scores)
            central_mask = np.array(centrality_scores) > centrality_median
            
            # Get predictions
            predictions = self.model.predict(self.X)
            
            # Calculate metrics
            metrics['network_position_fairness'] = {
                'central_accuracy': np.mean(predictions[central_mask] == 
                                         self.y[central_mask]),
                'peripheral_accuracy': np.mean(predictions[~central_mask] == 
                                            self.y[~central_mask]),
                'disparity': abs(np.mean(predictions[central_mask]) - 
                               np.mean(predictions[~central_mask]))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in network position fairness analysis: {str(e)}")
            return {}

    def _plot_fairness_metrics(self, metrics):
        """Visualize fairness metrics"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot disparities
            disparities = [metrics[k]['disparity'] 
                         for k in metrics.keys()]
            plt.subplot(2, 2, 1)
            plt.bar(range(len(disparities)), disparities)
            plt.xticks(range(len(disparities)), 
                      [k.replace('_fairness', '') for k in metrics.keys()],
                      rotation=45)
            plt.title('Fairness Disparities Across Groups')
            plt.ylabel('Disparity Score')
            
            # Plot accuracies
            plt.subplot(2, 2, 2)
            for i, (metric_name, metric_values) in enumerate(metrics.items()):
                accuracies = [v for k, v in metric_values.items() 
                            if k.endswith('accuracy')]
                plt.bar([i*2, i*2+1], accuracies)
            plt.title('Group-wise Accuracies')
            plt.ylabel('Accuracy')
            
            plt.tight_layout()
            plt.savefig('figures/ethical/fairness_metrics.png')
            plt.close()
            
            # Log to W&B
            try:
                wandb.log({"fairness_metrics": 
                          wandb.Image('figures/ethical/fairness_metrics.png')})
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"Error plotting fairness metrics: {str(e)}")

    def privacy_impact_assessment(self):
        """Assess privacy implications of exchange detection"""
        try:
            results = {
                'privacy_risk_metrics': self._analyze_privacy_risks(),
                'data_exposure_analysis': self._analyze_data_exposure(),
                'deanonymization_risks': self._analyze_deanonymization_risks()
            }
            
            # Visualize results
            self._plot_privacy_assessment(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in privacy impact assessment: {str(e)}")
            return None

    def _analyze_privacy_risks(self):
        """Analyze potential privacy risks"""
        try:
            risks = {}
            
            # Model confidence analysis
            probs = self.model.predict_proba(self.X)
            high_confidence_mask = np.max(probs, axis=1) > 0.9
            
            risks['high_confidence_predictions'] = {
                'count': np.sum(high_confidence_mask),
                'percentage': np.mean(high_confidence_mask) * 100
            }
            
            # Feature importance privacy impact
            if hasattr(self.model, 'feature_importances_'):
                risks['feature_privacy_impact'] = {
                    'high_impact_features': np.sum(
                        self.model.feature_importances_ > 0.1
                    )
                }
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error in privacy risk analysis: {str(e)}")
            return {}

    def _analyze_data_exposure(self):
        """Analyze potential data exposure through the model"""
        try:
            exposure = {}
            
            # Analyze prediction patterns
            predictions = self.model.predict(self.X)
            
            # Check for correlation between features and predictions
            for i in range(self.X.shape[1]):
                correlation = np.corrcoef(self.X[:, i], predictions)[0, 1]
                exposure[f'feature_{i}_correlation'] = correlation
            
            return exposure
            
        except Exception as e:
            self.logger.error(f"Error in data exposure analysis: {str(e)}")
            return {}

    def _analyze_deanonymization_risks(self):
        """Analyze potential deanonymization risks"""
        try:
            risks = {}
            
            # Analyze network structure vulnerability
            degrees = [d for n, d in self.graph.degree()]
            risks['network_vulnerability'] = {
                'high_degree_nodes': np.sum(np.array(degrees) > np.percentile(degrees, 90)),
                'mean_degree': np.mean(degrees),
                'max_degree': np.max(degrees)
            }
            
            # Analyze clustering patterns
            clustering_coeffs = nx.clustering(self.graph)
            risks['clustering_vulnerability'] = {
                'mean_clustering': np.mean(list(clustering_coeffs.values())),
                'high_clustering_nodes': np.sum(
                    np.array(list(clustering_coeffs.values())) > 0.5
                )
            }
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error in deanonymization risk analysis: {str(e)}")
            return {}

    def _plot_privacy_assessment(self, results):
        """Visualize privacy assessment results"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot privacy risk metrics
            plt.subplot(2, 2, 1)
            risk_metrics = results['privacy_risk_metrics']
            plt.bar(range(len(risk_metrics)), 
                   [v['count'] if isinstance(v, dict) and 'count' in v 
                    else v for v in risk_metrics.values()])
            plt.title('Privacy Risk Metrics')
            plt.xticks(range(len(risk_metrics)), 
                      list(risk_metrics.keys()), 
                      rotation=45)
            
            # Plot data exposure correlations
            plt.subplot(2, 2, 2)
            correlations = list(results['data_exposure_analysis'].values())
            plt.hist(correlations, bins=20)
            plt.title('Feature-Prediction Correlations')
            plt.xlabel('Correlation Strength')
            plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('figures/ethical/privacy_assessment.png')
            plt.close()
            
            # Log to W&B
            try:
                wandb.log({"privacy_assessment": 
                          wandb.Image('figures/ethical/privacy_assessment.png')})
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"Error plotting privacy assessment: {str(e)}")

    def generate_ethical_report(self):
        """Generate comprehensive ethical analysis report"""
        try:
            # Calculate all metrics
            fairness_results = self.fairness_metrics()
            privacy_results = self.privacy_impact_assessment()
            
            report = {
                'fairness_metrics': fairness_results,
                'privacy_assessment': privacy_results
            }
            
            # Save detailed report
            with open('results/ethical/ethical_analysis.txt', 'w') as f:
                f.write("Ethereum Exchange Detection - Ethical Analysis Report\n")
                f.write("=============================================\n\n")
                
                for section, metrics in report.items():
                    f.write(f"\n{section.upper()}\n")
                    f.write("-" * len(section) + "\n")
                    for metric, value in metrics.items():
                        f.write(f"{metric}: {value}\n")  # Fixed string literal here
            
            # Save results as numpy arrays for further analysis
            np.save('results/ethical_analysis_results.npy', report)
            
            # Log to W&B if available
            try:
                wandb.log({"ethical_analysis": report})
            except:
                pass
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating ethical report: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        plt.close('all')
        gc.collect()