import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import logging
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wandb
from scipy import stats
import pandas as pd
import networkx as nx
from collections import defaultdict

class EthereumRobustnessAnalysis:
    def __init__(self, model, graph, X, y, address_to_id, id_to_address):
        self.model = model
        self.graph = graph
        self.X = X
        self.y = y
        self.address_to_id = address_to_id
        self.id_to_address = id_to_address
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs('figures/robustness', exist_ok=True)
        os.makedirs('results/robustness', exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('results/robustness/robustness_analysis.log'),
                logging.StreamHandler()
            ]
        )

    def network_perturbation_test(self, perturbation_levels=[0.1, 0.2, 0.3, 0.4]):
        """Test model robustness against network structure perturbations"""
        logging.info("Testing network perturbation robustness...")
        results = {}
        
        try:
            original_pred = self.model.predict(self.X)
            original_score = f1_score(self.y, original_pred, average='binary')
            results['original'] = original_score
            
            for level in tqdm(perturbation_levels):
                # Create perturbed graph
                perturbed_graph = self.graph.copy()
                edges_to_remove = int(perturbed_graph.number_of_edges() * level)
                
                # Randomly remove edges
                edges = list(perturbed_graph.edges())
                removed_edges = np.random.choice(
                    len(edges), 
                    edges_to_remove, 
                    replace=False
                )
                for idx in removed_edges:
                    perturbed_graph.remove_edge(*edges[idx])
                
                # Generate features from perturbed graph
                perturbed_X = self._generate_features(perturbed_graph)
                perturbed_pred = self.model.predict(perturbed_X)
                
                results[f'level_{level}'] = {
                    'f1_score': f1_score(self.y, perturbed_pred, average='binary'),
                    'accuracy': accuracy_score(self.y, perturbed_pred),
                    'prediction_change': np.mean(original_pred != perturbed_pred)
                }
                
                # Log to W&B
                wandb.log({
                    f'network_perturbation_{level}_f1': results[f'level_{level}']['f1_score'],
                    f'network_perturbation_{level}_change': results[f'level_{level}']['prediction_change']
                })
            
            # Plot results
            self._plot_network_perturbation_results(results, perturbation_levels)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in network perturbation test: {str(e)}")
            return None

    def _generate_features(self, graph):
        """Generate features from a graph"""
        features_list = []
        
        for node in graph.nodes():
            # Network centrality features
            degree_centrality = nx.degree_centrality(graph)[node]
            in_degree = graph.in_degree(node, weight='weight')
            out_degree = graph.out_degree(node, weight='weight')
            
            try:
                clustering_coef = nx.clustering(graph, node)
            except:
                clustering_coef = 0
            
            # Token features
            in_tokens = set(graph[pred][node]['token_address'] 
                          for pred in graph.predecessors(node))
            out_tokens = set(graph[node][succ]['token_address'] 
                           for succ in graph.successors(node))
            
            token_diversity = len(in_tokens.union(out_tokens))
            
            # Transaction patterns
            in_transfers = sum(graph[pred][node].get('transfers', 0) 
                             for pred in graph.predecessors(node))
            out_transfers = sum(graph[node][succ].get('transfers', 0) 
                              for succ in graph.successors(node))
            
            features = [
                degree_centrality,
                in_degree,
                out_degree,
                clustering_coef,
                token_diversity,
                in_transfers,
                out_transfers,
                len(in_tokens),
                len(out_tokens),
                in_degree / (out_degree + 1)
            ]
            
            features_list.append(features)
        
        return np.array(features_list)

    def _plot_network_perturbation_results(self, results, perturbation_levels):
        """Plot network perturbation test results"""
        plt.figure(figsize=(12, 6))
        
        f1_scores = [results[f'level_{level}']['f1_score'] 
                    for level in perturbation_levels]
        changes = [results[f'level_{level}']['prediction_change'] 
                  for level in perturbation_levels]
        
        plt.plot(perturbation_levels, f1_scores, 'bo-', label='F1 Score')
        plt.plot(perturbation_levels, changes, 'ro-', label='Prediction Changes')
        
        plt.xlabel('Perturbation Level')
        plt.ylabel('Score')
        plt.title('Model Robustness to Network Perturbations')
        plt.legend()
        plt.grid(True)
        plt.savefig('figures/robustness/network_perturbation.png')
        plt.close()
        
        # Log to W&B
        wandb.log({"network_perturbation_plot": 
                  wandb.Image('figures/robustness/network_perturbation.png')})

    def token_value_noise_test(self, noise_levels=[0.1, 0.2, 0.3, 0.4]):
        """Test model robustness against token value perturbations"""
        logging.info("Testing token value noise robustness...")
        results = {}
        
        try:
            original_pred = self.model.predict(self.X)
            original_score = f1_score(self.y, original_pred, average='binary')
            results['original'] = original_score
            
            for noise in noise_levels:
                # Create noisy graph
                noisy_graph = self.graph.copy()
                
                # Add noise to edge weights (token values)
                for u, v, data in noisy_graph.edges(data=True):
                    if 'weight' in data:
                        data['weight'] *= (1 + np.random.normal(0, noise))
                
                # Generate features from noisy graph
                noisy_X = self._generate_features(noisy_graph)
                noisy_pred = self.model.predict(noisy_X)
                
                results[f'noise_{noise}'] = {
                    'f1_score': f1_score(self.y, noisy_pred, average='binary'),
                    'accuracy': accuracy_score(self.y, noisy_pred),
                    'prediction_change': np.mean(original_pred != noisy_pred)
                }
                
                # Log to W&B
                wandb.log({
                    f'token_value_noise_{noise}_f1': results[f'noise_{noise}']['f1_score'],
                    f'token_value_noise_{noise}_change': results[f'noise_{noise}']['prediction_change']
                })
            
            # Plot results
            self._plot_token_value_noise_results(results, noise_levels)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in token value noise test: {str(e)}")
            return None

    def _plot_token_value_noise_results(self, results, noise_levels):
        """Plot token value noise test results"""
        plt.figure(figsize=(12, 6))
        
        f1_scores = [results[f'noise_{level}']['f1_score'] 
                    for level in noise_levels]
        changes = [results[f'noise_{level}']['prediction_change'] 
                  for level in noise_levels]
        
        plt.plot(noise_levels, f1_scores, 'bo-', label='F1 Score')
        plt.plot(noise_levels, changes, 'ro-', label='Prediction Changes')
        
        plt.xlabel('Noise Level')
        plt.ylabel('Score')
        plt.title('Model Robustness to Token Value Noise')
        plt.legend()
        plt.grid(True)
        plt.savefig('figures/robustness/token_value_noise.png')
        plt.close()
        
        # Log to W&B
        wandb.log({"token_value_noise_plot": 
                  wandb.Image('figures/robustness/token_value_noise.png')})

    def temporal_stability_test(self, window_sizes=[100, 500, 1000, 5000]):
        """Test model stability across different time windows"""
        logging.info("Testing temporal stability...")
        results = {}
        
        try:
            # Sort edges by block number
            edges_by_block = defaultdict(list)
            for u, v, data in self.graph.edges(data=True):
                if 'block_number' in data:
                    edges_by_block[data['block_number']].append((u, v, data))
            
            block_numbers = sorted(edges_by_block.keys())
            
            for window in window_sizes:
                window_scores = []
                
                # Test model on different time windows
                for start_idx in range(0, len(block_numbers), window):
                    end_idx = start_idx + window
                    if end_idx > len(block_numbers):
                        break
                    
                    # Create window-specific graph
                    window_graph = nx.DiGraph()
                    for block in block_numbers[start_idx:end_idx]:
                        for u, v, data in edges_by_block[block]:
                            window_graph.add_edge(u, v, **data)
                    
                    # Generate features and predict
                    window_X = self._generate_features(window_graph)
                    window_pred = self.model.predict(window_X)
                    
                    window_scores.append(
                        f1_score(self.y, window_pred, average='binary')
                    )
                
                results[f'window_{window}'] = {
                    'mean_f1': np.mean(window_scores),
                    'std_f1': np.std(window_scores),
                    'stability': 1 - np.std(window_scores)
                }
                
                # Log to W&B
                wandb.log({
                    f'temporal_stability_{window}_mean_f1': results[f'window_{window}']['mean_f1'],
                    f'temporal_stability_{window}_stability': results[f'window_{window}']['stability']
                })
            
            # Plot results
            self._plot_temporal_stability_results(results, window_sizes)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in temporal stability test: {str(e)}")
            return None

    def _plot_temporal_stability_results(self, results, window_sizes):
        """Plot temporal stability test results"""
        plt.figure(figsize=(12, 6))
        
        mean_scores = [results[f'window_{size}']['mean_f1'] 
                      for size in window_sizes]
        stabilities = [results[f'window_{size}']['stability'] 
                      for size in window_sizes]
        
        plt.plot(window_sizes, mean_scores, 'bo-', label='Mean F1 Score')
        plt.plot(window_sizes, stabilities, 'ro-', label='Stability Score')
        
        plt.xlabel('Window Size (blocks)')
        plt.ylabel('Score')
        plt.title('Model Temporal Stability')
        plt.legend()
        plt.grid(True)
        plt.savefig('figures/robustness/temporal_stability.png')
        plt.close()
        
        # Log to W&B
        wandb.log({"temporal_stability_plot": 
                  wandb.Image('figures/robustness/temporal_stability.png')})

    def generate_robustness_report(self):
        """Generate comprehensive robustness analysis report"""
        logging.info("Generating robustness report...")
        try:
            # Run all tests
            network_results = self.network_perturbation_test()
            value_results = self.token_value_noise_test()
            temporal_results = self.temporal_stability_test()
            
            report = {
                'network_perturbation': network_results,
                'token_value_noise': value_results,
                'temporal_stability': temporal_results
            }
            
            # Save detailed report
            with open('results/robustness/robustness_analysis.txt', 'w') as f:
                f.write("Ethereum Exchange Detection - Robustness Analysis Report\n")
                f.write("================================================\n\n")
                
                for section, results in report.items():
                    f.write(f"\n{section.upper()}\n")
                    f.write("-" * len(section) + "\n")
                    f.write(str(results) + "\n")
            
            # Save results as numpy array
            np.save('results/robustness/robustness_results.npy', report)
            
            # Log to W&B
            wandb.log({"robustness_analysis": report})
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating robustness report: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        plt.close('all')
        gc.collect()