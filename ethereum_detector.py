import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                           precision_score, recall_score,
                           f1_score, classification_report)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import joblib
import argparse
import logging
import time
import gc

warnings.filterwarnings('ignore')

# Set up output directory for figures
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EnhancedEthereumDetector:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.graph = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        # Define individual models
        self.models = {
            'svm': SVC(probability=True, random_state=42),
            'mlp': MLPClassifier(random_state=42, max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42)
        }
        
        # Create ensemble model
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('svm', self.models['svm']),
                ('mlp', self.models['mlp']),
                ('rf', self.models['rf']),
                ('xgboost', self.models['xgboost']),
                ('lightgbm', self.models['lightgbm'])
            ],
            voting='soft'
        )
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def load_and_preprocess(self):
        """Load and preprocess the dataset"""
        logging.info("Loading dataset...")
        try:
            # Load CSV file
            self.df = pd.read_csv(self.file_path)
            
            # Basic validation
            if self.df.isnull().any().any():
                logging.warning("Dataset contains missing values")
            
            # Create mapping for addresses to numeric IDs
            all_addresses = pd.concat([
                self.df['from_address'],
                self.df['to_address']
            ]).unique()
            
            self.address_to_id = {addr: idx for idx, addr in enumerate(all_addresses)}
            self.id_to_address = {idx: addr for addr, idx in self.address_to_id.items()}
            
            # Create graph
            self.create_graph()
            
            logging.info(f"Original number of nodes: {len(self.address_to_id)}")
            
            # Extract features
            self.X = self.create_features()
            
            # Handle class imbalance (if labels are provided)
            if hasattr(self, 'y') and self.y is not None:
                logging.info("Applying SMOTE for class balance...")
                smote = SMOTE(random_state=42)
                self.X, self.y = smote.fit_resample(self.X, self.y)
                
                logging.info(f"Balanced dataset size: {self.X.shape[0]}")
                logging.info(f"Class distribution: {np.unique(self.y, return_counts=True)}")
            
        except Exception as e:
            logging.error(f"Error in load_and_preprocess: {str(e)}")
            raise
            
    def create_graph(self):
        """Create network graph from token transfers"""
        logging.info("Creating graph from token transfers...")
        self.graph = nx.DiGraph()
        
        # Add edges with attributes
        for _, row in self.df.iterrows():
            from_id = self.address_to_id[row['from_address']]
            to_id = self.address_to_id[row['to_address']]
            
            # Add or update edge
            if self.graph.has_edge(from_id, to_id):
                # Aggregate multiple transfers
                self.graph[from_id][to_id]['weight'] += float(row['value'])
                self.graph[from_id][to_id]['transfers'] += 1
            else:
                self.graph.add_edge(
                    from_id, 
                    to_id,
                    weight=float(row['value']),
                    transfers=1,
                    token_address=row['token_address'],
                    block_number=row['block_number']
                )
        
        logging.info(f"Graph created with {self.graph.number_of_nodes()} nodes and "
                    f"{self.graph.number_of_edges()} edges")
    
    def create_features(self):
        """Create features from graph structure"""
        logging.info("Creating network features...")
        
        features_list = []
        for node in self.graph.nodes():
            # Network centrality features
            degree_centrality = nx.degree_centrality(self.graph)[node]
            in_degree = self.graph.in_degree(node, weight='weight')
            out_degree = self.graph.out_degree(node, weight='weight')
            
            try:
                clustering_coef = nx.clustering(self.graph, node)
            except:
                clustering_coef = 0
                
            # Token-specific features
            incoming_tokens = set(
                self.graph[pred][node]['token_address'] 
                for pred in self.graph.predecessors(node)
            )
            outgoing_tokens = set(
                self.graph[node][succ]['token_address'] 
                for succ in self.graph.successors(node)
            )
            
            token_diversity = len(incoming_tokens.union(outgoing_tokens))
            
            # Transaction patterns
            incoming_transfers = sum(
                self.graph[pred][node]['transfers'] 
                for pred in self.graph.predecessors(node)
            )
            outgoing_transfers = sum(
                self.graph[node][succ]['transfers'] 
                for succ in self.graph.successors(node)
            )
            
            # Temporal features
            block_numbers = []
            for _, _, data in self.graph.in_edges(node, data=True):
                block_numbers.append(data['block_number'])
            for _, _, data in self.graph.out_edges(node, data=True):
                block_numbers.append(data['block_number'])
                
            block_number_std = np.std(block_numbers) if block_numbers else 0
            
            # Combine all features
            features = [
                degree_centrality,
                in_degree,
                out_degree,
                clustering_coef,
                token_diversity,
                incoming_transfers,
                outgoing_transfers,
                block_number_std,
                len(incoming_tokens),
                len(outgoing_tokens),
                in_degree / (out_degree + 1),  # Avoid division by zero
                incoming_transfers / (outgoing_transfers + 1)
            ]
            
            features_list.append(features)
        
        # Convert to numpy array and scale
        features_array = np.array(features_list)
        return self.scaler.fit_transform(features_array)
    
    def set_labels(self, exchange_addresses):
        """Set binary labels for exchange detection"""
        self.y = np.zeros(len(self.graph.nodes()))
        
        for addr in exchange_addresses:
            if addr in self.address_to_id:
                node_id = self.address_to_id[addr]
                self.y[node_id] = 1
    
    def train_models(self):
        """Train all models"""
        logging.info("Training models...")
        start_time = time.time()
        
        try:
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Train individual models
            for name, model in self.models.items():
                logging.info(f"Training {name}...")
                model.fit(self.X_train, self.y_train)
                
            # Train ensemble
            logging.info("Training ensemble model...")
            self.ensemble_model.fit(self.X_train, self.y_train)
            
            training_time = time.time() - start_time
            logging.info(f"Training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise
            
    def cross_validate_models(self):
        """Perform cross-validation"""
        results = {}
        
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X, self.y, cv=5)
            results[name] = {
                "mean_cv_score": np.mean(scores),
                "std_cv_score": np.std(scores)
            }
            logging.info(f"{name} CV Score: {results[name]['mean_cv_score']:.4f} "
                        f"± {results[name]['std_cv_score']:.4f}")
            
        return results
        
    def evaluate_models(self):
        """Evaluate all models"""
        results = {}
        
        try:
            # Evaluate individual models
            for name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                results[name] = self.calculate_metrics(self.y_test, y_pred, model)
                
            # Evaluate ensemble
            y_pred_ensemble = self.ensemble_model.predict(self.X_test)
            results['ensemble'] = self.calculate_metrics(
                self.y_test, y_pred_ensemble, self.ensemble_model
            )
            
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise
            
        return results
        
    def calculate_metrics(self, y_true, y_pred, model):
        """Calculate evaluation metrics"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='binary'),
            "recall": recall_score(y_true, y_pred, average='binary'),
            "f1_score": f1_score(y_true, y_pred, average='binary'),
            "roc_auc": roc_auc_score(y_true, model.predict_proba(self.X_test)[:, 1]),
            "classification_report": classification_report(y_true, y_pred)
        }
        
    def save_models(self):
        """Save trained models"""
        try:
            for name, model in self.models.items():
                path = os.path.join('models', f'{name}_model.pkl')
                joblib.dump(model, path)
                logging.info(f"Saved {name} model")
                
            # Save ensemble
            path = os.path.join('models', 'ensemble_model.pkl')
            joblib.dump(self.ensemble_model, path)
            logging.info("Saved ensemble model")
            
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            raise
            
    def cleanup(self):
        """Clean up resources"""
        gc.collect()
        plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Ethereum Token Transfer Detector')
    parser.add_argument('--file_path', type=str, required=True,
                       help='Path to the token transfers CSV file')
    args = parser.parse_args()

    try:
        detector = EnhancedEthereumDetector(args.file_path)
        
        detector.load_and_preprocess()
        
        # Example: Set labels for known exchange addresses
        exchange_addresses = [
            # Add known exchange addresses here
        ]
        detector.set_labels(exchange_addresses)
        
        detector.train_models()
        detector.cross_validate_models()
        detector.save_models()
        
        results = detector.evaluate_models()
        for model_name, metrics in results.items():
            logging.info(f"\nResults for {model_name}:")
            for metric, value in metrics.items():
                if metric != 'classification_report':
                    logging.info(f"{metric}: {value:.4f}")
                    
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
        
    finally:
        detector.cleanup()