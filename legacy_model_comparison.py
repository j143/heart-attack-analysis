#!/usr/bin/env python3
"""
Legacy compatibility script for model_comparison.py

This script maintains compatibility with the original model comparison while using the new package structure.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from heart_attack_analysis.data_processing.data_loader import DataLoader
from heart_attack_analysis.modeling.model_trainer import ModelTrainer
from heart_attack_analysis.visualization.plotter import Plotter
from heart_attack_analysis.utils.helpers import Logger, print_analysis_summary
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def load_original_models(X_test, y_test):
    """Load and evaluate original SystemDS models."""
    original_results = {}
    
    try:
        # Load original logistic regression weights
        log_reg_weights = joblib.load('outputs/models/logistic_regression_weights.pkl')
        
        # Load original L2SVM weights
        l2svm_weights = joblib.load('outputs/models/l2svm_weights.pkl')
        
        # Load scaler parameters
        scaler_params = joblib.load('outputs/models/scaler.pkl')
        mean = scaler_params['mean']
        std = scaler_params['std']
        
        # Scale test data using original scaler
        X_test_scaled = (X_test - mean) / std
        
        # Original logistic regression prediction function
        def logistic_regression_predict(X, weights):
            bias = weights[0]
            w = weights[1:]
            scores = np.dot(X, w) + bias
            probs = 1 / (1 + np.exp(-scores))
            return (probs > 0.5).astype(int), probs
        
        # Original L2SVM prediction function
        def l2svm_predict(X, weights):
            if len(weights) == X.shape[1] + 1:
                w = weights[:-1]
                bias = weights[-1]
            else:
                w = weights
                bias = 0
            scores = np.dot(X, w) + bias
            return (scores > 0).astype(int), 1 / (1 + np.exp(-scores))
        
        # Get predictions from original models
        y_pred_log_reg, y_prob_log_reg = logistic_regression_predict(X_test_scaled, log_reg_weights)
        y_pred_l2svm, y_prob_l2svm = l2svm_predict(X_test_scaled, l2svm_weights)
        
        # Calculate metrics for original models
        original_results['Original Logistic Regression'] = {
            'accuracy': accuracy_score(y_test, y_pred_log_reg),
            'precision': precision_score(y_test, y_pred_log_reg),
            'recall': recall_score(y_test, y_pred_log_reg),
            'f1_score': f1_score(y_test, y_pred_log_reg),
            'auc': roc_auc_score(y_test, y_prob_log_reg)
        }
        
        original_results['Original L2SVM'] = {
            'accuracy': accuracy_score(y_test, y_pred_l2svm),
            'precision': precision_score(y_test, y_pred_l2svm),
            'recall': recall_score(y_test, y_pred_l2svm),
            'f1_score': f1_score(y_test, y_pred_l2svm),
            'auc': roc_auc_score(y_test, y_prob_l2svm)
        }
        
        # Print original model results
        print("\n=== Original Models Results ===")
        for name, metrics in original_results.items():
            print(f"\nResults for {name}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"ROC AUC: {metrics['auc']:.4f}")
        
    except Exception as e:
        print(f"Error loading original models: {e}")
    
    return original_results


def load_refined_models():
    """Load the refined models."""
    models = {}
    
    try:
        model_names = ['logistic', 'svm', 'random_forest', 'gradient_boosting', 'ensemble']
        
        for name in model_names:
            try:
                model_path = f'outputs/models/refined_{name}_model.pkl'
                models[name] = joblib.load(model_path)
                print(f"Successfully loaded {model_path}")
            except FileNotFoundError:
                print(f"Model {model_path} not found")
    except Exception as e:
        print(f"Error loading refined models: {e}")
    
    return models


def evaluate_refined_models(models, X_test, y_test):
    """Evaluate refined models on test data."""
    print("\n=== Evaluating Refined Models ===")
    
    results = {}
    
    for name, model in models.items():
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results[f"Refined {name.title()}"] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }
            
            # Print results
            print(f"\nResults for Refined {name.title()}:")
            print(f"Accuracy: {results[f'Refined {name.title()}']['accuracy']:.4f}")
            print(f"Precision: {results[f'Refined {name.title()}']['precision']:.4f}")
            print(f"Recall: {results[f'Refined {name.title()}']['recall']:.4f}")
            print(f"F1 Score: {results[f'Refined {name.title()}']['f1_score']:.4f}")
            print(f"ROC AUC: {results[f'Refined {name.title()}']['auc']:.4f}")
        
        except Exception as e:
            print(f"Error evaluating refined model {name}: {e}")
    
    return results


def main():
    """Main function to evaluate and compare models."""
    logger = Logger()
    logger.log("Starting Model Comparison")
    
    # Initialize components
    data_loader = DataLoader('data/Heart_Attack_Analysis_Data.csv')
    plotter = Plotter('outputs/plots')
    
    # Load the data
    print("Loading data...")
    df = data_loader.load_data()
    X = df.drop('Target', axis=1).values
    y = df['Target'].values
    
    # Split the data the same way as in the original code
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * num_samples)
    train_idx, test_idx = indices[:split], indices[split:]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    # Load and evaluate original models
    logger.log("Loading and evaluating original SystemDS models")
    original_results = load_original_models(X_test, y_test)
    
    # Load refined models
    logger.log("Loading refined models")
    refined_models = load_refined_models()
    
    # Evaluate refined models
    logger.log("Evaluating refined models")
    refined_results = evaluate_refined_models(refined_models, X_test, y_test)
    
    # Combine all results
    all_results = {**original_results, **refined_results}
    
    # Create comparison plots
    if all_results:
        logger.log("Creating comparison plots")
        plotter.plot_model_comparison_metrics(all_results)
        
        # Print comprehensive comparison
        print_analysis_summary(all_results, data_loader.get_feature_names())
    
    logger.log("Model comparison completed")


if __name__ == "__main__":
    main()