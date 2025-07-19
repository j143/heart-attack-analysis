#!/usr/bin/env python3
"""
Legacy compatibility script for heart_attack_systemds.py

This script maintains compatibility with the original analysis while using the new package structure.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from heart_attack_analysis.data_processing.data_loader import DataLoader
from heart_attack_analysis.modeling.model_trainer import ModelTrainer
from heart_attack_analysis.visualization.plotter import Plotter
from heart_attack_analysis.utils.helpers import Logger
import itertools
import pandas as pd
import numpy as np
import joblib


def main():
    """Run the original SystemDS analysis with new structure."""
    logger = Logger()
    logger.log("Starting Heart Attack Analysis (SystemDS Style)")
    
    # Initialize components
    data_loader = DataLoader('data/Heart_Attack_Analysis_Data.csv')
    plotter = Plotter('outputs/plots')
    trainer = ModelTrainer('outputs/models')
    
    # 1. Load data
    logger.log("Loading data")
    df = data_loader.load_data()
    
    # 2. Exploratory Data Analysis (EDA)
    logger.log("Performing exploratory data analysis")
    data_loader.explore_data()
    
    # Create visualizations
    plotter.plot_correlation_heatmap(df)
    plotter.plot_feature_distributions(df, data_loader.categorical_cols)
    
    # 3. Data split and scaling (DataManager-style)
    logger.log("Preparing data with SystemDS-style approach")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler_params = data_loader.prepare_data_systemds_style()
    
    # Save the scaler
    scaler_path = 'outputs/models/scaler.pkl'
    joblib.dump(scaler_params, scaler_path)
    logger.log(f"Scaler saved as '{scaler_path}'")
    
    # Feature Selection Step
    logger.log("Performing feature selection analysis")
    high_corr_pairs = data_loader.find_high_correlation_pairs(threshold=0.8)
    
    if high_corr_pairs:
        print("\n[Feature Selection] Highly correlated feature pairs (|corr| > 0.8):")
        for f1, f2, corr in high_corr_pairs:
            print(f"{f1} <-> {f2}: correlation = {corr:.2f}")
    else:
        print("\n[Feature Selection] No highly correlated feature pairs found (|corr| > 0.8).")
    
    # Train SystemDS models
    logger.log("Training SystemDS models")
    systemds_results = trainer.train_systemds_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Feature importance analysis from SystemDS Logistic Regression
    if 'logistic_regression_weights.pkl' in os.listdir('outputs/models'):
        weights = joblib.load('outputs/models/logistic_regression_weights.pkl')
        feature_names = data_loader.get_feature_names()
        
        # Extract feature coefficients (skip bias term)
        feature_coefs = list(zip(feature_names, weights[1:]))
        feature_coefs_sorted = sorted(feature_coefs, key=lambda x: abs(x[1]))
        
        print("\n[Feature Selection] Features with lowest absolute importance (SystemDS coefficients):")
        for name, coef in feature_coefs_sorted[:3]:
            print(f"{name}: {coef:.4f}")
        
        # Print all feature importances sorted by absolute value
        feature_coefs_sorted_desc = sorted(feature_coefs, key=lambda x: abs(x[1]), reverse=True)
        print("\nSystemDS Logistic Regression Feature Importances (sorted by absolute value):")
        for name, coef in feature_coefs_sorted_desc:
            print(f"{name}: {coef:.4f}")
        
        print("\nInterpretation: Features with higher absolute coefficient values have a stronger influence on the prediction.")
        print("Positive values increase risk, negative values decrease risk.")
    
    # Systematic Feature Removal Experimentation
    logger.log("Running systematic feature removal experiments")
    investigate_features = ['Cholestrol', 'BloodPressure', 'ExerciseAngia']
    
    results = []
    for n in range(len(investigate_features) + 1):
        for drop_set in itertools.combinations(investigate_features, n):
            drop_list = list(drop_set)
            print(f"\n[Experiment] Dropping features: {drop_list if drop_list else 'None'}")
            
            # Prepare data with selected features dropped
            X_exp = df.drop(['Target'] + drop_list, axis=1).values
            feature_names_exp = df.drop(['Target'] + drop_list, axis=1).columns
            
            # Data split and scaling
            num_samples = X_exp.shape[0]
            indices = np.arange(num_samples)
            np.random.seed(42)
            np.random.shuffle(indices)
            split = int(0.8 * num_samples)
            train_idx, test_idx = indices[:split], indices[split:]
            X_train_exp, X_test_exp = X_exp[train_idx], X_exp[test_idx]
            y_train_exp, y_test_exp = y_train[train_idx], y_test[test_idx]
            
            mean_exp = X_train_exp.mean(axis=0)
            std_exp = X_train_exp.std(axis=0)
            X_train_scaled_exp = (X_train_exp - mean_exp) / std_exp
            X_test_scaled_exp = (X_test_exp - mean_exp) / std_exp
            
            # Train SystemDS model on reduced features
            if trainer.SYSTEMDS_AVAILABLE:
                from systemds.context import SystemDSContext
                from systemds.operator.algorithm import multiLogReg, multiLogRegPredict
                
                with SystemDSContext() as sds:
                    X_ds = sds.from_numpy(X_train_scaled_exp)
                    y_ds = sds.from_numpy(y_train_exp + 1.0)
                    bias = multiLogReg(X_ds, y_ds, maxi=100, verbose=False)
                    Xt_ds = sds.from_numpy(X_test_scaled_exp)
                    yt_ds = sds.from_numpy(y_test_exp + 1.0)
                    _, y_pred, acc = multiLogRegPredict(Xt_ds, bias, Y=yt_ds, verbose=False).compute()
                    print(f"Test Accuracy: {acc}")
                    results.append({'dropped': drop_list, 'accuracy': acc, 'features': list(feature_names_exp)})
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df['dropped'] = results_df['dropped'].apply(lambda x: ','.join(x) if x else 'None')
        results_df['features'] = results_df['features'].apply(lambda x: ','.join(x))
        results_path = 'data/feature_removal_results.csv'
        results_df.to_csv(results_path, index=False)
        
        print("\n[Summary of Experiments]")
        print(results_df)
        
        # Visualization of feature removal experiments
        plotter.plot_feature_removal_analysis(results_df)
    
    # Create prediction plots for original models
    if systemds_results:
        # Load models and create prediction plots
        if 'logistic_regression_weights.pkl' in os.listdir('outputs/models'):
            weights = joblib.load('outputs/models/logistic_regression_weights.pkl')
            
            # Original logistic regression prediction function
            def logistic_regression_predict(X, weights):
                bias = weights[0]
                w = weights[1:]
                scores = np.dot(X, w) + bias
                probs = 1 / (1 + np.exp(-scores))
                return (probs > 0.5).astype(int)
            
            # Get predictions on all data
            X_all = df.drop('Target', axis=1).values
            y_all = df['Target'].values
            X_all_scaled = (X_all - scaler_params['mean']) / scaler_params['std']
            
            lr_predictions = logistic_regression_predict(X_all_scaled, weights)
            plotter.plot_predictions_vs_true(y_all, lr_predictions, "Logistic Regression")
        
        if 'l2svm_weights.pkl' in os.listdir('outputs/models'):
            l2svm_weights = joblib.load('outputs/models/l2svm_weights.pkl')
            
            # Original L2SVM prediction function
            def l2svm_predict(X, weights):
                if len(weights) == X.shape[1] + 1:
                    w = weights[:-1]
                    bias = weights[-1]
                else:
                    w = weights
                    bias = 0
                scores = np.dot(X, w) + bias
                return (scores > 0).astype(int)
            
            # Get predictions on all data
            X_all = df.drop('Target', axis=1).values
            y_all = df['Target'].values
            X_all_scaled = (X_all - scaler_params['mean']) / scaler_params['std']
            
            svm_predictions = l2svm_predict(X_all_scaled, l2svm_weights)
            plotter.plot_predictions_vs_true(y_all, svm_predictions, "L2SVM")
    
    logger.log("Heart Attack Analysis (SystemDS Style) completed")
    
    # Print results summary
    if systemds_results:
        print("\nModel Comparison:")
        for model_name, metrics in systemds_results.items():
            print(f"{model_name} Test Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()