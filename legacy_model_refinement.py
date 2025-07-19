#!/usr/bin/env python3
"""
Legacy compatibility script for model_refinement.py

This script maintains compatibility with the original model refinement while using the new package structure.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from heart_attack_analysis.data_processing.data_loader import DataLoader
from heart_attack_analysis.modeling.model_trainer import ModelTrainer
from heart_attack_analysis.visualization.plotter import Plotter
from heart_attack_analysis.utils.helpers import Logger, print_analysis_summary


def main():
    """Run the model refinement process using the new package structure."""
    logger = Logger()
    logger.log("Starting Model Refinement")
    
    # Initialize components
    data_loader = DataLoader('data/Heart_Attack_Analysis_Data.csv')
    trainer = ModelTrainer('outputs/models')
    plotter = Plotter('outputs/plots')
    
    # Load data
    logger.log("Loading data")
    df = data_loader.load_data()
    
    # Get feature names for later interpretation
    feature_names = data_loader.get_feature_names()
    
    # Prepare data using sklearn approach for refined models
    X_train, X_test, y_train, y_test = data_loader.prepare_data(test_size=0.2, random_state=42)
    
    # Save scaler
    data_loader.save_scaler('outputs/models/refined_scaler.pkl')
    
    logger.log("Starting model tuning with simplified parameters for demonstration...")
    
    # Train refined sklearn models
    refined_models = {}
    
    logger.log("Tuning Logistic Regression")
    refined_models['logistic'] = trainer.tune_logistic_regression(X_train, y_train)
    
    logger.log("Tuning SVM")
    refined_models['svm'] = trainer.tune_svm(X_train, y_train)
    
    logger.log("Tuning Random Forest")
    refined_models['random_forest'] = trainer.tune_random_forest(X_train, y_train)
    
    logger.log("Tuning Gradient Boosting")
    refined_models['gradient_boosting'] = trainer.tune_gradient_boosting(X_train, y_train)
    
    # Create ensemble
    logger.log("Creating ensemble model")
    refined_models['ensemble'] = trainer.create_ensemble(refined_models)
    refined_models['ensemble'].fit(X_train, y_train)
    
    # Evaluate all models
    logger.log("Evaluating refined models")
    results = trainer.evaluate_models(refined_models, X_test, y_test)
    
    # Plot ROC curves
    logger.log("Creating ROC curves")
    plotter.plot_roc_curves(refined_models, X_test, y_test, "model_refinement_roc_curves.png")
    
    # Plot feature importance for models that support it
    logger.log("Creating feature importance plots")
    for name, model in refined_models.items():
        importance = trainer.get_feature_importance(model, feature_names)
        if importance is not None:
            plotter.plot_feature_importance(feature_names, importance, name)
    
    # Find the best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    logger.log(f"Best model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    # Save all models
    for name, model in refined_models.items():
        trainer.save_model(model, name)
    
    # Print comprehensive results
    print_analysis_summary(results, feature_names)
    
    logger.log("Model refinement completed successfully")


if __name__ == "__main__":
    main()