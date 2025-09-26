#!/usr/bin/env python3
"""
Main CLI interface for Heart Attack Analysis.

This script provides a command-line interface to run the complete heart attack analysis workflow.
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from heart_attack_analysis.data_processing.data_loader import DataLoader
from heart_attack_analysis.modeling.model_trainer import ModelTrainer
from heart_attack_analysis.visualization.plotter import Plotter
from heart_attack_analysis.utils.helpers import ConfigManager, Logger, print_analysis_summary, create_gitignore


def run_complete_analysis(config_path: str = None) -> None:
    """
    Run the complete heart attack analysis workflow.
    
    Args:
        config_path (str, optional): Path to configuration file
    """
    # Initialize components
    config = ConfigManager(config_path) if config_path else ConfigManager()
    logger = Logger()
    
    logger.log("Starting Heart Attack Analysis")
    
    # Initialize data loader
    data_loader = DataLoader(config.get('data_path'))
    
    # Load and explore data
    logger.log("Loading data")
    df = data_loader.load_data()
    
    logger.log("Exploring data")
    data_loader.explore_data()
    
    # Prepare data
    logger.log("Preparing data for modeling")
    X_train, X_test, y_train, y_test, scaler_params = data_loader.prepare_data_systemds_style(
        random_state=config.get('random_state')
    )
    
    # Save scaler
    import joblib
    scaler_path = os.path.join(config.get('output_models_dir'), 'scaler.pkl')
    os.makedirs(config.get('output_models_dir'), exist_ok=True)
    joblib.dump(scaler_params, scaler_path)
    logger.log(f"Scaler saved to {scaler_path}")
    
    # Initialize visualization
    plotter = Plotter(config.get('output_plots_dir'))
    
    # Create visualizations
    logger.log("Creating visualizations")
    plotter.plot_correlation_heatmap(df)
    plotter.plot_feature_distributions(df, data_loader.categorical_cols)
    
    # Initialize model trainer
    trainer = ModelTrainer(config.get('output_models_dir'))
    
    # Train all models
    logger.log("Training models")
    models, results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Create model comparison plots
    logger.log("Creating model comparison plots")
    plotter.plot_model_comparison_metrics(results)
    
    # Plot ROC curves for refined models
    refined_models = {k: v for k, v in models.items() if 'ensemble' not in k.lower()}
    if refined_models:
        plotter.plot_roc_curves(refined_models, X_test, y_test.flatten(), "refined_models_roc_curves.png")
    
    # Plot feature importance for models that support it
    feature_names = data_loader.get_feature_names()
    for name, model in models.items():
        importance = trainer.get_feature_importance(model, feature_names)
        if importance is not None:
            plotter.plot_feature_importance(feature_names, importance, name)
    
    # Print summary
    print_analysis_summary(results, feature_names)
    
    logger.log("Heart Attack Analysis completed successfully")


def run_data_exploration(data_path: str) -> None:
    """
    Run only data exploration and visualization.
    
    Args:
        data_path (str): Path to data file
    """
    data_loader = DataLoader(data_path)
    plotter = Plotter()
    
    # Load and explore data
    df = data_loader.load_data()
    data_loader.explore_data()
    
    # Create basic visualizations
    plotter.plot_correlation_heatmap(df)
    plotter.plot_feature_distributions(df, data_loader.categorical_cols)
    
    # Print correlation analysis
    high_corr_pairs = data_loader.find_high_correlation_pairs()
    if high_corr_pairs:
        print("\nHighly correlated feature pairs:")
        for f1, f2, corr in high_corr_pairs:
            print(f"{f1} <-> {f2}: {corr:.3f}")
    else:
        print("\nNo highly correlated feature pairs found.")


def run_model_training(data_path: str, model_type: str = 'all') -> None:
    """
    Run only model training.
    
    Args:
        data_path (str): Path to data file
        model_type (str): Type of model to train ('all', 'systemds', 'sklearn')
    """
    data_loader = DataLoader(data_path)
    trainer = ModelTrainer()
    
    # Load and prepare data
    df = data_loader.load_data()
    X_train, X_test, y_train, y_test, _ = data_loader.prepare_data_systemds_style()
    
    if model_type in ['all', 'systemds']:
        # Train SystemDS models
        systemds_results = trainer.train_systemds_models(X_train, X_test, y_train, y_test)
        print("SystemDS Models Results:")
        for name, metrics in systemds_results.items():
            print(f"{name}: Accuracy = {metrics['accuracy']:.4f}")
    
    if model_type in ['all', 'sklearn']:
        # Train sklearn models
        models = {}
        models['logistic'] = trainer.tune_logistic_regression(X_train, y_train)
        models['svm'] = trainer.tune_svm(X_train, y_train)
        models['random_forest'] = trainer.tune_random_forest(X_train, y_train)
        models['gradient_boosting'] = trainer.tune_gradient_boosting(X_train, y_train)
        models['ensemble'] = trainer.create_ensemble(models)
        models['ensemble'].fit(X_train, y_train.flatten())
        
        # Evaluate models
        results = trainer.evaluate_models(models, X_test, y_test.flatten())
        
        # Save models
        for name, model in models.items():
            trainer.save_model(model, name)


def run_model_evaluation(models_dir: str, data_path: str) -> None:
    """
    Run model evaluation on saved models.
    
    Args:
        models_dir (str): Directory containing saved models
        data_path (str): Path to data file
    """
    from heart_attack_analysis.utils.helpers import ModelValidator
    
    # Validate models
    validation_results = ModelValidator.validate_all_models(models_dir)
    
    print("Model Validation Results:")
    for model_file, is_valid in validation_results.items():
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"  {model_file}: {status}")
    
    # Load data for evaluation
    data_loader = DataLoader(data_path)
    df = data_loader.load_data()
    X_train, X_test, y_train, y_test, _ = data_loader.prepare_data_systemds_style()
    
    # Load and evaluate models
    trainer = ModelTrainer(models_dir)
    models = {}
    
    model_names = ['logistic', 'svm', 'random_forest', 'gradient_boosting', 'ensemble']
    for name in model_names:
        try:
            models[name] = trainer.load_model(name)
        except FileNotFoundError:
            print(f"Model not found: {name}")
    
    if models:
        results = trainer.evaluate_models(models, X_test, y_test.flatten())
        print_analysis_summary(results, data_loader.get_feature_names())


def setup_project() -> None:
    """Set up project structure and configuration."""
    from heart_attack_analysis.utils.helpers import FileManager
    
    directories = [
        'data',
        'outputs/models',
        'outputs/plots',
        'config',
        'tests'
    ]
    
    FileManager.ensure_directories(directories)
    create_gitignore()
    
    # Create default config
    config = ConfigManager()
    config.save_config()
    
    # Create data README
    data_readme = """# Data Directory

Place your Heart_Attack_Analysis_Data.csv file in this directory.

## Data Description

The dataset should contain the following columns:
- Age: Age of the patient
- Sex: Gender (1=male, 0=female)
- CP_Type: Chest Pain Type (1-4)
- BloodPressure: Resting blood pressure
- Cholestrol: Serum cholesterol level
- BloodSugar: Fasting blood sugar (1 if > 120 mg/dl, 0 otherwise)
- ECG: Resting electrocardiographic results (0-2)
- MaxHeartRate: Maximum heart rate achieved
- ExerciseAngia: Exercise induced angina (1=yes, 0=no)
- FamilyHistory: Number of family members with heart disease
- Target: Target variable (1=heart attack risk, 0=no risk)
"""
    
    with open('data/README.md', 'w') as f:
        f.write(data_readme)
    
    print("Project setup completed!")
    print("Please place your data file in the 'data/' directory and run:")
    print("  python main.py analyze")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Heart Attack Analysis CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Complete analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Run complete analysis')
    analyze_parser.add_argument('--config', type=str, help='Path to config file')
    
    # Data exploration command
    explore_parser = subparsers.add_parser('explore', help='Run data exploration only')
    explore_parser.add_argument('--data', type=str, default='data/Heart_Attack_Analysis_Data.csv',
                               help='Path to data file')
    
    # Model training command
    train_parser = subparsers.add_parser('train', help='Run model training only')
    train_parser.add_argument('--data', type=str, default='data/Heart_Attack_Analysis_Data.csv',
                             help='Path to data file')
    train_parser.add_argument('--type', choices=['all', 'systemds', 'sklearn'], default='all',
                             help='Type of models to train')
    
    # Model evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate saved models')
    eval_parser.add_argument('--models', type=str, default='outputs/models',
                            help='Directory containing saved models')
    eval_parser.add_argument('--data', type=str, default='data/Heart_Attack_Analysis_Data.csv',
                            help='Path to data file')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up project structure')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        run_complete_analysis(args.config)
    elif args.command == 'explore':
        run_data_exploration(args.data)
    elif args.command == 'train':
        run_model_training(args.data, args.type)
    elif args.command == 'evaluate':
        run_model_evaluation(args.models, args.data)
    elif args.command == 'setup':
        setup_project()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()