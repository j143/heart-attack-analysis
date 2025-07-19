#!/usr/bin/env python3
"""
Demonstration of the new heart attack analysis package structure.

This script shows how to use the reorganized codebase programmatically.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from heart_attack_analysis.data_processing.data_loader import DataLoader
from heart_attack_analysis.modeling.model_trainer import ModelTrainer
from heart_attack_analysis.visualization.plotter import Plotter
from heart_attack_analysis.utils.helpers import ConfigManager, Logger, print_analysis_summary


def demonstrate_new_structure():
    """Demonstrate the capabilities of the new package structure."""
    
    print("="*80)
    print("HEART ATTACK ANALYSIS - NEW PACKAGE STRUCTURE DEMONSTRATION")
    print("="*80)
    
    # 1. Configuration Management
    print("\n1. Configuration Management")
    print("-" * 30)
    config = ConfigManager()
    print(f"Data path: {config.get('data_path')}")
    print(f"Random state: {config.get('random_state')}")
    print(f"Test size: {config.get('test_size')}")
    
    # 2. Logging System
    print("\n2. Logging System")
    print("-" * 20)
    logger = Logger('outputs/demo.log')
    logger.log("Starting demonstration")
    
    # 3. Data Processing
    print("\n3. Data Processing Module")
    print("-" * 30)
    data_loader = DataLoader(config.get('data_path'))
    
    # Load data
    df = data_loader.load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Check data quality
    from heart_attack_analysis.utils.helpers import DataValidator
    validation_results = DataValidator.validate_dataframe(df)
    print(f"Data validation: {'✓ Valid' if validation_results['is_valid'] else '✗ Issues found'}")
    
    # Analyze correlations
    high_corr = data_loader.find_high_correlation_pairs()
    print(f"High correlation pairs: {len(high_corr)}")
    
    # 4. Data Preparation
    print("\n4. Data Preparation")
    print("-" * 20)
    X_train, X_test, y_train, y_test = data_loader.prepare_data(
        test_size=config.get('test_size'),
        random_state=config.get('random_state')
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 5. Visualization
    print("\n5. Visualization Module")
    print("-" * 25)
    plotter = Plotter(config.get('output_plots_dir'))
    
    # Create sample visualization
    plotter.plot_correlation_heatmap(df, "demo_correlation.png")
    print("✓ Correlation heatmap created")
    
    # 6. Model Training (Quick Demo)
    print("\n6. Model Training Module")
    print("-" * 27)
    trainer = ModelTrainer(config.get('output_models_dir'))
    
    # Train a simple model for demonstration
    print("Training a quick Random Forest model...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Create a simple model
    simple_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    simple_model.fit(X_train, y_train)
    
    # Evaluate
    results = trainer.evaluate_model(simple_model, X_test, y_test, "Demo Random Forest")
    
    # 7. Feature Importance
    print("\n7. Feature Analysis")
    print("-" * 20)
    feature_names = data_loader.get_feature_names()
    importance = trainer.get_feature_importance(simple_model, feature_names)
    
    if importance is not None:
        plotter.plot_feature_importance(feature_names, importance, "Demo Model")
        print("✓ Feature importance plot created")
    
    # 8. Utilities Demonstration
    print("\n8. Utilities")
    print("-" * 12)
    from heart_attack_analysis.utils.helpers import ModelValidator, FileManager
    
    # Show project structure
    structure = FileManager.get_project_structure()
    print(f"Project directories: {len(structure)}")
    
    # 9. Summary
    print("\n9. Analysis Summary")
    print("-" * 20)
    demo_results = {"Demo Random Forest": results}
    print_analysis_summary(demo_results, feature_names)
    
    logger.log("Demonstration completed successfully")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Benefits of the New Structure:")
    print("• Modular design with clear separation of concerns")
    print("• Easy to import and use components independently")
    print("• Professional Python package standards")
    print("• Comprehensive configuration management")
    print("• Built-in logging and validation")
    print("• Backward compatibility with legacy scripts")
    print("• CLI interface for easy usage")
    print("• Enhanced error handling and documentation")


if __name__ == '__main__':
    demonstrate_new_structure()