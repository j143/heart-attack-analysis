"""
Utility functions for heart attack analysis.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import joblib


class ConfigManager:
    """Manage configuration settings for the project."""
    
    DEFAULT_CONFIG = {
        'data_path': 'data/Heart_Attack_Analysis_Data.csv',
        'output_models_dir': 'outputs/models',
        'output_plots_dir': 'outputs/plots',
        'random_state': 42,
        'test_size': 0.2,
        'cross_validation_folds': 5,
        'correlation_threshold': 0.8
    }
    
    def __init__(self, config_path: str = 'config/config.json'):
        """
        Initialize ConfigManager.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file if it exists."""
        if os.path.exists(self.config_path):
            try:
                import json
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                self.config.update(file_config)
                print(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"Error loading config: {e}. Using default configuration.")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        import json
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        print(f"Configuration saved to {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary."""
        self.config.update(config_dict)


class Logger:
    """Simple logging utility."""
    
    def __init__(self, log_file: str = 'outputs/analysis.log'):
        """
        Initialize Logger.
        
        Args:
            log_file (str): Path to log file
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message: str, level: str = 'INFO') -> None:
        """
        Log a message.
        
        Args:
            message (str): Message to log
            level (str): Log level (INFO, WARNING, ERROR)
        """
        import datetime
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        # Print to console
        print(f"{level}: {message}")
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)


class ModelValidator:
    """Validate saved models and ensure compatibility."""
    
    @staticmethod
    def validate_model_file(filepath: str) -> bool:
        """
        Validate that a model file exists and can be loaded.
        
        Args:
            filepath (str): Path to model file
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        try:
            model = joblib.load(filepath)
            print(f"Successfully validated model: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model {filepath}: {e}")
            return False
    
    @staticmethod
    def validate_all_models(models_dir: str = 'outputs/models') -> Dict[str, bool]:
        """
        Validate all model files in the models directory.
        
        Args:
            models_dir (str): Directory containing model files
            
        Returns:
            dict: Validation results for each model
        """
        results = {}
        
        if not os.path.exists(models_dir):
            print(f"Models directory not found: {models_dir}")
            return results
        
        for filename in os.listdir(models_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(models_dir, filename)
                results[filename] = ModelValidator.validate_model_file(filepath)
        
        return results


class DataValidator:
    """Validate data quality and consistency."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a dataframe and return quality metrics.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            
        Returns:
            dict: Validation results
        """
        results = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Check for potential issues
        issues = []
        
        if results['duplicate_rows'] > 0:
            issues.append(f"Found {results['duplicate_rows']} duplicate rows")
        
        total_missing = sum(results['missing_values'].values())
        if total_missing > 0:
            issues.append(f"Found {total_missing} missing values")
        
        results['issues'] = issues
        results['is_valid'] = len(issues) == 0
        
        return results
    
    @staticmethod
    def check_target_distribution(df: pd.DataFrame, target_col: str = 'Target') -> Dict[str, Any]:
        """
        Check target variable distribution.
        
        Args:
            df (pd.DataFrame): Dataframe with target column
            target_col (str): Name of target column
            
        Returns:
            dict: Target distribution information
        """
        if target_col not in df.columns:
            return {'error': f'Target column {target_col} not found'}
        
        value_counts = df[target_col].value_counts()
        total = len(df)
        
        return {
            'value_counts': value_counts.to_dict(),
            'proportions': (value_counts / total).to_dict(),
            'is_balanced': min(value_counts) / max(value_counts) > 0.3  # Rough balance check
        }


class FileManager:
    """Manage file operations and project structure."""
    
    @staticmethod
    def ensure_directories(directories: List[str]) -> None:
        """
        Ensure that all specified directories exist.
        
        Args:
            directories (list): List of directory paths to create
        """
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory ensured: {directory}")
    
    @staticmethod
    def clean_output_directory(directory: str, file_pattern: str = '*') -> None:
        """
        Clean output directory of old files.
        
        Args:
            directory (str): Directory to clean
            file_pattern (str): Pattern of files to remove
        """
        import glob
        
        if not os.path.exists(directory):
            return
        
        pattern_path = os.path.join(directory, file_pattern)
        files = glob.glob(pattern_path)
        
        for file in files:
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    @staticmethod
    def get_project_structure() -> Dict[str, List[str]]:
        """
        Get the current project structure.
        
        Returns:
            dict: Project structure by directory
        """
        structure = {}
        
        for root, dirs, files in os.walk('.'):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            if files:  # Only include directories with files
                structure[root] = files
        
        return structure


def print_analysis_summary(results: Dict[str, Dict[str, float]], 
                         feature_names: List[str] = None) -> None:
    """
    Print a comprehensive analysis summary.
    
    Args:
        results (dict): Model evaluation results
        feature_names (list, optional): List of feature names
    """
    print("\n" + "="*80)
    print("HEART ATTACK ANALYSIS SUMMARY".center(80))
    print("="*80)
    
    print(f"\nEvaluated {len(results)} models:")
    
    # Find best model for each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    best_models = {}
    
    for metric in metrics:
        if all(metric in result for result in results.values()):
            best_model = max(results.keys(), key=lambda k: results[k][metric])
            best_score = results[best_model][metric]
            best_models[metric] = (best_model, best_score)
    
    print("\nBest Models by Metric:")
    for metric, (model, score) in best_models.items():
        print(f"  {metric.title()}: {model} ({score:.4f})")
    
    # Print detailed results
    print(f"\nDetailed Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric.title()}: {value:.4f}")
    
    if feature_names:
        print(f"\nDataset contains {len(feature_names)} features:")
        print(f"  {', '.join(feature_names)}")
    
    print("\n" + "="*80)


def create_gitignore() -> None:
    """Create a .gitignore file for the project."""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# MacOS
.DS_Store

# Project specific
outputs/models/*.pkl
outputs/plots/*.png
outputs/*.log
data/*.csv
!data/README.md
temp/
tmp/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("Created .gitignore file")