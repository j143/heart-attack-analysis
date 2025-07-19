"""
Visualization module for heart attack analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import os


class Plotter:
    """Handle all visualization tasks for heart attack analysis."""
    
    def __init__(self, output_dir: str = "outputs/plots"):
        """
        Initialize Plotter with output directory.
        
        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        sns.set(style='whitegrid', palette='muted')
        plt.style.use('default')
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> None:
        """
        Plot correlation heatmap for the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            filename (str): Output filename
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation heatmap saved to {filepath}")
    
    def plot_feature_distributions(self, df: pd.DataFrame, categorical_cols: List[str], 
                                 filename: str = "distributions_all.png") -> None:
        """
        Plot distributions for all features in a grid.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_cols (list): List of categorical column names
            filename (str): Output filename
        """
        num_features = len(df.columns)
        cols = 3
        rows = int(np.ceil(num_features / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        for idx, col in enumerate(df.columns):
            ax = axes[idx] if len(axes) > 1 else axes
            
            if col in categorical_cols:
                sns.countplot(x=col, data=df, ax=ax)
            else:
                sns.histplot(df[col], kde=True, ax=ax)
            
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
        
        # Remove any unused subplots
        for j in range(idx+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature distributions saved to {filepath}")
    
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray, 
                              model_name: str, filename: Optional[str] = None) -> None:
        """
        Plot feature importance as horizontal bar chart.
        
        Args:
            feature_names (list): List of feature names
            importances (np.ndarray): Feature importance values
            model_name (str): Name of the model
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Feature Importance for {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {filepath}")
    
    def plot_model_comparison_metrics(self, results: dict, filename: str = "model_comparison.png") -> None:
        """
        Plot comparison of multiple models across different metrics.
        
        Args:
            results (dict): Dictionary with model names as keys and metrics as values
            filename (str): Output filename
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            model_names = list(results.keys())
            metric_values = [results[model][metric] for model in model_names]
            
            # Create bar colors (blue for original, green for refined)
            colors = ['blue' if 'Original' in name else 'green' for name in model_names]
            
            bars = plt.bar(model_names, metric_values, color=colors)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
            
            plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Models')
            plt.ylabel(metric.replace("_", " ").title())
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save individual metric plots
            metric_filename = f'model_comparison_{metric}.png'
            filepath = os.path.join(self.output_dir, metric_filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Model comparison plots saved")
    
    def plot_roc_curves(self, models: dict, X_test: np.ndarray, y_test: np.ndarray, 
                       filename: str = "roc_curves.png") -> None:
        """
        Plot ROC curves for multiple models.
        
        Args:
            models (dict): Dictionary of trained models
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            filename (str): Output filename
        """
        from sklearn.metrics import roc_curve, roc_auc_score
        
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            # Get probabilities
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.4f})')
        
        # Add random guess line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
        
        # Add labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Heart Attack Prediction Models')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Save the plot
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curves plot saved to {filepath}")
    
    def plot_predictions_vs_true(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str, filename: Optional[str] = None) -> None:
        """
        Plot predictions vs true values scatter plot.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_')}_predictions_vs_true.png"
        
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_true)), y_true.flatten(), label='True Values', 
                   alpha=0.7, color='blue')
        plt.scatter(range(len(y_pred)), y_pred.flatten(), label=f'{model_name} Predictions', 
                   alpha=0.7, color='red', marker='x')
        
        plt.title(f'{model_name}: Predictions vs True Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction / Label')
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Predictions vs true values plot saved to {filepath}")
    
    def plot_feature_removal_analysis(self, results_df: pd.DataFrame, 
                                    filename: str = "feature_removal_accuracy.png") -> None:
        """
        Plot feature removal experiment results.
        
        Args:
            results_df (pd.DataFrame): Results dataframe with 'dropped' and 'accuracy' columns
            filename (str): Output filename
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(x='dropped', y='accuracy', data=results_df, palette='viridis')
        plt.xlabel('Dropped Features', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title('Test Accuracy for Each Feature Removal Experiment', fontsize=14)
        plt.ylim(70, 90)  # Zoom in on the accuracy range
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature removal analysis plot saved to {filepath}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str, filename: Optional[str] = None) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            filename (str, optional): Output filename
        """
        from sklearn.metrics import confusion_matrix
        
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Heart Attack', 'Heart Attack'],
                   yticklabels=['No Heart Attack', 'Heart Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to {filepath}")