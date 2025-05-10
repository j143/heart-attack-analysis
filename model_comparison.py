import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_refined_models():
    """Load the refined models"""
    models = {}
    
    try:
        # Try to load all possible refined models
        model_names = ['logistic', 'svm', 'random_forest', 'gradient_boosting', 'ensemble']
        
        for name in model_names:
            try:
                model_path = f'refined_{name}_model.pkl'
                models[name] = joblib.load(model_path)
                print(f"Successfully loaded {model_path}")
            except FileNotFoundError:
                print(f"Model {model_path} not found")
    except Exception as e:
        print(f"Error loading models: {e}")
    
    return models

def compare_with_original_models(X_test, y_test):
    """Compare refined models with original models"""
    print("\n=== Comparing Original and Refined Models ===")
    
    # Load original models
    try:
        # Load original logistic regression weights
        log_reg_weights = joblib.load('logistic_regression_weights.pkl')
        
        # Load original L2SVM weights
        l2svm_weights = joblib.load('l2svm_weights.pkl')
        
        # Load scaler parameters
        scaler_params = joblib.load('scaler.pkl')
        mean = scaler_params['mean']
        std = scaler_params['std']
        
        # Scale test data using original scaler
        X_test_scaled = (X_test - mean) / std
        
        # Original logistic regression prediction function
        def logistic_regression_predict(X, weights):
            # For logistic regression, the bias is the first element
            bias = weights[0]
            w = weights[1:]
            scores = np.dot(X, w) + bias
            probs = 1 / (1 + np.exp(-scores))
            return (probs > 0.5).astype(int), probs
        
        # Original L2SVM prediction function
        def l2svm_predict(X, weights):
            # For L2SVM, if the weights include a bias term (it's the last element)
            if len(weights) == X.shape[1] + 1:
                w = weights[:-1]
                bias = weights[-1]
            else:
                w = weights
                bias = 0
            scores = np.dot(X, w) + bias
            return (scores > 0).astype(int), 1 / (1 + np.exp(-scores))  # Approximating probabilities
        
        # Get predictions from original models
        y_pred_log_reg, y_prob_log_reg = logistic_regression_predict(X_test_scaled, log_reg_weights)
        y_pred_l2svm, y_prob_l2svm = l2svm_predict(X_test_scaled, l2svm_weights)
        
        # Calculate metrics for original models
        original_results = {
            'Original Logistic Regression': {
                'accuracy': accuracy_score(y_test, y_pred_log_reg),
                'precision': precision_score(y_test, y_pred_log_reg),
                'recall': recall_score(y_test, y_pred_log_reg),
                'f1_score': f1_score(y_test, y_pred_log_reg),
                'auc': roc_auc_score(y_test, y_prob_log_reg)
            },
            'Original L2SVM': {
                'accuracy': accuracy_score(y_test, y_pred_l2svm),
                'precision': precision_score(y_test, y_pred_l2svm),
                'recall': recall_score(y_test, y_pred_l2svm),
                'f1_score': f1_score(y_test, y_pred_l2svm),
                'auc': roc_auc_score(y_test, y_prob_l2svm)
            }
        }
        
        # Print original model results
        for name, metrics in original_results.items():
            print(f"\nResults for {name}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"ROC AUC: {metrics['auc']:.4f}")
        
        return original_results
    
    except Exception as e:
        print(f"Error comparing with original models: {e}")
        return {}

def evaluate_refined_models(models, X_test, y_test):
    """Evaluate refined models on test data"""
    print("\n=== Evaluating Refined Models ===")
    
    results = {}
    
    for name, model in models.items():
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results[f"Refined {name}"] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }
            
            # Print results
            print(f"\nResults for Refined {name}:")
            print(f"Accuracy: {results[f'Refined {name}']['accuracy']:.4f}")
            print(f"Precision: {results[f'Refined {name}']['precision']:.4f}")
            print(f"Recall: {results[f'Refined {name}']['recall']:.4f}")
            print(f"F1 Score: {results[f'Refined {name}']['f1_score']:.4f}")
            print(f"ROC AUC: {results[f'Refined {name}']['auc']:.4f}")
        
        except Exception as e:
            print(f"Error evaluating refined model {name}: {e}")
    
    return results

def plot_comparison(original_results, refined_results):
    """Create comparison plots between original and refined models"""
    # Combine results
    all_results = {**original_results, **refined_results}
    
    # Get metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    # Create bar plot for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        model_names = list(all_results.keys())
        metric_values = [all_results[model][metric] for model in model_names]
        
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
        
        # Save the plot
        plt.savefig(f'model_comparison_{metric}.png')
        plt.close()
    
    print("Comparison plots saved as model_comparison_*.png files")

def main():
    """Main function to evaluate and compare models"""
    # Load the data
    print("Loading data...")
    df = pd.read_csv('Heart_Attack_Analysis_Data.csv')
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
    
    # Load refined models
    refined_models = load_refined_models()
    
    # Compare with original models
    original_results = compare_with_original_models(X_test, y_test)
    
    # Evaluate refined models
    refined_results = evaluate_refined_models(refined_models, X_test, y_test)
    
    # Plot comparison
    plot_comparison(original_results, refined_results)

if __name__ == "__main__":
    main()
