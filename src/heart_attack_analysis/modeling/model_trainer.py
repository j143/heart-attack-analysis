"""
Model training and evaluation module for heart attack analysis.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional, Any
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from systemds.context import SystemDSContext
    from systemds.operator.algorithm import multiLogReg, multiLogRegPredict, l2svm, l2svmPredict
    SYSTEMDS_AVAILABLE = True
except ImportError:
    SYSTEMDS_AVAILABLE = False
    print("SystemDS not available. Original SystemDS models will be skipped.")


class ModelTrainer:
    """Handle model training, tuning, and evaluation for heart attack analysis."""
    
    def __init__(self, output_dir: str = "outputs/models"):
        """
        Initialize ModelTrainer with output directory.
        
        Args:
            output_dir (str): Directory to save models
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.models = {}
        self.results = {}
        
    def train_systemds_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Train original SystemDS models (Logistic Regression and L2SVM).
        
        Args:
            X_train (np.ndarray): Training features (scaled)
            X_test (np.ndarray): Test features (scaled)
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Results for SystemDS models
        """
        if not SYSTEMDS_AVAILABLE:
            print("SystemDS not available. Skipping SystemDS models.")
            return {}
        
        results = {}
        
        # Train Logistic Regression with SystemDS
        print("Training SystemDS Logistic Regression...")
        with SystemDSContext() as sds:
            X_ds = sds.from_numpy(X_train)
            y_ds = sds.from_numpy(y_train + 1.0)  # SystemDS expects 1-based labels
            bias = multiLogReg(X_ds, y_ds, maxi=100, verbose=False)
            weights = bias.compute().flatten()
            
            # Evaluate on test set
            Xt_ds = sds.from_numpy(X_test)
            yt_ds = sds.from_numpy(y_test + 1.0)
            _, y_pred, acc = multiLogRegPredict(Xt_ds, bias, Y=yt_ds, verbose=False).compute()
            
            # Convert predictions back to 0-based
            y_pred_binary = (y_pred.flatten() > 1.5).astype(int)
            
            # Calculate metrics
            results['SystemDS Logistic Regression'] = {
                'accuracy': acc,
                'precision': precision_score(y_test.flatten(), y_pred_binary),
                'recall': recall_score(y_test.flatten(), y_pred_binary),
                'f1_score': f1_score(y_test.flatten(), y_pred_binary),
                'auc': 0.0  # SystemDS doesn't provide probabilities easily
            }
            
            # Save weights
            weights_path = os.path.join(self.output_dir, 'logistic_regression_weights.pkl')
            joblib.dump(weights, weights_path)
            print(f"Logistic Regression weights saved to {weights_path}")
        
        # Train L2SVM with SystemDS
        print("Training SystemDS L2SVM...")
        with SystemDSContext() as sds:
            X_ds = sds.from_numpy(X_train)
            y_ds = sds.from_numpy(y_train + 1.0)
            l2svm_model = l2svm(X_ds, y_ds, reg=0.01, maxIterations=100, verbose=False)
            l2svm_weights = l2svm_model.compute().flatten()
            
            # Evaluate on test set
            Xt_ds = sds.from_numpy(X_test)
            l2svm_y_pred_raw, l2svm_y_pred_maxed = l2svmPredict(Xt_ds, l2svm_model, verbose=False).compute()
            
            # Convert predictions back to 0-based
            l2svm_y_pred_binary = (l2svm_y_pred_maxed.flatten() > 1.5).astype(int)
            
            # Calculate accuracy manually
            l2svm_acc = np.mean((l2svm_y_pred_binary == y_test.flatten()).astype(float))
            
            results['SystemDS L2SVM'] = {
                'accuracy': l2svm_acc,
                'precision': precision_score(y_test.flatten(), l2svm_y_pred_binary),
                'recall': recall_score(y_test.flatten(), l2svm_y_pred_binary),
                'f1_score': f1_score(y_test.flatten(), l2svm_y_pred_binary),
                'auc': 0.0  # SystemDS doesn't provide probabilities easily
            }
            
            # Save weights
            weights_path = os.path.join(self.output_dir, 'l2svm_weights.pkl')
            joblib.dump(l2svm_weights, weights_path)
            print(f"L2SVM weights saved to {weights_path}")
        
        return results
    
    def tune_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """
        Tune hyperparameters for Logistic Regression using GridSearchCV.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            Pipeline: Best logistic regression model
        """
        print("Tuning Logistic Regression...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['liblinear'],
            'classifier__class_weight': [None, 'balanced']
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='accuracy', 
            verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train.flatten())
        
        print(f"Best Logistic Regression parameters: {grid_search.best_params_}")
        print(f"Best Logistic Regression CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def tune_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """
        Tune hyperparameters for SVM using GridSearchCV.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            Pipeline: Best SVM model
        """
        print("Tuning SVM...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True, random_state=42))
        ])
        
        param_grid = {
            'classifier__C': [1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale']
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='accuracy', 
            verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train.flatten())
        
        print(f"Best SVM parameters: {grid_search.best_params_}")
        print(f"Best SVM CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def tune_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """
        Tune hyperparameters for Random Forest using GridSearchCV.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            Pipeline: Best Random Forest model
        """
        print("Tuning Random Forest...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5],
            'classifier__class_weight': [None, 'balanced']
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='accuracy', 
            verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train.flatten())
        
        print(f"Best Random Forest parameters: {grid_search.best_params_}")
        print(f"Best Random Forest CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def tune_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """
        Tune hyperparameters for Gradient Boosting using GridSearchCV.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            Pipeline: Best Gradient Boosting model
        """
        print("Tuning Gradient Boosting...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
        
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 4]
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='accuracy', 
            verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train.flatten())
        
        print(f"Best Gradient Boosting parameters: {grid_search.best_params_}")
        print(f"Best Gradient Boosting CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def create_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """
        Create a voting ensemble from the best models.
        
        Args:
            models (dict): Dictionary of trained models
            
        Returns:
            VotingClassifier: Ensemble model
        """
        print("Creating Voting Ensemble...")
        
        named_estimators = [
            ('log_reg', models['logistic']),
            ('svm', models['svm']),
            ('rf', models['random_forest']),
            ('gb', models['gradient_boosting'])
        ]
        
        voting_clf = VotingClassifier(
            estimators=named_estimators,
            voting='soft'  # Use probabilities for voting
        )
        
        return voting_clf
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model and return metrics.
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob)
        }
        
        print(f"\nResults for {model_name}:")
        for metric, value in results.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return results
    
    def evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                       y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models.
        
        Args:
            models (dict): Dictionary of trained models
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Results for all models
        """
        results = {}
        
        for name, model in models.items():
            results[name] = self.evaluate_model(model, X_test, y_test, name)
        
        return results
    
    def save_model(self, model: Any, model_name: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model
            model_name (str): Name for the saved model
        """
        filepath = os.path.join(self.output_dir, f'refined_{model_name}_model.pkl')
        joblib.dump(model, filepath)
        print(f"Model saved as {filepath}")
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a previously saved model.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            Loaded model
        """
        filepath = os.path.join(self.output_dir, f'refined_{model_name}_model.pkl')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[np.ndarray]:
        """
        Get feature importance from models that support it.
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
            
        Returns:
            np.ndarray or None: Feature importances if available
        """
        # Extract the classifier from pipeline if needed
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        if hasattr(classifier, 'feature_importances_'):
            return classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            return np.abs(classifier.coef_[0])
        else:
            return None
    
    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
        """
        Train all models (both SystemDS and refined sklearn models).
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Test labels
            
        Returns:
            tuple: (models_dict, results_dict)
        """
        all_models = {}
        all_results = {}
        
        # Train SystemDS models if available
        if SYSTEMDS_AVAILABLE:
            systemds_results = self.train_systemds_models(X_train, X_test, y_train, y_test)
            all_results.update(systemds_results)
        
        # Train refined sklearn models
        print("\n=== Training Refined Models ===")
        
        refined_models = {}
        refined_models['logistic'] = self.tune_logistic_regression(X_train, y_train)
        refined_models['svm'] = self.tune_svm(X_train, y_train)
        refined_models['random_forest'] = self.tune_random_forest(X_train, y_train)
        refined_models['gradient_boosting'] = self.tune_gradient_boosting(X_train, y_train)
        
        # Create ensemble
        refined_models['ensemble'] = self.create_ensemble(refined_models)
        refined_models['ensemble'].fit(X_train, y_train.flatten())
        
        # Evaluate refined models
        refined_results = self.evaluate_models(refined_models, X_test, y_test.flatten())
        
        # Add prefix to distinguish refined models
        for name, result in refined_results.items():
            all_results[f"Refined {name.title()}"] = result
        
        # Save all refined models
        for name, model in refined_models.items():
            self.save_model(model, name)
        
        all_models.update(refined_models)
        
        return all_models, all_results