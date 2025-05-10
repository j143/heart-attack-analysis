import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load and prepare the heart attack data"""
    print("Loading and preparing data...")
    df = pd.read_csv('Heart_Attack_Analysis_Data.csv')
    X = df.drop('Target', axis=1).values
    y = df['Target'].values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return df, X_train, X_test, y_train, y_test

def tune_logistic_regression(X_train, y_train):
    """Tune hyperparameters for Logistic Regression using GridSearchCV"""
    print("\n=== Tuning Logistic Regression ===")
    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Define the parameter grid
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
        'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'classifier__class_weight': [None, 'balanced']
    }
    
    # Create a restricted grid to avoid incompatible combinations
    valid_param_grid = []
    
    # Create a list of valid parameter combinations manually
    valid_param_grid = []
    
    # Handle solver-specific penalties
    for solver in param_grid['classifier__solver']:
        for penalty in param_grid['classifier__penalty']:
            # Skip invalid combinations
            if solver in ['newton-cg', 'lbfgs', 'sag'] and penalty == 'l1':
                continue
            if solver == 'liblinear' and penalty == 'elasticnet':
                continue
            if penalty == 'elasticnet' and solver != 'saga':
                continue
            if penalty is None and solver != 'lbfgs':
                continue
                
            for c in param_grid['classifier__C']:
                for weight in param_grid['classifier__class_weight']:
                    valid_param_grid.append({
                        'classifier__C': [c],  # Wrap single values in a list
                        'classifier__penalty': [penalty],
                        'classifier__solver': [solver],
                        'classifier__class_weight': [weight]
                    })
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create the grid search
    grid_search = GridSearchCV(
        pipeline, 
        valid_param_grid, 
        cv=cv, 
        scoring='accuracy', 
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_svm(X_train, y_train):
    """Tune hyperparameters for SVM using RandomizedSearchCV"""
    print("\n=== Tuning SVM ===")
    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True))
    ])
    
    # Define the parameter grid
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf', 'poly'],
        'classifier__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'classifier__class_weight': [None, 'balanced'],
        'classifier__degree': [2, 3, 4]  # For polynomial kernel
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create the randomized search (more efficient than grid search for large parameter spaces)
    random_search = RandomizedSearchCV(
        pipeline, 
        param_grid, 
        n_iter=30,  # Number of parameter combinations to try
        cv=cv, 
        scoring='accuracy', 
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the randomized search
    random_search.fit(X_train, y_train)
    
    # Print the best parameters
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_random_forest(X_train, y_train):
    """Tune hyperparameters for Random Forest using RandomizedSearchCV"""
    print("\n=== Tuning Random Forest ===")
    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
    
    # Define the parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [None, 5, 10, 15, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__bootstrap': [True, False],
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create the randomized search
    random_search = RandomizedSearchCV(
        pipeline, 
        param_grid, 
        n_iter=30,
        cv=cv, 
        scoring='accuracy', 
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the randomized search
    random_search.fit(X_train, y_train)
    
    # Print the best parameters
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_gradient_boosting(X_train, y_train):
    """Tune hyperparameters for Gradient Boosting using RandomizedSearchCV"""
    print("\n=== Tuning Gradient Boosting ===")
    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier())
    ])
    
    # Define the parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create the randomized search
    random_search = RandomizedSearchCV(
        pipeline, 
        param_grid, 
        n_iter=30,
        cv=cv, 
        scoring='accuracy', 
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the randomized search
    random_search.fit(X_train, y_train)
    
    # Print the best parameters
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def create_ensemble(best_models):
    """Create a voting ensemble from the best models"""
    print("\n=== Creating Voting Ensemble ===")
    
    # Create named estimators
    named_estimators = [
        ('log_reg', best_models['logistic']),
        ('svm', best_models['svm']),
        ('rf', best_models['random_forest']),
        ('gb', best_models['gradient_boosting'])
    ]
    
    # Create a voting classifier
    voting_clf = VotingClassifier(
        estimators=named_estimators,
        voting='soft'  # Use probabilities for voting
    )
    
    return voting_clf

def evaluate_models(models, X_test, y_test):
    """Evaluate models on test data and print metrics"""
    print("\n=== Model Evaluation ===")
    
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Store results
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc': auc
        }
        
        # Print results
        print(f"\nResults for {name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
    
    return results

def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models"""
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
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Heart Attack Prediction Models')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('model_refinement_roc_curves.png')
    plt.close()

def feature_importance(models, feature_names):
    """Plot feature importance for models that support it"""
    # Models that support feature_importances_
    importance_models = ['random_forest', 'gradient_boosting']
    
    for name in importance_models:
        if name in models:
            # Get the model (extract from pipeline)
            model = models[name].named_steps['classifier']
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create DataFrame for visualization
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'Feature Importance for {name}')
            plt.tight_layout()
            plt.savefig(f'{name}_feature_importance.png')
            plt.close()

def save_best_model(model, model_name):
    """Save the best model"""
    joblib.dump(model, f'refined_{model_name}_model.pkl')
    print(f"Model saved as refined_{model_name}_model.pkl")

def main():
    """Main function to run the model refinement process"""
    # Load data
    df, X_train, X_test, y_train, y_test = load_data()
    
    # Get feature names for later interpretation
    feature_names = df.drop('Target', axis=1).columns.tolist()
    
    # Tune models - using a more focused approach for efficiency
    print("Starting model tuning with simplified parameters for demonstration...")
    best_models = {}
    
    # Use simplified tuning for demonstration purposes
    # Logistic Regression with simplified parameters
    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    lr_param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear'],
        'classifier__class_weight': [None, 'balanced']
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_lr = GridSearchCV(pipeline_lr, lr_param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_lr.fit(X_train, y_train)
    best_models['logistic'] = grid_search_lr.best_estimator_
    print(f"Best Logistic Regression parameters: {grid_search_lr.best_params_}")
    print(f"Best Logistic Regression CV score: {grid_search_lr.best_score_:.4f}")
    
    # SVM with simplified parameters
    pipeline_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True))
    ])
    
    svm_param_grid = {
        'classifier__C': [1, 10],
        'classifier__kernel': ['rbf', 'linear'],
        'classifier__gamma': ['scale']
    }
    
    grid_search_svm = GridSearchCV(pipeline_svm, svm_param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_svm.fit(X_train, y_train)
    best_models['svm'] = grid_search_svm.best_estimator_
    print(f"Best SVM parameters: {grid_search_svm.best_params_}")
    print(f"Best SVM CV score: {grid_search_svm.best_score_:.4f}")
    
    # Random Forest with simplified parameters
    pipeline_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': [None, 'balanced']
    }
    
    grid_search_rf = GridSearchCV(pipeline_rf, rf_param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_models['random_forest'] = grid_search_rf.best_estimator_
    print(f"Best Random Forest parameters: {grid_search_rf.best_params_}")
    print(f"Best Random Forest CV score: {grid_search_rf.best_score_:.4f}")
    
    # Gradient Boosting with simplified parameters
    pipeline_gb = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    gb_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 4]
    }
    
    grid_search_gb = GridSearchCV(pipeline_gb, gb_param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_gb.fit(X_train, y_train)
    best_models['gradient_boosting'] = grid_search_gb.best_estimator_
    print(f"Best Gradient Boosting parameters: {grid_search_gb.best_params_}")
    print(f"Best Gradient Boosting CV score: {grid_search_gb.best_score_:.4f}")
    
    # Create ensemble
    best_models['ensemble'] = create_ensemble(best_models)
    
    # Train the ensemble
    best_models['ensemble'].fit(X_train, y_train)
    
    # Evaluate all models
    results = evaluate_models(best_models, X_test, y_test)
    
    # Plot ROC curves
    plot_roc_curves(best_models, X_test, y_test)
    
    # Plot feature importance
    feature_importance(best_models, feature_names)
    
    # Find the best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    # Save the best model
    save_best_model(best_models[best_model_name], best_model_name)
    
    # Also save the ensemble model (often preferred in practice)
    save_best_model(best_models['ensemble'], 'ensemble')

if __name__ == "__main__":
    main()
