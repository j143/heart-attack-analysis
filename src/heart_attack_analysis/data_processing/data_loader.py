"""
Data loading and preprocessing module for heart attack analysis.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os


class DataLoader:
    """Handle data loading, preprocessing, and splitting for heart attack analysis."""
    
    def __init__(self, data_path: str = "data/Heart_Attack_Analysis_Data.csv"):
        """
        Initialize DataLoader with data path.
        
        Args:
            data_path (str): Path to the heart attack dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.scaler = None
        self.categorical_cols = ['Sex', 'CP_Type', 'BloodSugar', 'ECG', 'ExerciseAngia', 'FamilyHistory', 'Target']
        
    def load_data(self) -> pd.DataFrame:
        """
        Load heart attack data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self) -> None:
        """Print basic exploratory data analysis information."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        print('\nMissing values per column:')
        print(self.df.isnull().sum())
        
        print('\nSummary statistics:')
        print(self.df.describe(include='all'))
        
        for col in self.categorical_cols[:-1]:  # Exclude 'Target'
            if col in self.df.columns:
                print(f'\nValue counts for {col}:')
                print(self.df[col].value_counts())
        
        print('\nCorrelation matrix:')
        print(self.df.corr())
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for modeling: split and scale.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple of X_train, X_test, y_train, y_test (all scaled)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        X = self.df.drop('Target', axis=1).values
        y = self.df['Target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def prepare_data_systemds_style(self, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data using the original SystemDS approach for compatibility.
        
        Args:
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple of X_train_scaled, X_test_scaled, y_train, y_test
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        X = self.df.drop('Target', axis=1).values
        y = self.df['Target'].values.reshape(-1, 1)
        
        # Split data the same way as in original code
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        split = int(0.8 * num_samples)
        train_idx, test_idx = indices[:split], indices[split:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features using manual approach for SystemDS compatibility
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        
        # Save scaler parameters
        scaler_params = {'mean': mean, 'std': std}
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler_params
    
    def save_scaler(self, filepath: str = "outputs/models/scaler.pkl") -> None:
        """
        Save the fitted scaler to disk.
        
        Args:
            filepath (str): Path to save the scaler
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call prepare_data() first.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str = "outputs/models/scaler.pkl") -> StandardScaler:
        """
        Load a previously saved scaler.
        
        Args:
            filepath (str): Path to the saved scaler
            
        Returns:
            StandardScaler: Loaded scaler
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found at {filepath}")
            
        self.scaler = joblib.load(filepath)
        print(f"Scaler loaded from {filepath}")
        return self.scaler
    
    def get_feature_names(self) -> list:
        """
        Get feature names (column names excluding target).
        
        Returns:
            list: Feature names
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return self.df.drop('Target', axis=1).columns.tolist()
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix of features (excluding target).
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return self.df.drop('Target', axis=1).corr()
    
    def find_high_correlation_pairs(self, threshold: float = 0.8) -> list:
        """
        Find highly correlated feature pairs.
        
        Args:
            threshold (float): Correlation threshold
            
        Returns:
            list: List of tuples (feature1, feature2, correlation)
        """
        corr_matrix = self.get_correlation_matrix().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        return high_corr_pairs