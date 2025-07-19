"""
Heart Attack Analysis Package

A comprehensive package for heart attack prediction using machine learning.
"""

__version__ = "0.1.0"
__author__ = "Heart Attack Analysis Team"

from .data_processing.data_loader import DataLoader
from .modeling.model_trainer import ModelTrainer
from .visualization.plotter import Plotter

__all__ = ['DataLoader', 'ModelTrainer', 'Plotter']