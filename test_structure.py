#!/usr/bin/env python3
"""
Test script to verify the reorganized project structure works correctly.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from heart_attack_analysis.data_processing.data_loader import DataLoader
        print("✓ DataLoader import successful")
    except Exception as e:
        print(f"✗ DataLoader import failed: {e}")
        return False
    
    try:
        from heart_attack_analysis.modeling.model_trainer import ModelTrainer
        print("✓ ModelTrainer import successful")
    except Exception as e:
        print(f"✗ ModelTrainer import failed: {e}")
        return False
    
    try:
        from heart_attack_analysis.visualization.plotter import Plotter
        print("✓ Plotter import successful")
    except Exception as e:
        print(f"✗ Plotter import failed: {e}")
        return False
    
    try:
        from heart_attack_analysis.utils.helpers import ConfigManager, Logger
        print("✓ Utilities import successful")
    except Exception as e:
        print(f"✗ Utilities import failed: {e}")
        return False
    
    print("All imports successful!")
    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from heart_attack_analysis.data_processing.data_loader import DataLoader
        
        data_loader = DataLoader('data/Heart_Attack_Analysis_Data.csv')
        df = data_loader.load_data()
        
        print(f"✓ Data loaded successfully: {df.shape}")
        
        # Test data preparation
        X_train, X_test, y_train, y_test, scaler_params = data_loader.prepare_data_systemds_style()
        print(f"✓ Data prepared successfully: Train={X_train.shape}, Test={X_test.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    try:
        from heart_attack_analysis.data_processing.data_loader import DataLoader
        from heart_attack_analysis.visualization.plotter import Plotter
        
        data_loader = DataLoader('data/Heart_Attack_Analysis_Data.csv')
        df = data_loader.load_data()
        
        plotter = Plotter('outputs/plots')
        plotter.plot_correlation_heatmap(df, "test_correlation_heatmap.png")
        
        print("✓ Visualization test successful")
        return True
    
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    try:
        from heart_attack_analysis.utils.helpers import ConfigManager
        
        config = ConfigManager('config/test_config.json')
        config.set('test_key', 'test_value')
        config.save_config()
        
        # Load config again
        config2 = ConfigManager('config/test_config.json')
        if config2.get('test_key') == 'test_value':
            print("✓ Configuration test successful")
            
            # Clean up
            if os.path.exists('config/test_config.json'):
                os.remove('config/test_config.json')
            
            return True
        else:
            print("✗ Configuration test failed: value mismatch")
            return False
    
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_cli():
    """Test CLI functionality."""
    print("\nTesting CLI...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'main.py', '--help'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and 'Heart Attack Analysis CLI' in result.stdout:
            print("✓ CLI test successful")
            return True
        else:
            print(f"✗ CLI test failed: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("HEART ATTACK ANALYSIS - PROJECT STRUCTURE TESTS")
    print("="*60)
    
    tests = [
        test_imports,
        test_data_loading,
        test_visualization,
        test_configuration,
        test_cli
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project structure is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)