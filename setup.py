from setuptools import setup, find_packages

setup(
    name='heart-attack-analysis',
    version='0.1.0',
    description='Heart Attack Prediction and Analysis',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'systemds',
    ],
    python_requires='>=3.7',
)
