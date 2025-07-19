from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name='heart-attack-analysis',
    version='0.1.0',
    description='Heart Attack Prediction and Analysis using Machine Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Heart Attack Analysis Team',
    author_email='',
    url='https://github.com/j143/heart-attack-analysis',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        'console_scripts': [
            'heart-attack-analysis=heart_attack_analysis.main:main',
        ],
    },
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=5.0',
            'isort>=5.0',
        ],
    },
)
