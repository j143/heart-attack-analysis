# Heart Attack Analysis and Prediction

![Heart Attack Analysis](correlation_heatmap.png)

## Heart Attack Analysis
1.	Introduction
 A heart attack occurs when an artery supplying your heart with blood and oxygen becomes blocked. A blood clot can form and block your arteries, causing a heart attack. This Heart Attack Analysis helps to understand the chance of attack occurrence in persons based on varied health conditions.
2.	Dataset
The dataset is Heart_Attack_Analysis_Data.csv. It has been added to this. 
This dataset contains data about some hundreds of patients mentioning Age, Sex, Exercise Include Angia(1=YES, 0=NO), Chest Pain Type(Value 1: typical angina, Value2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic), ECG Results, Blood Pressure, Cholesterol, Blood Sugar, Family History (Number of persons affected in the family), Maximum Heart Rate, Target -0=LESS CHANCE , 1= MORE CHANCE

## Aim of the assignment is to 

•	Building a Predictive Model    (Which features decide heart attack?)
•	Evaluate the model.
•	Refine the model, as appropriate

## What I need to do?

a)	Select a method for performing the analytic task
b)	Preprocess the data to enhance quality
c)	Carry out descriptive summarization of data and make observations
d)	Identify relevant, irrelevant attributes for building model. 
e)	Perform appropriate data transformations with justifications
f)	Generate new features if needed
g)	Carry out the chosen analytic task. Show results including intermediate results, as needed
h)	Evaluate the solutions
i)	Look for refinement opportunities

## Setup and Running Instructions

### Prerequisites
- Python 3.8+ 
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/j143/heart-attack-analysis
   cd heart-attack-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Analysis

#### Complete Analysis Workflow
To run the entire analysis pipeline including model training, refinement, and evaluation:

```
python summary.py
```

This will:
- Check for required dependencies
- Run model refinement if needed
- Compare original models with refined models
- Display a complete project summary

#### Step-by-Step Analysis

1. Run the original analysis with SystemDS:
   ```
   python heart_attack_systemds.py
   ```
   This performs the initial data analysis, trains logistic regression and L2SVM models, and saves the model weights.

2. Verify saved models:
   ```
   python verify_models.py
   ```
   This script verifies that the saved models can be loaded and used for predictions.

3. Run model refinement:
   ```
   python model_refinement.py
   ```
   This script performs hyperparameter tuning with cross-validation and creates an ensemble model.

4. Compare model performance:
   ```
   python model_comparison.py
   ```
   This script compares the performance of the original models with the refined models.

### Key Results

- The original L2SVM achieved ~82% accuracy
- After refinement, the Random Forest model achieved ~95% accuracy
- Key features for predicting heart attacks based on different models:
  - SystemDS Logistic Regression: Age, ECG Results, Sex, MaxHeartRate
  - Refined Random Forest: CP_Type, MaxHeartRate, Age, Cholestrol, BloodPressure

For detailed information about the analysis process and results, please refer to the `solution.md` file.

### Project Structure

- `heart_attack_systemds.py`: Main analysis script using SystemDS
- `verify_models.py`: Script to verify saved models
- `model_refinement.py`: Implements hyperparameter tuning, cross-validation, and ensemble methods
- `model_comparison.py`: Compares original and refined models
- `summary.py`: Complete workflow script with project
- `solution.md`: Detailed documentation of the approach and results
- `Heart_Attack_Analysis_Data.csv`: Dataset
- `requirements.txt`: List of required Python packages

### Visualizations

The analysis generates several visualizations:

- Correlation heatmap
- Feature distributions
- Feature importance plots
- Model performance comparisons
- ROC curves

### Saved Models

The following models are saved during the analysis:

- `logistic_regression_weights.pkl`: Original logistic regression model
- `l2svm_weights.pkl`: Original L2SVM model
- `scaler.pkl`: Data standardization parameters
- `refined_random_forest_model.pkl`: Tuned Random Forest model
- `refined_ensemble_model.pkl`: Ensemble of tuned models

## Conclusion

I have applied ML technique in heart attack analysis. I have utilized systemds and scikit-learn

Key findings:
1. Different models identified different important predictors:
   - Initial models (Logistic Regression): Age, ECG Results, Sex, Maximum Heart Rate
   - Refined models (Random Forest): CP_Type, Maximum Heart Rate, Age, Cholestrol, BloodPressure
2. Hyperparameter tuning and cross-validation improved performance
