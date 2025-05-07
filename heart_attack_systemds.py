import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from systemds.context import SystemDSContext
from systemds.operator.algorithm import multiLogReg, multiLogRegPredict, l2svm, l2svmPredict
import itertools

# 1. Load data
df = pd.read_csv('Heart_Attack_Analysis_Data.csv')

# 2. Exploratory Data Analysis (EDA)
print('Missing values per column:')
print(df.isnull().sum())

print('\nSummary statistics:')
print(df.describe(include='all'))

categorical_cols = ['Sex', 'CP_Type', 'BloodSugar', 'ECG', 'ExerciseAngia', 'FamilyHistory', 'Target']
for col in categorical_cols:
    print(f'\nValue counts for {col}:')
    print(df[col].value_counts())

print('\nCorrelation matrix:')
print(df.corr())

sns.set(style='whitegrid', palette='muted')
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

for col in df.columns:
    plt.figure()
    if col in categorical_cols:
        sns.countplot(x=col, data=df)
    else:
        sns.histplot(df[col], kde=True)
    # plt.title(f'Distribution of {col}')
    # plt.tight_layout()
    # plt.savefig(f'distribution_{col}.png')
    # plt.close()

# Save all distributions in one file
num_features = len(df.columns)
cols = 3
rows = int(np.ceil(num_features / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()

for idx, col in enumerate(df.columns):
    ax = axes[idx]
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
plt.savefig('distributions_all.png')
plt.close()

# 3. Data split and scaling (DataManager-style)
X = df.drop('Target', axis=1).values
y = df['Target'].values.reshape(-1, 1)
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * num_samples)
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# --- Feature Selection Step ---
# 1. Identify highly correlated feature pairs (|corr| > 0.8)
corr_matrix = df.drop('Target', axis=1).corr().abs()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("\n[Feature Selection] Highly correlated feature pairs (|corr| > 0.8):")
    for f1, f2, corr in high_corr_pairs:
        print(f"{f1} <-> {f2}: correlation = {corr:.2f}")
else:
    print("\n[Feature Selection] No highly correlated feature pairs found (|corr| > 0.8).")

# 2. SystemDS: Feature importances, model training, and evaluation in one context
with SystemDSContext() as sds:
    X_ds = sds.from_numpy(X_train_scaled)
    y_ds = sds.from_numpy(y_train + 1.0)
    bias = multiLogReg(X_ds, y_ds, maxi=100, verbose=False)
    weights = bias.compute().flatten()
    feature_names = df.drop('Target', axis=1).columns
    feature_coefs = list(zip(feature_names, weights[1:]))
    feature_coefs_sorted = sorted(feature_coefs, key=lambda x: abs(x[1]))
    print("\n[Feature Selection] Features with lowest absolute importance (SystemDS coefficients):")
    for name, coef in feature_coefs_sorted[:3]:
        print(f"{name}: {coef:.4f}")

    # Model evaluation
    Xt_ds = sds.from_numpy(X_test_scaled)
    yt_ds = sds.from_numpy(y_test + 1.0)
    _, y_pred, acc = multiLogRegPredict(Xt_ds, bias, Y=yt_ds, verbose=False).compute()

    # Print all feature importances sorted by absolute value
    feature_coefs_sorted_desc = sorted(feature_coefs, key=lambda x: abs(x[1]), reverse=True)
    print("\nSystemDS Logistic Regression Feature Importances (sorted by absolute value):")
    for name, coef in feature_coefs_sorted_desc:
        print(f"{name}: {coef:.4f}")

    print("\nInterpretation: Features with higher absolute coefficient values have a stronger influence on the prediction.\nPositive values increase risk, negative values decrease risk.")

    print(f"SystemDS Logistic Regression Test Accuracy: {acc}")

# --- Adding L2SVM for Model Comparison ---
with SystemDSContext() as sds:
    # Train L2SVM model
    X_ds = sds.from_numpy(X_train_scaled)
    y_ds = sds.from_numpy(y_train + 1.0)
    l2svm_model = l2svm(X_ds, y_ds, reg=0.01, maxIterations=100, verbose=False)
    l2svm_weights = l2svm_model.compute().flatten()

    # Evaluate L2SVM model
    Xt_ds = sds.from_numpy(X_test_scaled)
    l2svm_y_pred_raw, l2svm_y_pred_maxed = l2svmPredict(Xt_ds, l2svm_model, verbose=False).compute()

    # Calculate accuracy for L2SVM manually
    l2svm_acc = np.mean((l2svm_y_pred_maxed.flatten() == y_test.flatten()).astype(float))
    print(f"L2SVM Test Accuracy: {l2svm_acc}")

# Compare L2SVM with Logistic Regression
print("\nModel Comparison:")
print(f"Logistic Regression Test Accuracy: {acc}")
print(f"L2SVM Test Accuracy: {l2svm_acc}")

# --- Systematic Experimentation with Feature Removal ---
# Features to experiment with removing
investigate_features = ['Cholestrol', 'BloodPressure', 'ExerciseAngia']

# All combinations of features to drop (including none)
results = []
for n in range(len(investigate_features) + 1):
    for drop_set in itertools.combinations(investigate_features, n):
        drop_list = list(drop_set)
        print(f"\n[Experiment] Dropping features: {drop_list if drop_list else 'None'}")
        # Prepare data with selected features dropped
        X_exp = df.drop(['Target'] + drop_list, axis=1).values
        feature_names_exp = df.drop(['Target'] + drop_list, axis=1).columns
        # Data split and scaling
        num_samples = X_exp.shape[0]
        indices = np.arange(num_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(0.8 * num_samples)
        train_idx, test_idx = indices[:split], indices[split:]
        X_train_exp, X_test_exp = X_exp[train_idx], X_exp[test_idx]
        y_train_exp, y_test_exp = y[train_idx], y[test_idx]
        mean_exp = X_train_exp.mean(axis=0)
        std_exp = X_train_exp.std(axis=0)
        X_train_scaled_exp = (X_train_exp - mean_exp) / std_exp
        X_test_scaled_exp = (X_test_exp - mean_exp) / std_exp
        with SystemDSContext() as sds:
            X_ds = sds.from_numpy(X_train_scaled_exp)
            y_ds = sds.from_numpy(y_train_exp + 1.0)
            bias = multiLogReg(X_ds, y_ds, maxi=100, verbose=False)
            Xt_ds = sds.from_numpy(X_test_scaled_exp)
            yt_ds = sds.from_numpy(y_test_exp + 1.0)
            _, y_pred, acc = multiLogRegPredict(Xt_ds, bias, Y=yt_ds, verbose=False).compute()
            print(f"Test Accuracy: {acc}")
            results.append({'dropped': drop_list, 'accuracy': acc, 'features': list(feature_names_exp)})

# Save results to CSV for systematic tracking
results_df = pd.DataFrame(results)
results_df['dropped'] = results_df['dropped'].apply(lambda x: ','.join(x) if x else 'None')
results_df['features'] = results_df['features'].apply(lambda x: ','.join(x))
results_df.to_csv('feature_removal_results.csv', index=False)

print("\n[Summary of Experiments]")
print(results_df)

# Visualization of feature removal experiments
results_df = pd.read_csv('feature_removal_results.csv')

# Improved visualization with zoomed-in accuracy range for better insights
plt.figure(figsize=(12, 6))
sns.barplot(x='dropped', y='accuracy', data=results_df, palette='viridis')
plt.xlabel('Dropped Features', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Test Accuracy for Each Feature Removal Experiment', fontsize=14)
plt.ylim(70, 90)  # Zoom in on the accuracy range 70 to 90
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('feature_removal_accuracy.png')
plt.close()
print("Saved feature_removal_accuracy.png with experiment results.")