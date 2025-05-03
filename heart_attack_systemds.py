import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from systemds.context import SystemDSContext
from systemds.operator.algorithm import multiLogReg, multiLogRegPredict, decisionTree, decisionTreePredict, randomForest, randomForestPredict
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

# --- Model Comparison: Decision Tree and Random Forest (SystemDS) ---
model_results = []

# Prepare data (no features dropped)
X_full = df.drop('Target', axis=1).values
y_full = df['Target'].values.reshape(-1, 1)
num_samples = X_full.shape[0]
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * num_samples)
train_idx, test_idx = indices[:split], indices[split:]
X_train_full, X_test_full = X_full[train_idx], X_full[test_idx]
y_train_full, y_test_full = y_full[train_idx], y_full[test_idx]
mean_full = X_train_full.mean(axis=0)
std_full = X_train_full.std(axis=0)
X_train_scaled_full = (X_train_full - mean_full) / std_full
X_test_scaled_full = (X_test_full - mean_full) / std_full

# Prepare ctypes for SystemDS tree models
# 2 = numerical, 1 = categorical
ctypes = np.array([2,1,1,2,2,1,2,2,1,1])

# Get accuracy for 'no features dropped' robustly
logreg_row = results_df[results_df['dropped'].str.strip().str.lower() == 'none']
if not logreg_row.empty:
    logreg_acc = logreg_row['accuracy'].values[0]
else:
    print("[ERROR] Could not find 'None' in 'dropped' column for logistic regression accuracy. Using 0.0 as fallback.")
    logreg_acc = 0.0
model_results.append({'model': 'SystemDS Logistic Regression', 'accuracy': logreg_acc})

with SystemDSContext() as sds:
    # Decision Tree
    X_train_sds = sds.from_numpy(X_train_scaled_full)
    y_train_sds = sds.from_numpy(y_train_full + 1.0)
    X_test_sds = sds.from_numpy(X_test_scaled_full)
    y_test_sds = sds.from_numpy(y_test_full + 1.0)
    ctypes_sds = sds.from_numpy(ctypes)
    dt_model = decisionTree(X_train_sds, y_train_sds, ctypes_sds, icpt=1, max_depth=5)
    dt_pred = decisionTreePredict(X_test_sds, dt_model).compute()
    dt_acc = np.mean((dt_pred > 0.5).astype(int) == y_test_full)
    print(f"SystemDS Decision Tree Test Accuracy: {dt_acc:.4f}")
    model_results.append({'model': 'SystemDS Decision Tree', 'accuracy': dt_acc})

    # Random Forest
    rf_model = randomForest(X_train_sds, y_train_sds, ctypes_sds, num_trees=100, icpt=1, max_depth=5)
    rf_pred = randomForestPredict(X_test_sds, rf_model).compute()
    rf_acc = np.mean((rf_pred > 0.5).astype(int) == y_test_full)
    print(f"SystemDS Random Forest Test Accuracy: {rf_acc:.4f}")
    model_results.append({'model': 'SystemDS Random Forest', 'accuracy': rf_acc})

# Save and visualize model comparison
model_results_df = pd.DataFrame(model_results)
model_results_df.to_csv('model_comparison_results.csv', index=False)

plt.figure(figsize=(8, 5))
plt.bar(model_results_df['model'], model_results_df['accuracy'], color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylabel('Test Accuracy')
plt.title('Model Comparison: Test Accuracy (SystemDS)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('model_comparison_accuracy.png')
plt.close()
print("Saved model_comparison_accuracy.png with SystemDS model comparison results.")