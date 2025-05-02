import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from systemds.context import SystemDSContext
from systemds.operator.algorithm import multiLogReg, multiLogRegPredict

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

# 4. Baseline Model: Logistic Regression with SystemDS
with SystemDSContext() as sds:
    X_ds = sds.from_numpy(X_train_scaled)
    y_ds = sds.from_numpy(y_train + 1.0)
    bias = multiLogReg(X_ds, y_ds, maxi=100, verbose=False)
    Xt_ds = sds.from_numpy(X_test_scaled)
    yt_ds = sds.from_numpy(y_test + 1.0)
    _, y_pred, acc = multiLogRegPredict(Xt_ds, bias, Y=yt_ds, verbose=False).compute()

print(f"SystemDS Logistic Regression Test Accuracy: {acc}")
