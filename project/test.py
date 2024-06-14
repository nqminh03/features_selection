import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = 'D:/features_selection\data_sample\LoanDataset - LoansDatasest.csv'
df = pd.read_csv(file_path)

# Replace missing values in 'loan_int_rate' with median
imputer = SimpleImputer(strategy='median')
df['loan_int_rate'] = imputer.fit_transform(df[['loan_int_rate']])

# Remove currency symbol and convert to float
df['loan_amnt'] = df['loan_amnt'].replace('[\£,]', '', regex=True).astype(float)

# Remove comma in customer_income and convert to float
df['customer_income'] = df['customer_income'].replace(',', '', regex=True).astype(float)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['home_ownership', 'loan_intent', 'loan_grade', 'historical_default', 'Current_loan_status']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Replace missing values in categorical columns with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# Define features (X) and target (y)
X = df.drop(['customer_id', 'Current_loan_status'], axis=1)
y = df['Current_loan_status']

# Replace any remaining missing values with the median
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Chi-square Test
chi2_selector = SelectKBest(chi2, k=5)
X_kbest = chi2_selector.fit_transform(X, y)
chi2_selected_features = df.drop(['customer_id', 'Current_loan_status'], axis=1).columns[chi2_selector.get_support()]

# Variance Threshold
threshold_selector = VarianceThreshold(threshold=0.1)
X_high_variance = threshold_selector.fit_transform(X)
variance_selected_features = df.drop(['customer_id', 'Current_loan_status'], axis=1).columns[threshold_selector.get_support()]

# Recursive Feature Elimination (RFE)
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
rfe_selected_features = df.drop(['customer_id', 'Current_loan_status'], axis=1).columns[rfe.get_support()]

# Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
model = SelectFromModel(lasso, prefit=True)
X_lasso = model.transform(X)
lasso_selected_features = df.drop(['customer_id', 'Current_loan_status'], axis=1).columns[model.get_support()]

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X, y)
selector = SelectFromModel(rf_model, prefit=True)
X_rf = selector.transform(X)
rf_selected_features = df.drop(['customer_id', 'Current_loan_status'], axis=1).columns[selector.get_support()]

# Plot selected features
methods = ['Chi-square', 'Variance Threshold', 'RFE', 'Lasso', 'Random Forest']
selected_features = [chi2_selected_features, variance_selected_features, rfe_selected_features, lasso_selected_features, rf_selected_features]

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, method, features in zip(axes, methods, selected_features):
    sns.barplot(x=features, y=[1]*len(features), ax=ax)
    ax.set_title(f'{method} Selected Features')
    ax.set_yticks([])
    ax.set_xlabel('Features')

# Remove the last empty subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()
