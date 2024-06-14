import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import impute
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('D:/features_selection\data_sample\LoanDataset - LoansDatasest.csv')
df.info()
# Tìm dữ liệu lỗi
df.isna()
# Xử lý giá trị thiếu bằng cách điền giá trị trung bình
# df.fillna(df.mean(), inplace=True)

imputer = SimpleImputer(strategy='median')
df['loan_int_rate'] = imputer.fit_transform(df[['loan_int_rate']])

# Xử lý dữ liệu không hợp lệ: loại bỏ ký hiệu tiền tệ
df['loan_amnt'] = df['loan_amnt'].replace('[\£,]', '', regex=True).astype(float)

# xóa dấu dữ liệu trong cột customer_income và đổi định dạng float
df['customer_income'] = df['customer_income'].replace(',', '', regex=True).astype(float)

# Mã hóa các biến phân loại
label_encoder = LabelEncoder()
for column in ['home_ownership', 'loan_intent', 'loan_grade', 'historical_default', 'Current_loan_status']:
    df[column] = label_encoder.fit_transform(df[column])
    

# Xử lý biến mục tiêu
df['Current_loan_status'] = df['Current_loan_status'].apply(lambda x: 1 if x == 'DEFAULT' else 0)

X = df.drop(['customer_id', 'Current_loan_status'], axis=1)
y = df['Current_loan_status']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ensure no NaN values are present in the dataset
#assert not pd.DataFrame(X).isnull().values.any()
print (df)
# Lựa chọn biến bằng Chi-square Test
chi2_selector = SelectKBest(chi2, k=5)
X_kbest = chi2_selector.fit_transform(X, y)

print("Chi-square Test - Reduced shape:", X_kbest.shape)

# Lựa chọn biến bằng Variance Threshold
threshold_selector = VarianceThreshold(threshold=0.1)
X_high_variance = threshold_selector.fit_transform(X)
variance_selected_features = df.drop(['customer_id', 'Current_loan_status'], axis=1).columns[threshold_selector.get_support()]
print("Features_Important:", variance_selected_features)

# Lựa chọn biến bằng Recursive Feature Elimination (RFE)
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)

print("RFE - Reduced shape:", X_rfe.shape)

# Lựa chọn biến bằng Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
model = SelectFromModel(lasso, prefit=True)
X_lasso = model.transform(X)

print("Lasso - Reduced shape:", X_lasso.shape)

# Lựa chọn biến bằng Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X, y)
selector = SelectFromModel(rf_model, prefit=True)
X_rf = selector.transform(X)

print("Random Forest - Reduced shape:", X_rf.shape)

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

plt.tight_layout()
plt.show()