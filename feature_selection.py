import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats

# Read the data
df_loan = pd.read_csv('synthetic_loan_data.csv')
df = df_loan.copy()

# 1. Create New Features
def create_features(df):
    """Create new features from existing ones"""
    
    # Loan amount to income ratio
    df['Loan_To_Income_Ratio'] = df['Loan_Amount'] / df['Income']
    
    # Calculate the monthly payment based on loan amount, interest rate and loan term
    # The formula is based on the monthly payment formula for a fixed-rate loan
    # M = P[i(1+i)^n]/[(1+i)^n - 1]
    # where:
    # M = monthly payment
    # P = loan amount
    # i = monthly interest rate (annual interest rate divided by 12)
    # n = number of payments (loan term in months)
    # Monthly payment calculation
    df['Monthly_Payment'] = (df['Loan_Amount'] * (df['Interest_Rate']/1200) * 
                           (1 + df['Interest_Rate']/1200)**df['Loan_Term']) / \
                           ((1 + df['Interest_Rate']/1200)**df['Loan_Term'] - 1)
    
    # Payment to income ratio
    df['Payment_To_Income_Ratio'] = df['Monthly_Payment'] / (df['Income'] / 12)
    
    # Credit score groups
    df['Credit_Score_Group'] = pd.qcut(df['Credit_Score'], q=5, 
                                     labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    
    # Income groups
    df['Income_Group'] = pd.qcut(df['Income'], q=5, 
                               labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    
    # Loan term categories
    df['Term_Category'] = pd.cut(df['Loan_Term'], 
                                bins=[0, 12, 36, 60, np.inf],
                                labels=['Short', 'Medium', 'Long', 'Very_Long'])
    
    return df

# Apply feature engineering
df_engineered = create_features(df)

# 2. Prepare features for selection
def prepare_features(df):
    """Prepare categorical and numerical features"""
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Employment_Status', 'Education_Level', 
                         'Marital_Status', 'Credit_Score_Group',
                         'Income_Group', 'Term_Category']
    
    for col in categorical_columns:
        df[col + '_Encoded'] = le.fit_transform(df[col])
    
    # Select numerical columns
    numerical_columns = ['Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Loan_Term',
                        'Interest_Rate', 'Debt_To_Income_Ratio', 'Previous_Loans',
                        'Loan_To_Income_Ratio', 'Monthly_Payment', 'Payment_To_Income_Ratio']
    
    # Combine encoded categorical and numerical columns
    feature_columns = numerical_columns + [col + '_Encoded' for col in categorical_columns]
    
    return df[feature_columns], feature_columns

# Prepare features
X, feature_names = prepare_features(df_engineered)
y = df_engineered['Default']

# 3. Feature Selection Method

# 3.1 Correlation Analysis
plt.figure(figsize=(12, 8))
correlation_with_default = pd.DataFrame({
    'Feature': feature_names,
    'Correlation': [abs(stats.pointbiserialr(X[col], y)[0]) for col in feature_names]
})
correlation_with_default = correlation_with_default.sort_values('Correlation', ascending=True)
plt.barh(range(len(correlation_with_default)), correlation_with_default['Correlation'])
plt.yticks(range(len(correlation_with_default)), correlation_with_default['Feature'])
plt.title('Absolute Correlation with Default')
plt.xlabel('Absolute Correlation Coefficient')
plt.tight_layout()
plt.show()

# Show top 10 features
top_10_features = correlation_with_default.head(10)
plt.figure(figsize=(12, 8))
plt.barh(range(len(top_10_features)), top_10_features['Correlation'])
plt.yticks(range(len(top_10_features)), top_10_features['Feature'])
plt.title('Top 10 Features by Correlation with Default')
plt.xlabel('Absolute Correlation Coefficient')
plt.tight_layout()
plt.show()


# Save new features to CSV
df_engineered.to_csv('loan_data_with_features.csv', index=False)
