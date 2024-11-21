import numpy as np
import pandas as pd
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

def generate_loan_data(n_records=5000):
    # Generate age with a realistic distribution (25-75 years)
    age = stats.truncnorm((25-45)/10, (75-45)/10, loc=45, scale=10).rvs(n_records)
    age = age.astype(int)
    
    # Generate gender with roughly equal distribution
    gender = np.random.choice(['Male', 'Female'], size=n_records, p=[0.58, 0.42])
    
    # Generate income based on age (correlation) with realistic distribution
    base_income = stats.lognorm(s=0.5, loc=300000, scale=500000).rvs(n_records)
    age_income_factor = (age - 25) * 3000  # Income tends to increase with age
    income = base_income + age_income_factor
    income = income.clip(250000, 2500000)  # Set realistic bounds
    income = income.round(-3)  # Round to thousands
    
    # Generate credit scores (300-850)
    credit_score = stats.truncnorm((300-700)/100, (850-700)/100, loc=700, scale=100).rvs(n_records)
    credit_score = credit_score.astype(int)
    
    # Generate loan amounts based on income and credit score
    income_factor = income / 50000
    credit_factor = (credit_score - 300) / 550
    base_loan = stats.lognorm(s=0.6, loc=10000, scale=30000).rvs(n_records)
    loan_amount = base_loan * income_factor * (0.5 + 0.5 * credit_factor)
    loan_amount = loan_amount.clip(50000, 3000000)
    loan_amount = loan_amount.round(-3)  # Round to thousands
    
    # Generate loan terms (common terms: 12, 24, 36, 48, 60 months)
    loan_terms = np.random.choice([12, 24, 36, 48, 60], size=n_records, 
                                p=[0.1, 0.2, 0.4, 0.2, 0.1])
    
    # Generate employment status with realistic probabilities
    employment_status = np.random.choice(
        ['Employed', 'Self-employed', 'Unemployed'],
        size=n_records,
        p=[0.8, 0.15, 0.05]
    )
    
    # Calculate DTI ratio based on income and loan amount
    existing_debt = stats.lognorm(s=0.5, loc=0, scale=20000).rvs(n_records)
    monthly_debt = existing_debt * 0.05
    monthly_income = income / 12
    dti_ratio = (monthly_debt / monthly_income * 100).clip(0, 65)
    dti_ratio = np.round(dti_ratio, 1)
    
    # Generate education level with realistic probabilities
    education_probs = [0.25, 0.35, 0.25, 0.15]  # Adjusted for reality
    education_level = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        size=n_records,
        p=education_probs
    )
    
    # Generate number of previous loans based on age and credit score
    max_loans = ((age - 25) / 10).astype(int).clip(0, 5)
    credit_factor = ((credit_score - 300) / 550 * 3).astype(int)
    previous_loans = np.random.randint(0, max_loans + credit_factor + 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Income': income,
        'Credit_Score': credit_score,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_terms,
        'Employment_Status': employment_status,
        'Debt_To_Income_Ratio': dti_ratio,
        'Education_Level': education_level,
        'Previous_Loans': previous_loans
    })

     # Calculate default probabilities
    default_probs = np.zeros(n_records)
    
    # Calculate default probability for each record
    for i in range(n_records):
        # Base probability
        prob = 0.05
        
        # Credit score factor (higher score = lower probability)
        credit_factor = (850 - df.loc[i, 'Credit_Score']) / 550
        prob += credit_factor * 0.15
        
        # DTI ratio factor (higher DTI = higher probability)
        dti_factor = df.loc[i, 'Debt_To_Income_Ratio'] / 65
        prob += dti_factor * 0.15
        
        # Employment status factor
        if df.loc[i, 'Employment_Status'] == 'Unemployed':
            prob += 0.15
        elif df.loc[i, 'Employment_Status'] == 'Self-employed':
            prob += 0.05
        
        # Previous loans factor
        prob += (df.loc[i, 'Previous_Loans'] * 0.02)
        
        # Loan amount to income ratio factor
        loan_to_income = df.loc[i, 'Loan_Amount'] / df.loc[i, 'Income']
        prob += min(loan_to_income * 0.05, 0.15)
        
        # Education level factor (higher education = lower probability)
        education_factor = {
            'High School': 0.05,
            'Bachelor': 0.03,
            'Master': 0.02,
            'PhD': 0.01
        }
        prob += education_factor[df.loc[i, 'Education_Level']]
        
        # Clip probability between 0 and 1
        default_probs[i] = min(max(prob, 0), 1)
    
    # Generate actual defaults based on probabilities
    df['Default'] = np.random.binomial(n=1, p=default_probs)

    return df

# Generate the data
loan_data = generate_loan_data(5000)

# Save to CSV
loan_data.to_csv('synthetic_loan_data.csv', index=False)