import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# for loading dataset
df = pd.read_csv('Master_Dataset_Final.csv')

# renames columns for regression formula
df = df.rename(columns={
    'Launch Base Price (USD)': 'Launch_Price',
    'Current Base Price (USD)': 'Current_Price',
    'Global Review Count': 'Global_Review_Count',
    'Price Tier': 'Price_Tier',
    'Macro Genre': 'Macro_Genre',
    'Dev Tier': 'Dev_Tier',
    'Is_Multiplayer': 'Is_Multiplayer',
    'Is_Early_Access': 'Is_Early_Access'
})

# merging tier override to avoid confusion
df['Price_Tier'] = df['Price_Tier'].astype(str).replace({
    '3 (Tier Override: Mid substituted for Premium)': '3'
})
df['Price_Tier'] = pd.to_numeric(df['Price_Tier'])

df['Is_Multiplayer'] = df['Is_Multiplayer'].replace({1: 0, 2: 1})

# variable transforms
# calculates price drop % from launch
df['Price_Drop_Pct'] = ((df['Launch_Price'] - df['Current_Price']) / df['Launch_Price']) * 100

# logging review count to fix right-skew
df['Global_Review_Count'] = pd.to_numeric(df['Global_Review_Count'], errors='coerce')
df['Log_Review_Count'] = np.log(df['Global_Review_Count'])

# used for mean-centering and logged rev count
mean_log_reviews = df['Log_Review_Count'].mean()
df['Centered_Log_Review_Count'] = df['Log_Review_Count'] - mean_log_reviews

# drop any missing data for clean models
df_clean = df.dropna(subset=['Mean_Sentiment_Score', 'Is_AI_Cohort', 'Centered_Log_Review_Count', 
                             'Price_Tier', 'Macro_Genre', 'Dev_Tier', 
                             'Is_Multiplayer', 'Is_Early_Access', 'Theme_Tech_Ratio', 'Price_Drop_Pct'])

print("="*60)
print(f"Executing Regressions on N={len(df_clean)} perfectly matched observations...")
print("="*60)

# model 1: sentiment penalty (hc3)
model1 = smf.ols("Mean_Sentiment_Score ~ Is_AI_Cohort + Centered_Log_Review_Count + C(Price_Tier) + C(Macro_Genre) + C(Dev_Tier) + Is_Multiplayer + Is_Early_Access", data=df_clean).fit(cov_type='HC3')
print("\nMODEL 1: VADER SENTIMENT SCORE (ROBUST)\n")
print(model1.summary().tables[1])

# model 2: tech hostility flag (hc3)
model2 = smf.ols("Theme_Tech_Ratio ~ Is_AI_Cohort + Centered_Log_Review_Count + C(Price_Tier) + C(Macro_Genre) + C(Dev_Tier) + Is_Multiplayer + Is_Early_Access", data=df_clean).fit(cov_type='HC3')
print("\nMODEL 2: TECHNICAL MALFUNCTION FLAG RATIO (ROBUST)\n")
print(model2.summary().tables[1])

# model 3: price decay (hc3)
model3 = smf.ols("Price_Drop_Pct ~ Is_AI_Cohort + Centered_Log_Review_Count + C(Price_Tier) + C(Macro_Genre) + C(Dev_Tier) + Is_Multiplayer + Is_Early_Access", data=df_clean).fit(cov_type='HC3')
print("\nMODEL 3: BASE PRICE DECAY PERCENTAGE (ROBUST)\n")
print(model3.summary().tables[1])


# exporting results and fit stats
def extract_results(model, model_name):
    res_df = pd.DataFrame({
        'Coefficient': model.params,
        'Robust_Std_Err': model.bse,
        't_stat': model.tvalues,
        'p_value': model.pvalues
    })
    return res_df

# getting fit stats
def fit_stats(model):
    return pd.DataFrame({
        'R_squared': [round(model.rsquared, 3)],
        'Adj_R_squared': [round(model.rsquared_adj, 3)],
        'Robust_Wald_F': [round(model.fvalue, 3)],
        'F_p_value': [model.f_pvalue]
    }, index=['Model fit'])

results_m1 = extract_results(model1, 'Model 1 (Sentiment)')
results_m2 = extract_results(model2, 'Model 2 (Tech Hostility)')
results_m3 = extract_results(model3, 'Model 3 (Price Drop)')

# saving as Excel file
with pd.ExcelWriter('Regression_Results_Robust_Final.xlsx') as writer:
    results_m1.to_excel(writer, sheet_name='M1 - Sentiment')
    fit_stats(model1).to_excel(writer, sheet_name='M1 - Fit')
    
    results_m2.to_excel(writer, sheet_name='M2 - Tech Hostility')
    fit_stats(model2).to_excel(writer, sheet_name='M2 - Fit')
    
    results_m3.to_excel(writer, sheet_name='M3 - Price Drop')
    fit_stats(model3).to_excel(writer, sheet_name='M3 - Fit')

print("\n" + "="*60)
print("Regression results and fit statistics successfully exported to 'Regression_Results_Robust_Final.xlsx'")
print("="*60)