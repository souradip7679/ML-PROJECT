import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = 'Mobile_Price_ML_Analysis.xlsx'
df = pd.read_excel(file_path, sheet_name='Clean Data')

# Clean string whitespace and convert Memory to numeric
df['Models_clean'] = df['Models'].astype(str).str.strip()
df['Mobile_clean'] = df['Mobile'].astype(str).str.strip()
df['Memory_num'] = pd.to_numeric(df['Memory'], errors='coerce')
df['Discount_Pct'] = (df['Discount'] / df['Original Price']) * 100

# Standardize RAM and Storage to GB for accurate spec comparisons
# (Values > 18 in Memory are legacy phones in MB; Storage == 1 on >50k phones is 1TB = 1024GB)
df['RAM_GB'] = df['Memory_num'].apply(lambda x: x / 1024.0 if pd.notnull(x) and x > 18 else x)
df['Storage_GB'] = df.apply(
    lambda r: 1024 if (r['Storage'] == 1 and r['Selling Price'] > 50000) else r['Storage'], axis=1
)

# ==============================================================================
# Q1. Brand with the highest number of mobile models
# ==============================================================================
print("=== Q1: Brand with Highest Number of Mobile Models ===")
q1_variants = df['Brands'].value_counts()
q1_unique_models = df.groupby('Brands')['Models_clean'].nunique().sort_values(ascending=False)
q1_df = pd.DataFrame({'Total_Variants': q1_variants, 'Unique_Models': q1_unique_models})
print(q1_df.sort_values(by='Total_Variants', ascending=False))

# ==============================================================================
# Q2. Top 10 mobile models based on their selling price
# ==============================================================================
print("\n=== Q2: Top 10 Mobile Models Based on Selling Price ===")
q2_unique = (
    df.groupby(['Brands', 'Mobile_clean'])['Selling Price']
    .agg(Max_Selling_Price='max', Mean_Selling_Price='mean')
    .reset_index()
    .sort_values(by='Max_Selling_Price', ascending=False)
    .head(10)
)
print(q2_unique.to_string(index=False))

# ==============================================================================
# Q3. Average selling price of mobiles for each brand
# ==============================================================================
print("\n=== Q3: Average Selling Price for Each Brand ===")
q3 = (
    df.groupby('Brands')['Selling Price']
    .agg(Variant_Count='count', Avg_Selling_Price='mean', Median_Selling_Price='median')
    .sort_values(by='Avg_Selling_Price', ascending=False)
    .round(2)
)
print(q3)

# ==============================================================================
# Q4. Mobile brand offering the highest average discount
# ==============================================================================
print("\n=== Q4: Average Discount by Brand ===")
q4 = (
    df.groupby('Brands')[['Discount', 'Discount_Pct']]
    .mean()
    .rename(columns={'Discount': 'Avg_Discount_Amt', 'Discount_Pct': 'Avg_Discount_Pct'})
    .sort_values(by='Avg_Discount_Amt', ascending=False)
    .round(2)
)
print(q4)

# ==============================================================================
# Q5. Brands with highest average customer rating & average selling price
# ==============================================================================
print("\n=== Q5: Average Customer Rating vs Average Selling Price by Brand ===")
q5 = (
    df.groupby('Brands')
    .agg(
        Avg_Rating=('Rating', 'mean'),
        Avg_Selling_Price=('Selling Price', 'mean'),
        Model_Count=('Mobile', 'count')
    )
    .sort_values(by='Avg_Rating', ascending=False)
    .round(2)
)
print(q5)

# ==============================================================================
# Q6. How mobile storage capacity affects average selling price
# ==============================================================================
print("\n=== Q6: Selling Price by Storage Capacity ===")
q6 = (
    df.groupby('Storage_GB')['Selling Price']
    .agg(Count='count', Avg_Price='mean', Median_Price='median', Min_Price='min', Max_Price='max')
    .sort_index()
    .round(2)
)
print(q6)

# ==============================================================================
# Q7. Mobile models providing highest discount amount & percentage discounted
# ==============================================================================
print("\n=== Q7: Top 10 Mobile Models by Discount Amount ===")
q7 = (
    df[['Brands', 'Mobile_clean', 'Colors', 'Original Price', 'Selling Price', 'Discount', 'Discount_Pct']]
    .sort_values(by='Discount', ascending=False)
    .head(10)
    .round(2)
)
print(q7.to_string(index=False))

# ==============================================================================
# Q8. Combination of Memory and Storage offering best balance of Price & Rating
# ==============================================================================
print("\n=== Q8: Best Memory & Storage Combinations (Min 10 Models, Smartphone Tiers) ===")
q8 = (
    df[(df['Memory_num'] <= 16) & (df['Storage'] >= 8)]
    .groupby(['Memory_num', 'Storage'])
    .agg(
        Count=('Mobile', 'count'),
        Avg_Price=('Selling Price', 'mean'),
        Median_Price=('Selling Price', 'median'),
        Avg_Rating=('Rating', 'mean')
    )
    .reset_index()
)
q8 = q8[q8['Count'] >= 10].copy()
q8['Norm_Rating'] = (q8['Avg_Rating'] - q8['Avg_Rating'].min()) / (q8['Avg_Rating'].max() - q8['Avg_Rating'].min())
q8['Norm_Price_Inv'] = 1 - (q8['Avg_Price'] - q8['Avg_Price'].min()) / (q8['Avg_Price'].max() - q8['Avg_Price'].min())
q8['Balance_Score'] = 0.5 * q8['Norm_Rating'] + 0.5 * q8['Norm_Price_Inv']
print(q8.sort_values(by='Balance_Score', ascending=False).round(3).head(10).to_string(index=False))

# ==============================================================================
# Q9. Top 10 mobile models providing best value for money
# ==============================================================================
print("\n=== Q9: Top 10 Mobile Models for Value for Money ===")
df_smart = df.dropna(subset=['RAM_GB']).copy()
df_smart = df_smart[(df_smart['Selling Price'] >= 5000) & (df_smart['Memory_num'] <= 18)].copy()

scaler = MinMaxScaler()
scaled_cols = scaler.fit_transform(df_smart[['Rating', 'Discount', 'Storage_GB', 'RAM_GB', 'Selling Price']])
df_smart['norm_Rating'] = scaled_cols[:, 0]
df_smart['norm_Discount'] = scaled_cols[:, 1]
df_smart['norm_Storage'] = scaled_cols[:, 2]
df_smart['norm_RAM'] = scaled_cols[:, 3]
df_smart['norm_Price_Inv'] = 1 - scaled_cols[:, 4]

df_smart['Value_Score'] = (
    df_smart['norm_Rating'] +
    df_smart['norm_Discount'] +
    df_smart['norm_Storage'] +
    df_smart['norm_RAM'] +
    df_smart['norm_Price_Inv']
) / 5.0

q9_top10 = (
    df_smart.sort_values(by='Value_Score', ascending=False)
    .drop_duplicates(subset=['Mobile_clean'])
    [['Brands', 'Mobile_clean', 'Selling Price', 'Discount', 'RAM_GB', 'Storage_GB', 'Rating', 'Value_Score']]
    .head(10)
    .round(4)
)
print(q9_top10.to_string(index=False))

# ==============================================================================
# Q10. Factors determining whether a mobile is classified as a "Good Product"
# ==============================================================================
print("\n=== Q10: Factors Determining 'Good Product' ===")
df_q10 = df.dropna(subset=['Memory_num']).copy()

print("1. Exact Rating Threshold Verification:")
print(pd.crosstab(df_q10['Rating'] >= 4.0, df_q10['Good Product'], rownames=['Rating >= 4.0']))

print("\n2. Correlation with 'Good Product':")
corr_features = ['Rating', 'RAM_GB', 'Storage_GB', 'Selling Price', 'Original Price', 'Discount', 'Good Product']
print(df_q10[corr_features].corr()['Good Product'].sort_values(ascending=False).round(4))

print("\n3. Random Forest Feature Importance (With Rating):")
X = df_q10[['Selling Price', 'Discount', 'Rating', 'RAM_GB', 'Storage_GB']]
y = df_q10['Good Product']
rf = RandomForestClassifier(random_state=42).fit(X, y)
for feat, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"   {feat}: {imp:.4f}")