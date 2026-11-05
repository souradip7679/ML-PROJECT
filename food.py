import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('fastfood.csv')

# ==============================================================================
# Q1. Restaurant with the highest number of food items
# ==============================================================================
print("=== Q1: Number of Food Items per Restaurant ===")
q1 = df['restaurant'].value_counts()
print(q1)
print(f"\nHighest: {q1.index[0]} with {q1.iloc[0]} items.")

# ==============================================================================
# Q2. Top 10 food items with the highest number of calories
# ==============================================================================
print("\n=== Q2: Top 10 Food Items by Calories ===")
q2 = (
    df[['restaurant', 'item', 'calories', 'cal_fat', 'total_fat', 'protein']]
    .sort_values(by='calories', ascending=False)
    .head(10)
)
print(q2.to_string(index=False))

# ==============================================================================
# Q3. Average calorie count of food items for each restaurant
# ==============================================================================
print("\n=== Q3: Average Calorie Count by Restaurant ===")
q3 = (
    df.groupby('restaurant')['calories']
    .agg(Count='count', Mean_Calories='mean', Median_Calories='median', Std_Calories='std')
    .sort_values(by='Mean_Calories', ascending=False)
    .round(2)
)
print(q3)

# ==============================================================================
# Q4. Restaurant offering food items with highest average protein content
# ==============================================================================
print("\n=== Q4: Average Protein Content by Restaurant ===")
q4 = (
    df.groupby('restaurant')['protein']
    .agg(Count='count', Mean_Protein='mean', Median_Protein='median', Std_Protein='std')
    .sort_values(by='Mean_Protein', ascending=False)
    .round(2)
)
print(q4)

# ==============================================================================
# Q5. Restaurant with highest and lowest average total fat
# ==============================================================================
print("\n=== Q5: Average Total Fat by Restaurant ===")
q5 = (
    df.groupby('restaurant')['total_fat']
    .agg(Count='count', Mean_Fat='mean', Median_Fat='median', Min_Fat='min', Max_Fat='max')
    .sort_values(by='Mean_Fat', ascending=False)
    .round(2)
)
print(q5)
print(f"\nHighest Avg Total Fat: {q5.index[0]} ({q5['Mean_Fat'].iloc[0]}g)")
print(f"Lowest Avg Total Fat:  {q5.index[-1]} ({q5['Mean_Fat'].iloc[-1]}g)")

# ==============================================================================
# Q6. Food items containing highest amount of sugar and their calories
# ==============================================================================
print("\n=== Q6: Top 10 Food Items by Sugar Content ===")
q6 = (
    df[['restaurant', 'item', 'sugar', 'calories', 'total_carb', 'total_fat']]
    .sort_values(by=['sugar', 'calories'], ascending=[False, False])
    .head(10)
)
print(q6.to_string(index=False))

# ==============================================================================
# Q7. How food item type (Salad vs Other) affects calories, fat, carbs, protein
# ==============================================================================
print("\n=== Q7: Salad vs Other Nutritional Comparison ===")
print("1. Grouped by raw 'salad' column:")
print(df.groupby('salad')[['calories', 'total_fat', 'total_carb', 'protein']].mean().round(2))

df['item_type'] = np.where(df['item'].str.contains('salad', case=False), 'Salad', 'Other')
print("\n2. Grouped by whether item name contains 'Salad':")
q7_name = (
    df.groupby('item_type')[['calories', 'total_fat', 'total_carb', 'protein']]
    .agg(['count', 'mean', 'median'])
    .round(2)
)
print(q7_name)

# ==============================================================================
# Q8. Best combination of low calories and high protein
# ==============================================================================
print("\n=== Q8: Low Calories & High Protein Combination by Restaurant ===")
df['protein_per_100cal'] = (df['protein'] / df['calories']) * 100
q8 = (
    df.groupby('restaurant')
    .agg(
        Avg_Calories=('calories', 'mean'),
        Avg_Protein=('protein', 'mean'),
        Avg_Total_Fat=('total_fat', 'mean'),
        Protein_per_100Cal=('protein_per_100cal', 'mean')
    )
    .reset_index()
)
q8['Norm_Protein'] = (q8['Avg_Protein'] - q8['Avg_Protein'].min()) / (q8['Avg_Protein'].max() - q8['Avg_Protein'].min())
q8['Norm_Cal_Inv'] = 1 - (q8['Avg_Calories'] - q8['Avg_Calories'].min()) / (q8['Avg_Calories'].max() - q8['Avg_Calories'].min())
q8['Balance_Score'] = (0.5 * q8['Norm_Protein'] + 0.5 * q8['Norm_Cal_Inv']) * 100
print(q8.sort_values(by='Protein_per_100Cal', ascending=False).round(2).to_string(index=False))

# ==============================================================================
# Q9. Identify the 10 least healthy food items using a Health Risk Score
# ==============================================================================
print("\n=== Q9: Top 10 Least Healthy Food Items (Health Risk Score) ===")
risk_cols = ['calories', 'total_fat', 'sat_fat', 'trans_fat', 'cholesterol', 'sodium']
scaler = MinMaxScaler()
df['Health_Risk_Score'] = scaler.fit_transform(df[risk_cols]).mean(axis=1) * 100

q9 = (
    df[['restaurant', 'item'] + risk_cols + ['Health_Risk_Score']]
    .sort_values(by='Health_Risk_Score', ascending=False)
    .head(10)
    .round(2)
)
print(q9.to_string(index=False))

# ==============================================================================
# Q10. Compare nutritional profiles & rank healthiest overall restaurant menu
# ==============================================================================
print("\n=== Q10: Overall Restaurant Menu Health Ranking ===")
q10 = df.groupby('restaurant')[['calories', 'total_fat', 'sugar', 'sodium', 'fiber', 'protein', 'vit_a', 'vit_c']].mean()
q10['vitamins'] = q10[['vit_a', 'vit_c']].mean(axis=1).fillna(0)

neg_metrics = ['calories', 'total_fat', 'sugar', 'sodium']
pos_metrics = ['fiber', 'protein', 'vitamins']

for col in neg_metrics:
    q10[f'score_{col}'] = 1 - (q10[col] - q10[col].min()) / (q10[col].max() - q10[col].min())
for col in pos_metrics:
    q10[f'score_{col}'] = (q10[col] - q10[col].min()) / (q10[col].max() - q10[col].min())

score_cols = [f'score_{col}' for col in neg_metrics + pos_metrics]
q10['Menu_Health_Score'] = q10[score_cols].mean(axis=1) * 100

q10_result = q10[['calories', 'total_fat', 'sugar', 'sodium', 'fiber', 'protein', 'vitamins', 'Menu_Health_Score']].sort_values(by='Menu_Health_Score', ascending=False).round(2)
print(q10_result)