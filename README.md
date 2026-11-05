# Machine Learning and Data Analysis Projects

This repository contains three standalone Python data-analysis projects:

1. **Mobile price analysis** - explores mobile brands, prices, discounts, specifications, ratings, and product value.
2. **Fast-food nutrition analysis** - compares menu items and restaurants using calories, nutrients, and health-risk scores.
3. **Cricket batting analysis** - summarizes player, team, match, batting, and optional bowling performance.

The scripts print their results to the terminal. They do not create output files.

## Project structure

```text
.
├── batting.py
├── Batting.csv
├── food.py
├── fastfood.csv
├── mobile.py
└── Mobile_Price_ML_Analysis.xlsx
```

## Requirements

- Python 3.9 or newer
- pandas
- NumPy
- scikit-learn
- openpyxl (required by pandas to read the Excel workbook)

Install the dependencies with:

```bash
python -m pip install pandas numpy scikit-learn openpyxl
```

## Usage

Run commands from the directory containing the scripts and datasets:

```bash
python mobile.py
python food.py
python batting.py
```

The scripts use relative file paths, so running them from another directory may cause a file-not-found error.

## Mobile price analysis

`mobile.py` reads the `Clean Data` worksheet from `Mobile_Price_ML_Analysis.xlsx`.

The analysis reports:

- Brands with the most mobile variants and unique models
- The highest-priced models and average brand prices
- Average discount amounts and percentages
- Customer ratings compared with selling prices
- The relationship between storage capacity and price
- Models with the largest discounts
- Memory and storage combinations with a price-rating balance
- Top value-for-money smartphones
- Factors associated with the `Good Product` classification using correlations and a random forest

The script also standardizes memory and storage values before comparing devices.

## Fast-food nutrition analysis

`food.py` reads `fastfood.csv`.

The analysis reports:

- Menu-item counts by restaurant
- Highest-calorie and highest-sugar items
- Average calories, protein, and total fat by restaurant
- Salad-versus-other nutritional comparisons
- Restaurants with a favorable low-calorie/high-protein balance
- A health-risk score based on calories, fat, saturated fat, trans fat, cholesterol, and sodium
- An overall restaurant menu health ranking using positive and negative nutrition metrics

## Cricket batting analysis

`batting.py` reads `Batting.csv` and cleans player, team, opponent, and date values before calculating:

- Top run scorers
- Highest strike rates among players facing at least 50 balls
- Players with the most fours and sixes
- Team win totals and win percentages
- Highest run scorer in each match
- Batting-first versus chasing performance
- Boundary scoring and player performance scores

### Optional bowling data

If a CSV or Excel file whose name contains `bowl` is present in the same directory, the script also calculates:

- Top bowlers by total wickets and average economy
- Top all-round performers using a match-impact score

Without that file, the script falls back to extracting bowler names from the `Dismissal` column and prints a notice that full all-rounder analysis is unavailable.

## Notes

- Results depend on the contents and quality of the supplied datasets.
- The scripts are exploratory analyses rather than production machine-learning pipelines.
- Run each script independently to view its labeled Q1-Q10 output.
