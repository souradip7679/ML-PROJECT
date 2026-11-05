import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# Load & Clean Batting Data
# ==============================================================================
bat_df = pd.read_csv('Batting.csv')

# Clean Player names (strip captain '(c)' and keeper '?' symbols)
bat_df['Player_Clean'] = (
    bat_df['Player']
    .astype(str)
    .str.replace(r'\(c\)', '', regex=True)
    .str.replace('?', '', regex=False)
    .str.replace('†', '', regex=False)
    .str.strip()
)

# Standardize Team and Opponent names
team_map = {
    'New Zealand ': 'New Zealand',
    'New zealand': 'New Zealand',
    'ENG': 'England',
    'Zimbabwae': 'Zimbabwe',
    'Zimbawe': 'Zimbabwe',
    'Zimbawe ': 'Zimbabwe',
    'Afganisthan': 'Afghanistan',
    'Afganistan': 'Afghanistan',
    'Sri Lanka ': 'Sri Lanka',
    'Australia ': 'Australia',
    'Netherland': 'Netherlands'
}
bat_df['Team_Std'] = bat_df['Team'].astype(str).str.strip().replace(team_map)
bat_df['Vs_Std'] = bat_df['Vs'].astype(str).str.strip().replace(team_map)

# Standardize Dates (e.g. '7.2.26' -> '07.02.26') and create unique Match_ID
def std_date(d):
    parts = str(d).strip().split('.')
    if len(parts) == 3:
        return f"{int(parts[0]):02d}.{int(parts[1]):02d}.{parts[2]}"
    return str(d).strip()

bat_df['Date_Std'] = bat_df['Date'].apply(std_date)
bat_df['Date_Parsed'] = pd.to_datetime(bat_df['Date_Std'], format='%d.%m.%y')
bat_df['Match_ID'] = bat_df.apply(
    lambda r: f"{r['Date_Std']} | {min(r['Team_Std'], r['Vs_Std'])} vs {max(r['Team_Std'], r['Vs_Std'])}",
    axis=1
)

# ==============================================================================
# Q1. Top 10 batsmen based on total runs scored across all matches
# ==============================================================================
print("=== Q1: Top 10 Batsmen by Total Runs ===")
player_summary = (
    bat_df.groupby('Player_Clean')
    .agg(
        Team=('Team_Std', 'first'),
        Innings=('R', 'count'),
        Total_Runs=('R', 'sum'),
        Total_Balls=('B', 'sum'),
        Fours=('4s', 'sum'),
        Sixes=('6s', 'sum'),
        Mean_SR=('SR', 'mean'),
        Wins=('Win', 'sum')
    )
    .reset_index()
)
player_summary['Overall_SR'] = np.where(
    player_summary['Total_Balls'] > 0,
    (player_summary['Total_Runs'] / player_summary['Total_Balls']) * 100,
    0.0
)

q1_top10 = player_summary.sort_values(by='Total_Runs', ascending=False).head(10).round(2)
print(q1_top10[['Player_Clean', 'Team', 'Innings', 'Total_Runs', 'Total_Balls', 'Fours', 'Sixes', 'Overall_SR']].to_string(index=False))

# ==============================================================================
# Q2. Top 10 players with highest strike rate (at least 50 balls faced)
# ==============================================================================
print("\n=== Q2: Top 10 Players by Strike Rate (Min 50 Balls Faced) ===")
q2_top10 = (
    player_summary[player_summary['Total_Balls'] >= 50]
    .sort_values(by='Overall_SR', ascending=False)
    .head(10)
    .round(2)
)
print(q2_top10[['Player_Clean', 'Team', 'Total_Runs', 'Total_Balls', 'Fours', 'Sixes', 'Overall_SR', 'Mean_SR']].to_string(index=False))

# ==============================================================================
# Q3. Player with most fours and player with most sixes
# ==============================================================================
print("\n=== Q3: Most Fours and Most Sixes ===")
most_4s = player_summary.sort_values(by=['Fours', 'Total_Runs'], ascending=False).iloc[0]
most_6s = player_summary.sort_values(by=['Sixes', 'Total_Runs'], ascending=False).iloc[0]
print(f"Most Fours: {most_4s['Player_Clean']} ({most_4s['Team']}) - {most_4s['Fours']} fours ({most_4s['Total_Runs']} runs)")
print(f"Most Sixes: {most_6s['Player_Clean']} ({most_6s['Team']}) - {most_6s['Sixes']} sixes ({most_6s['Total_Runs']} runs)")

# ==============================================================================
# Q4. Team with the highest number of match wins
# ==============================================================================
print("\n=== Q4: Matches Won by Each Team ===")
innings_summary = (
    bat_df.groupby(['Match_ID', 'Date_Std', 'Team_Std', 'Vs_Std', 'Chase', 'Win'])
    .agg(Team_Runs=('R', 'sum'), Team_Balls=('B', 'sum'))
    .reset_index()
)
team_wins = (
    innings_summary.groupby('Team_Std')
    .agg(Matches_Played=('Match_ID', 'count'), Matches_Won=('Win', 'sum'))
    .reset_index()
)
team_wins['Win_Pct'] = (team_wins['Matches_Won'] / team_wins['Matches_Played'] * 100).round(2)
print(team_wins.sort_values(by=['Matches_Won', 'Win_Pct'], ascending=False).to_string(index=False))

# ==============================================================================
# Q5. Highest run scorer for each match
# ==============================================================================
print("\n=== Q5: Highest Run Scorer for Each Match ===")
q5_match_top = (
    bat_df.sort_values(by=['R', 'SR', 'B'], ascending=[False, False, True])
    .groupby('Match_ID', as_index=False)
    .first()
    .sort_values(by=['Date_Parsed', 'Match_ID'])
)
print(q5_match_top[['Date', 'Team_Std', 'Vs_Std', 'Player_Clean', 'R', 'B', 'SR']].to_string(index=False))

# ==============================================================================
# Q6. Compare performance while Chasing (Chase=1) vs Batting First (Chase=0)
# ==============================================================================
print("\n=== Q6: Batting First (Chase=0) vs Chasing (Chase=1) ===")
q6_team = (
    innings_summary.groupby('Chase')
    .agg(
        Total_Innings=('Match_ID', 'count'),
        Matches_Won=('Win', 'sum'),
        Win_Percentage=('Win', lambda x: round(x.mean() * 100, 2)),
        Avg_Team_Runs=('Team_Runs', 'mean'),
        Median_Team_Runs=('Team_Runs', 'median'),
        Total_Runs=('Team_Runs', 'sum'),
        Total_Balls=('Team_Balls', 'sum')
    )
    .reset_index()
)
q6_team['Overall_Team_SR'] = (q6_team['Total_Runs'] / q6_team['Total_Balls'] * 100).round(2)
print(q6_team.round(2).to_string(index=False))

# ==============================================================================
# Q7. Top 10 bowlers based on total wickets taken and average economy rates
# ==============================================================================
print("\n=== Q7: Top 10 Bowlers by Total Wickets & Average Economy Rate ===")
bowling_file = next((f for f in os.listdir() if 'bowl' in f.lower() and f.endswith(('.csv', '.xlsx'))), None)

if bowling_file:
    bowl_df = pd.read_csv(bowling_file) if bowling_file.endswith('.csv') else pd.read_excel(bowling_file)
    # Standardize column names dynamically
    col_map = {}
    for c in bowl_df.columns:
        cl = c.strip().lower()
        if cl in ['player', 'bowler', 'name']: col_map[c] = 'Player'
        elif cl in ['w', 'wkts', 'wickets']: col_map[c] = 'Wickets'
        elif cl in ['econ', 'economy', 'ec', 'economy rate']: col_map[c] = 'Economy'
        elif cl in ['o', 'overs']: col_map[c] = 'Overs'
        elif cl in ['r', 'runs']: col_map[c] = 'Runs_Conceded'
    bowl_df = bowl_df.rename(columns=col_map)
    bowl_df['Player_Clean'] = bowl_df['Player'].astype(str).str.replace(r'\(c\)|\?|†', '', regex=True).str.strip()
    
    q7_bowlers = (
        bowl_df.groupby('Player_Clean')
        .agg(Total_Wickets=('Wickets', 'sum'), Avg_Economy=('Economy', 'mean'))
        .reset_index()
        .sort_values(by=['Total_Wickets', 'Avg_Economy'], ascending=[False, True])
        .head(10)
        .round(2)
    )
    print(q7_bowlers.to_string(index=False))
else:
    print("Note: 'Bowling.csv' not found in directory. Extracting bowler wickets from 'Dismissal' column as fallback:")
    def extract_bowler(d):
        m = re.search(r'\bb\s+([A-Za-z\s\-\']+)$', str(d).strip())
        return m.group(1).strip() if m else None
    bat_df['Bowler_Extracted'] = bat_df['Dismissal'].apply(extract_bowler)
    print(bat_df['Bowler_Extracted'].value_counts().head(10))

# ==============================================================================
# Q8. Most runs through boundaries (4s and 6s) and strike rate comparison
# ==============================================================================
print("\n=== Q8: Top 10 Players by Boundary Runs & Strike Rate Comparison ===")
player_summary['Boundary_Runs'] = (player_summary['Fours'] * 4) + (player_summary['Sixes'] * 6)
player_summary['Boundary_Pct'] = np.where(
    player_summary['Total_Runs'] > 0,
    (player_summary['Boundary_Runs'] / player_summary['Total_Runs']) * 100,
    0.0
)
q8_top10 = player_summary.sort_values(by='Boundary_Runs', ascending=False).head(10).round(2)
print(q8_top10[['Player_Clean', 'Team', 'Total_Runs', 'Fours', 'Sixes', 'Boundary_Runs', 'Boundary_Pct', 'Overall_SR']].to_string(index=False))

others = player_summary[~player_summary['Player_Clean'].isin(q8_top10['Player_Clean'])]
print(f"\nTop 10 Boundary Scorers Aggregate SR: {(q8_top10['Total_Runs'].sum() / q8_top10['Total_Balls'].sum()) * 100:.2f}")
print(f"All Other Players Aggregate SR:       {(others['Total_Runs'].sum() / others['Total_Balls'].sum()) * 100:.2f}")

# ==============================================================================
# Q9. Player Performance Score (Top 20 Batsmen)
# ==============================================================================
print("\n=== Q9: Top 20 Batsmen by Player Performance Score ===")
qualified_bat = player_summary[player_summary['Total_Balls'] >= 20].copy()
scaler = MinMaxScaler()
norm_cols = scaler.fit_transform(qualified_bat[['Total_Runs', 'Overall_SR', 'Fours', 'Sixes', 'Total_Balls']])

qualified_bat['Performance_Score_Norm'] = (
    0.35 * norm_cols[:, 0] +
    0.25 * norm_cols[:, 1] +
    0.15 * norm_cols[:, 2] +
    0.15 * norm_cols[:, 3] +
    0.10 * norm_cols[:, 4]
) * 100

qualified_bat['Performance_Points'] = (
    qualified_bat['Total_Runs'] +
    0.25 * qualified_bat['Overall_SR'] +
    2 * qualified_bat['Fours'] +
    3 * qualified_bat['Sixes'] +
    0.10 * qualified_bat['Total_Balls']
)

q9_top20 = qualified_bat.sort_values(by='Performance_Score_Norm', ascending=False).head(20).round(2)
print(q9_top20[['Player_Clean', 'Team', 'Total_Runs', 'Total_Balls', 'Fours', 'Sixes', 'Overall_SR', 'Performance_Points', 'Performance_Score_Norm']].to_string(index=False))

# ==============================================================================
# Q10. Combine Batting & Bowling Data for Top 10 All-Round Performers
# ==============================================================================
print("\n=== Q10: Top 10 All-Round Performers (Match Impact Score) ===")
if bowling_file:
    bowl_agg = (
        bowl_df.groupby('Player_Clean')
        .agg(Total_Wickets=('Wickets', 'sum'), Avg_Economy=('Economy', 'mean'))
        .reset_index()
    )
    all_round = pd.merge(player_summary, bowl_agg, on='Player_Clean', how='inner')
    # Filter for true all-rounders (contributed with both bat and ball)
    all_round = all_round[(all_round['Total_Runs'] >= 20) & (all_round['Total_Wickets'] >= 1)].copy()
    
    all_round['Economy_Bonus'] = np.maximum(0, (10.0 - all_round['Avg_Economy']) * 5.0)
    all_round['Match_Impact_Score'] = (
        all_round['Total_Runs'] * 1.0 +
        all_round['Total_Wickets'] * 20.0 +
        all_round['Fours'] * 2.0 +
        all_round['Sixes'] * 3.0 +
        all_round['Overall_SR'] * 0.2 +
        all_round['Economy_Bonus'] +
        all_round['Wins'] * 10.0
    )
    q10_top10 = all_round.sort_values(by='Match_Impact_Score', ascending=False).head(10).round(2)
    print(q10_top10[['Player_Clean', 'Team', 'Total_Runs', 'Overall_SR', 'Total_Wickets', 'Avg_Economy', 'Wins', 'Match_Impact_Score']].to_string(index=False))
else:
    print("Please place 'Bowling.csv' in the working directory to compute full Batting + Bowling Match Impact Scores.")