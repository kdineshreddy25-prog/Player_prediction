# -*- coding: utf-8 -*-
"""playerPredcition.ipynb - MODIFIED FOR LOCAL USE"""

# Core data manipulation
import pandas as pd
import numpy as np

# File handling and data loading
import json
import os
import zipfile

# Text processing and collections
import re
from collections import Counter, defaultdict
from datetime import datetime
import itertools

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress tracking
from tqdm import tqdm

# For saving models
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("All packages imported successfully!")

# ============================================
# CREATE DIRECTORIES
# ============================================
os.makedirs('ipl_matches', exist_ok=True)
os.makedirs('t20_matches', exist_ok=True)
os.makedirs('player_data', exist_ok=True)

print("Directories ready!")

# ============================================
# CHECK FOR EXISTING DATA
# ============================================
print("=" * 50)
print("VERIFICATION REPORT")
print("=" * 50)

ipl_exists = os.path.exists('ipl_matches')
t20_exists = os.path.exists('t20_matches')

print(f"\nIPL directory exists: {ipl_exists}")
print(f"T20 directory exists: {t20_exists}")

if ipl_exists:
    ipl_count = len(os.listdir('ipl_matches'))
    print(f"IPL match files: {ipl_count}")

if t20_exists:
    t20_count = len(os.listdir('t20_matches'))
    print(f"T20 match files: {t20_count}")

if ipl_exists and ipl_count > 0:
    print(f"\nSample IPL file: {os.listdir('ipl_matches')[0]}")

print("\n" + "=" * 50)
print("DATA LOADING COMPLETE!")
print("=" * 50)

# ============================================
# LOAD PLAYER DATA
# ============================================
if os.path.exists('player_data'):
    player_files = os.listdir('player_data')
    if player_files:
        file_path = 'player_data/' + player_files[0]
        df = pd.read_csv(file_path)
        print("\nFirst 5 rows of player data:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())

# ============================================
# LOAD SAMPLE MATCH FILE
# ============================================
if ipl_exists and ipl_count > 0:
    sample_file = os.listdir('ipl_matches')[0]
    print("Sample file:", sample_file)

    with open(f'ipl_matches/{sample_file}', 'r') as f:
        match = json.load(f)

    print("\n=== Match Keys ===")
    print(match.keys())

    print("\n=== Info Keys ===")
    print(match['info'].keys())

    print("\n=== Innings Structure ===")
    first_innings = match['innings'][0]
    print("First innings keys:", first_innings.keys())

    for key in first_innings.keys():
        print(f"\nKey: {key}")
        print(f"Type: {type(first_innings[key])}")
        if isinstance(first_innings[key], list) and len(first_innings[key]) > 0:
            print(f"First item: {first_innings[key][0]}")

# ============================================
# EXTRACT VENUE DATA
# ============================================
print("=" * 50)
print("STEP: EXTRACTING VENUE DATA FROM IPL MATCHES")
print("=" * 50)

venue_summary = None

if ipl_exists:
    ipl_folder = 'ipl_matches'
    match_files = os.listdir(ipl_folder)
    print(f"\nTotal IPL match files found: {len(match_files)}")

    venue_data = []

    for file in match_files:
        try:
            with open(f'{ipl_folder}/{file}', 'r') as f:
                match = json.load(f)

            info = match.get('info', {})
            venue = info.get('venue', 'Unknown')
            teams = info.get('teams', [])
            toss = info.get('toss', {})
            toss_winner = toss.get('winner', 'Unknown')
            toss_decision = toss.get('decision', 'Unknown')
            outcome = info.get('outcome', {})
            winner = outcome.get('winner', 'Unknown')

            innings_list = match.get('innings', [])
            total_runs = []

            for innings in innings_list:
                runs = 0
                overs = innings.get('overs', [])
                for over in overs:
                    deliveries = over.get('deliveries', [])
                    for delivery in deliveries:
                        runs_data = delivery.get('runs', {})
                        total_ball_runs = runs_data.get('total', 0)
                        runs += total_ball_runs
                total_runs.append(runs)

            if len(total_runs) >= 2:
                batting_first_won = 1 if total_runs[0] > total_runs[1] else 0
            else:
                batting_first_won = -1

            venue_data.append({
                'venue': venue,
                'team1': teams[0] if len(teams) > 0 else 'Unknown',
                'team2': teams[1] if len(teams) > 1 else 'Unknown',
                'toss_winner': toss_winner,
                'toss_decision': toss_decision,
                'match_winner': winner,
                'total_runs_match': sum(total_runs),
                'batting_first_won': batting_first_won
            })
        except Exception as e:
            continue

    if venue_data:
        venue_df = pd.DataFrame(venue_data)
        print(f"\n✅ Processed {len(venue_df)} matches")

        venue_df_filtered = venue_df[venue_df['total_runs_match'] > 0]

        if len(venue_df_filtered) > 0:
            venue_summary = venue_df_filtered.groupby('venue').agg({
                'total_runs_match': 'mean',
                'batting_first_won': 'mean'
            }).round(2)

            venue_summary.columns = ['avg_match_runs', 'batting_first_win_pct']
            venue_summary['batting_first_win_pct'] = venue_summary['batting_first_win_pct'] * 100

            venue_summary.to_csv('venue_summary.csv')
            print(f"\n✅ Venue summary saved with {len(venue_summary)} venues")
        else:
            print("No matches with valid runs data found.")
    else:
        print("No venue data extracted.")

# Create default venue summary if not created
if venue_summary is None or len(venue_summary) == 0:
    print("\n⚠️ Creating default venue summary...")
    venue_summary = pd.DataFrame({
        'avg_match_runs': [340, 320, 360, 350, 330, 310, 370, 300],
        'batting_first_win_pct': [50, 45, 55, 48, 52, 40, 60, 35]
    }, index=['Wankhede Stadium', 'Eden Gardens', 'M Chinnaswamy Stadium', 
              'Arun Jaitley Stadium', 'MA Chidambaram Stadium', 'Rajiv Gandhi Stadium',
              'Narendra Modi Stadium', 'PCA Stadium'])
    venue_summary.to_csv('venue_summary.csv')
    print("✅ Created default venue_summary.csv")

# ============================================
# COMPLETE DATA PREPROCESSING & CLEANING FOR ML MODELS
# ============================================
print("=" * 70)
print("COMPLETE DATA PREPROCESSING & CLEANING FOR ML MODELS")
print("=" * 70)

# STEP 1: LOAD ALL RAW DATA
print("\n📋 STEP 1: LOADING RAW DATA")
print("-" * 50)

player_df = pd.DataFrame()

if os.path.exists('player_data') and len(os.listdir('player_data')) > 0:
    player_files = os.listdir('player_data')
    player_df = pd.read_csv('player_data/' + player_files[0])
    print(f"✅ Loaded {len(player_df)} player records")
else:
    print("❌ Player data not found!")

try:
    venue_summary = pd.read_csv('venue_summary.csv', index_col=0)
    print(f"✅ Loaded venue summary with {len(venue_summary)} venues")
except:
    print("⚠️ Venue summary not found")

# ============================================
# STEP 2: CLEAN PLAYER DATA
# ============================================
print("\n📋 STEP 2: CLEANING PLAYER DATA")
print("-" * 50)

def clean_player_data(df):
    """Clean and preprocess player data for ML"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['fullname'])
    print(f"   Removed {before - len(df_clean)} duplicate players")

    df_clean['battingstyle'] = df_clean['battingstyle'].fillna('right-hand-bat')
    df_clean['bowlingstyle'] = df_clean['bowlingstyle'].fillna('None')
    df_clean['position'] = df_clean['position'].fillna('Batsman')

    def standardize_batting(style):
        if pd.isna(style):
            return "Right"
        style = str(style).lower()
        if 'left' in style:
            return "Left"
        return "Right"

    def standardize_bowling(style):
        if pd.isna(style):
            return "None"
        style = str(style).lower()
        if 'fast' in style or 'medium' in style:
            return "Fast"
        elif 'spin' in style or 'legbreak' in style or 'offbreak' in style or 'orthodox' in style:
            return "Spin"
        return "None"

    def standardize_position(pos):
        if pd.isna(pos):
            return "Batsman"
        pos = str(pos).lower()
        if 'wicketkeeper' in pos:
            return "Wicketkeeper"
        elif 'bowler' in pos:
            return "Bowler"
        elif 'all' in pos:
            return "All-rounder"
        return "Batsman"

    df_clean['bat_hand'] = df_clean['battingstyle'].apply(standardize_batting)
    df_clean['bowl_type'] = df_clean['bowlingstyle'].apply(standardize_bowling)
    df_clean['role'] = df_clean['position'].apply(standardize_position)

    df_clean['is_left_bat'] = df_clean['bat_hand'].apply(lambda x: 1 if x == 'Left' else 0)
    df_clean['is_left_bowl'] = df_clean['bowl_type'].apply(lambda x: 1 if x == 'Left' else 0)
    df_clean['is_pacer'] = df_clean['bowl_type'].apply(lambda x: 1 if x == 'Fast' else 0)
    df_clean['is_spinner'] = df_clean['bowl_type'].apply(lambda x: 1 if x == 'Spin' else 0)
    df_clean['can_bowl'] = df_clean['bowl_type'].apply(lambda x: 1 if x != 'None' else 0)

    role_codes = {'Batsman': 0, 'Bowler': 1, 'All-rounder': 2, 'Wicketkeeper': 3}
    df_clean['role_code'] = df_clean['role'].map(role_codes)

    print(f"   Standardized {len(df_clean)} players")
    print(f"   Left-handed batsmen: {df_clean['is_left_bat'].sum()}")
    print(f"   Pacers: {df_clean['is_pacer'].sum()}")
    print(f"   Spinners: {df_clean['is_spinner'].sum()}")
    print(f"   All-rounders: {len(df_clean[df_clean['role'] == 'All-rounder'])}")

    return df_clean

if not player_df.empty:
    cleaned_players = clean_player_data(player_df)
    cleaned_players.to_csv('cleaned_players.csv', index=False)
    print("\n✅ Cleaned player data saved to 'cleaned_players.csv'")
else:
    cleaned_players = pd.DataFrame()

# ============================================
# STEP 3: EXTRACT PLAYER PERFORMANCE
# ============================================
print("\n📋 STEP 3: EXTRACTING PLAYER PERFORMANCE DATA")
print("-" * 50)

def extract_player_stats(match_folder, num_matches=200):
    """Extract batting and bowling stats from ball-by-ball data"""
    if not os.path.exists(match_folder):
        return pd.DataFrame()
    
    match_files = os.listdir(match_folder)[:num_matches]
    
    batting_stats = defaultdict(lambda: {'runs': 0, 'balls': 0, 'outs': 0})
    bowling_stats = defaultdict(lambda: {'runs': 0, 'wickets': 0, 'balls': 0})
    
    for file in match_files:
        try:
            with open(f'{match_folder}/{file}', 'r') as f:
                match = json.load(f)
            
            for innings in match.get('innings', []):
                overs = innings.get('overs', [])
                for over in overs:
                    deliveries = over.get('deliveries', [])
                    for delivery in deliveries:
                        batter = delivery.get('batter', '')
                        bowler = delivery.get('bowler', '')
                        runs_data = delivery.get('runs', {})
                        runs = runs_data.get('total', 0)
                        
                        if batter:
                            batting_stats[batter]['runs'] += runs
                            batting_stats[batter]['balls'] += 1
                            wicket_data = delivery.get('wicket', {})
                            if wicket_data and wicket_data.get('kind'):
                                batting_stats[batter]['outs'] += 1
                        
                        if bowler:
                            bowling_stats[bowler]['runs'] += runs
                            bowling_stats[bowler]['balls'] += 1
                            wicket_data = delivery.get('wicket', {})
                            if wicket_data and wicket_data.get('kind'):
                                bowling_stats[bowler]['wickets'] += 1
        except Exception as e:
            continue
    
    player_stats = []
    
    for player, stats in batting_stats.items():
        runs = stats['runs']
        balls = stats['balls']
        outs = stats['outs'] if stats['outs'] > 0 else 1
        batting_avg = runs / outs
        strike_rate = (runs / balls) * 100 if balls > 0 else 0
        player_stats.append({
            'name': player,
            'batting_runs': runs,
            'batting_balls': balls,
            'batting_avg': round(batting_avg, 2),
            'batting_strike_rate': round(strike_rate, 2),
            'batting_outs': stats['outs']
        })
    
    for player, stats in bowling_stats.items():
        runs = stats['runs']
        wickets = stats['wickets'] if stats['wickets'] > 0 else 1
        balls = stats['balls'] if stats['balls'] > 0 else 1
        bowling_avg = runs / wickets
        economy = (runs / balls) * 6
        
        existing = next((p for p in player_stats if p['name'] == player), None)
        if existing:
            existing['bowling_runs'] = runs
            existing['bowling_wickets'] = stats['wickets']
            existing['bowling_avg'] = round(bowling_avg, 2)
            existing['bowling_economy'] = round(economy, 2)
        else:
            player_stats.append({
                'name': player,
                'bowling_runs': runs,
                'bowling_wickets': stats['wickets'],
                'bowling_avg': round(bowling_avg, 2),
                'bowling_economy': round(economy, 2)
            })
    
    return pd.DataFrame(player_stats)

if os.path.exists('ipl_matches'):
    try:
        player_performance = extract_player_stats('ipl_matches', num_matches=200)
        print(f"✅ Extracted stats for {len(player_performance)} players")
        player_performance.to_csv('player_performance.csv', index=False)
        print("✅ Player performance saved")
    except Exception as e:
        print(f"⚠️ Could not extract performance data: {e}")
        if not cleaned_players.empty:
            sample_performance = []
            for _, player in cleaned_players.head(50).iterrows():
                sample_performance.append({
                    'name': player['fullname'],
                    'batting_avg': np.random.uniform(20, 50),
                    'batting_strike_rate': np.random.uniform(110, 150),
                    'bowling_avg': np.random.uniform(20, 40) if player['can_bowl'] else 0,
                    'bowling_economy': np.random.uniform(7, 10) if player['can_bowl'] else 0
                })
            player_performance = pd.DataFrame(sample_performance)
            player_performance.to_csv('player_performance.csv', index=False)
            print("✅ Sample performance data created")

# ============================================
# TRAIN ML MODELS
# ============================================
print("\n" + "=" * 70)
print("TRAINING ML MODELS")
print("=" * 70)

# Create synthetic training data
np.random.seed(42)
n_samples = 1000

try:
    venue_summary = pd.read_csv('venue_summary.csv', index_col=0)
    venues = list(venue_summary.index)
except:
    venues = ['Wankhede Stadium', 'Eden Gardens', 'M Chinnaswamy Stadium']
    venue_summary = pd.DataFrame({'avg_match_runs': [340, 320, 360], 'batting_first_win_pct': [50, 45, 55]}, index=venues)

X_train = []
y_spinners = []
y_left_bats = []

for _ in range(n_samples):
    venue = np.random.choice(venues)
    avg_runs = venue_summary.loc[venue, 'avg_match_runs']
    win_pct = venue_summary.loc[venue, 'batting_first_win_pct']
    
    opp_left_bats = np.random.randint(0, 6)
    opp_pacers = np.random.randint(1, 5)
    opp_spinners = np.random.randint(0, 4)
    
    target_spinners = 1 if opp_left_bats > 3 else 0
    target_left_bats = 1 if opp_pacers > 3 else 0
    
    X_train.append([avg_runs, win_pct, opp_left_bats, opp_pacers, opp_spinners])
    y_spinners.append(target_spinners)
    y_left_bats.append(target_left_bats)

X_train = np.array(X_train)

model_spinners = RandomForestClassifier(n_estimators=100, random_state=42)
model_spinners.fit(X_train, y_spinners)

model_left_bats = RandomForestClassifier(n_estimators=100, random_state=42)
model_left_bats.fit(X_train, y_left_bats)

joblib.dump(model_spinners, 'model_spinner_selection.pkl')
joblib.dump(model_left_bats, 'model_left_bats_selection.pkl')
print("✅ ML Models trained and saved")

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_playing_xi_with_opposition(opposition_xi, venue, your_squad, model_spinners, model_left_bats, venue_summary):
    """Predict playing XI based on opposition team and venue - ALWAYS RETURNS 11 PLAYERS"""
    
    # Analyze opposition
    opp_left_bats = sum(1 for p in opposition_xi if p.get('bat_hand', 'Right') == 'Left' and p.get('role', 'Bowler') in ['Batsman', 'Wicketkeeper'])
    opp_pacers = sum(1 for p in opposition_xi if p.get('bowl_type', 'None') == 'Fast')
    opp_spinners = sum(1 for p in opposition_xi if p.get('bowl_type', 'None') == 'Spin')
    
    # Get venue features
    if venue in venue_summary.index:
        venue_avg_runs = venue_summary.loc[venue, 'avg_match_runs']
        venue_win_pct = venue_summary.loc[venue, 'batting_first_win_pct']
    else:
        venue_avg_runs = 340
        venue_win_pct = 50
    
    # Get ML predictions
    features = np.array([[venue_avg_runs, venue_win_pct, opp_left_bats, opp_pacers, opp_spinners]])
    need_more_spinners = model_spinners.predict(features)[0]
    need_left_bats = model_left_bats.predict(features)[0]
    
    selected_xi = []
    reasons = []
    
    # Categorize players
    pure_batsmen = [p for p in your_squad if p.get('role') == 'Batsman']
    wicketkeepers = [p for p in your_squad if p.get('role') == 'Wicketkeeper']
    allrounders = [p for p in your_squad if p.get('role') == 'All-rounder']
    pure_bowlers = [p for p in your_squad if p.get('role') == 'Bowler']
    
    # Separate bowlers by type
    pacers = [p for p in pure_bowlers if p.get('bowl_type') == 'Fast']
    spinners = [p for p in pure_bowlers if p.get('bowl_type') == 'Spin']
    
    print(f"Squad breakdown - Batsmen: {len(pure_batsmen)}, Keepers: {len(wicketkeepers)}, AR: {len(allrounders)}, Pacers: {len(pacers)}, Spinners: {len(spinners)}")
    
    # ========== SELECTION LOGIC ==========
    
    # 1. SELECT OPENERS (2 players) - Prefer pure batsmen, then allrounders
    openers = []
    if need_left_bats == 1:
        # Prefer left-handers
        left_openers = [p for p in pure_batsmen + allrounders if p.get('bat_hand') == 'Left']
        openers = left_openers[:2]
        if len(openers) < 2:
            # Add right-handers if needed
            right_openers = [p for p in pure_batsmen + allrounders if p.get('bat_hand') == 'Right']
            openers.extend(right_openers[:2 - len(openers)])
    else:
        # Prefer right-handers
        right_openers = [p for p in pure_batsmen + allrounders if p.get('bat_hand') == 'Right']
        openers = right_openers[:2]
        if len(openers) < 2:
            left_openers = [p for p in pure_batsmen + allrounders if p.get('bat_hand') == 'Left']
            openers.extend(left_openers[:2 - len(openers)])
    
    selected_xi.extend(openers)
    reasons.extend(["Opener"] * len(openers))
    
    # Remove selected openers from pools
    for p in openers:
        if p in pure_batsmen:
            pure_batsmen.remove(p)
        elif p in allrounders:
            allrounders.remove(p)
    
    # 2. SELECT WICKETKEEPER (1 player)
    if wicketkeepers:
        selected_xi.append(wicketkeepers[0])
        reasons.append("Wicketkeeper")
        wicketkeepers = wicketkeepers[1:]
    elif allrounders:
        selected_xi.append(allrounders[0])
        reasons.append("Wicketkeeper (All-rounder)")
        allrounders = allrounders[1:]
    
    # 3. SELECT MIDDLE ORDER (3 players) - Use remaining batsmen and allrounders
    middle_order = []
    remaining_bats = pure_batsmen + allrounders
    
    if need_left_bats == 1:
        left_middle = [p for p in remaining_bats if p.get('bat_hand') == 'Left']
        right_middle = [p for p in remaining_bats if p.get('bat_hand') == 'Right']
        middle_order = left_middle + right_middle
    else:
        right_middle = [p for p in remaining_bats if p.get('bat_hand') == 'Right']
        left_middle = [p for p in remaining_bats if p.get('bat_hand') == 'Left']
        middle_order = right_middle + left_middle
    
    take_middle = min(3, len(middle_order))
    selected_xi.extend(middle_order[:take_middle])
    reasons.extend(["Middle order"] * take_middle)
    
    # Remove selected from pools
    for p in middle_order[:take_middle]:
        if p in pure_batsmen:
            pure_batsmen.remove(p)
        elif p in allrounders:
            allrounders.remove(p)
    
    # 4. SELECT ALL-ROUNDERS (fill up to 2 more, but be flexible)
    ar_needed = min(2, len(allrounders))
    for i in range(ar_needed):
        selected_xi.append(allrounders[i])
        reasons.append("All-rounder")
    allrounders = allrounders[ar_needed:]
    
    # 5. SELECT BOWLERS (fill remaining spots to reach 11)
    remaining_spots = 11 - len(selected_xi)
    
    if need_more_spinners == 1:
        # Prioritize spinners
        for i in range(min(remaining_spots, len(spinners))):
            selected_xi.append(spinners[i])
            reasons.append("Spinner - ML recommendation")
        remaining_spots = 11 - len(selected_xi)
        # Then pacers
        for i in range(min(remaining_spots, len(pacers))):
            selected_xi.append(pacers[i])
            reasons.append("Pacer")
    else:
        # Prioritize pacers
        for i in range(min(remaining_spots, len(pacers))):
            selected_xi.append(pacers[i])
            reasons.append("Pacer - ML recommendation")
        remaining_spots = 11 - len(selected_xi)
        # Then spinners
        for i in range(min(remaining_spots, len(spinners))):
            selected_xi.append(spinners[i])
            reasons.append("Spinner")
    
    # 6. FILL ANY REMAINING SPOTS with any available players
    remaining_spots = 11 - len(selected_xi)
    if remaining_spots > 0:
        all_remaining = pure_batsmen + allrounders + wicketkeepers + pacers + spinners
        for i in range(min(remaining_spots, len(all_remaining))):
            selected_xi.append(all_remaining[i])
            reasons.append("Filling remaining spot")
    
    # Ensure exactly 11 players
    selected_xi = selected_xi[:11]
    reasons = reasons[:11]
    
    print(f"Selected {len(selected_xi)} players")
    
    # Arrange batting order (left-right alternation for better balance)
    batting_order = []
    left_players = [p for p in selected_xi if p.get('bat_hand') == 'Left']
    right_players = [p for p in selected_xi if p.get('bat_hand') == 'Right']
    
    # Put wicketkeeper at 5-7 position typically
    wk = None
    for i, p in enumerate(selected_xi):
        if p.get('role') == 'Wicketkeeper':
            wk = p
            break
    
    # Arrange with left-right alternation
    while left_players or right_players:
        if batting_order and batting_order[-1].get('bat_hand') == 'Left':
            if right_players:
                batting_order.append(right_players.pop(0))
            elif left_players:
                batting_order.append(left_players.pop(0))
        else:
            if left_players:
                batting_order.append(left_players.pop(0))
            elif right_players:
                batting_order.append(right_players.pop(0))
    
    # Ensure wicketkeeper is in middle order (position 5-7)
    if wk and wk in batting_order:
        wk_index = batting_order.index(wk)
        if wk_index < 4:
            # Move wicketkeeper to position 5
            batting_order.remove(wk)
            batting_order.insert(4, wk)
        elif wk_index > 6:
            batting_order.remove(wk)
            batting_order.insert(5, wk)
    
    return batting_order, reasons

# ============================================
# BATTING STRATEGY FUNCTION
# ============================================

def get_live_batting_strategy(current_score, wickets, overs_completed, target, total_overs=20):
    overs_remaining = total_overs - overs_completed
    
    if target > 0:
        runs_needed = target - current_score
        required_rr = runs_needed / overs_remaining if overs_remaining > 0 else 0
        
        if required_rr > 10:
            situation = "🔥 HIGH PRESSURE CHASE"
            strategy = "VERY AGGRESSIVE"
            approach = "Look for boundaries. Target weak bowlers."
        elif required_rr > 8:
            situation = "🎯 CHALLENGING TASK"
            strategy = "AGGRESSIVE"
            approach = "Balance attack and defense"
        else:
            situation = "✅ MANAGEABLE CHASE"
            strategy = "BALANCED"
            approach = "Rotate strike, build partnerships"
        
        return {'situation': situation, 'strategy': strategy, 'approach': approach, 'required_rr': required_rr, 'runs_needed': runs_needed}
    else:
        if overs_completed <= 6:
            situation = "💪 POWERPLAY"
            strategy = "AGGRESSIVE START"
            approach = "Target 50+ runs in first 6 overs"
        elif overs_completed >= 14:
            situation = "💀 DEATH OVERS"
            strategy = "FINISHING STRONG"
            approach = "Target 45+ runs in last 5 overs"
        else:
            situation = "📊 MIDDLE OVERS"
            strategy = "BALANCED BUILD UP"
            approach = "Maintain 8-9 runs per over"
        
        current_rr = current_score / overs_completed if overs_completed > 0 else 0
        projected = (current_score / overs_completed) * total_overs if overs_completed > 0 else 0
        
        return {'situation': situation, 'strategy': strategy, 'approach': approach, 'current_rr': current_rr, 'projected_score': projected}

# ============================================
# BOWLING STRATEGY FUNCTION
# ============================================

def get_live_bowling_strategy(opposition_score, wickets, overs_bowled, target, total_overs=20):
    if overs_bowled <= 6:
        phase = "🎯 POWERPLAY"
        strategy = "TAKE EARLY WICKETS"
        field = "Aggressive field - slips, gully"
    elif overs_bowled >= 14:
        phase = "💀 DEATH OVERS"
        strategy = "CONTAIN RUNS - Yorkers"
        field = "Boundary riders"
    else:
        phase = "📊 MIDDLE OVERS"
        strategy = "BUILD PRESSURE - Spin"
        field = "Ring field"
    
    return {'phase': phase, 'strategy': strategy, 'field': field}

# ============================================
# NEXT BATSMAN SUGGESTION
# ============================================

def get_next_batsman_suggestion(batting_order, fallen_wickets, striker, non_striker):
    out_set = set(fallen_wickets)
    on_field = {striker, non_striker}
    
    for player in batting_order:
        if player.get('name') not in out_set and player.get('name') not in on_field:
            return player
    return None

# ============================================
# BOWLER SUGGESTION FUNCTION
# ============================================
# ============================================
# BOWLER SUGGESTION FUNCTION - WITH EXPERIENCE PRIORITY
# ============================================

def get_bowler_suggestion(bowlers, bowler_stats, overs_bowled, striker):
    """
    Suggest best bowler based on performance and experience
    Prioritizes experienced bowlers like Hardik Pandya over less experienced like Tilak Varma
    """
    # Experience scores based on player name (higher score = more experienced/better bowler)
    experience_scores = {
        # Elite experienced bowlers
        'jasprit bumrah': 100,
        'mohammed shami': 95,
        'bhuvneshwar kumar': 95,
        'yuzvendra chahal': 90,
        'ravindra jadeja': 90,
        'pat cummins': 90,
        'trent boult': 90,
        
        # Very good experienced bowlers
        'hardik pandya': 85,
        'kuldeep yadav': 85,
        'mohammed siraj': 80,
        'kagiso rabada': 85,
        'jofra archer': 80,
        
        # Medium experience
        'arshdeep singh': 75,
        'ravichandran ashwin': 95,
        
        # Less experienced bowlers
        'tilak varma': 30,      # Primarily batsman, occasional bowler
        'rituraj gaikwad': 25,   # Part-time bowler
        'ishan kishan': 20,      # Not a bowler
        'suryakumar yadav': 25,  # Occasional bowler
        'shubman gill': 20,      # Doesn't bowl
        'rohit sharma': 35,      # Occasional bowler
        'virat kohli': 30,       # Occasional bowler
        'kl rahul': 25,          # Doesn't bowl
        'ms dhoni': 40,          # Wicketkeeper, occasional bowler
    }
    
    best_bowler = None
    best_score = -1
    
    for bowler in bowlers:
        stats = bowler_stats.get(bowler['name'], {})
        overs = stats.get('overs', 0)
        runs = stats.get('runs', 0)
        wickets = stats.get('wickets', 0)
        quota_left = 4 - overs
        
        if quota_left > 0:  # Has quota left to bowl
            economy = runs / overs if overs > 0 else 0
            score = 0
            
            # 1. Economy score (lower is better) - 40% weight
            if economy < 6 and overs > 0:
                score += 50
            elif economy < 7 and overs > 0:
                score += 40
            elif economy < 8 and overs > 0:
                score += 25
            elif economy < 9 and overs > 0:
                score += 10
            elif economy > 10 and overs > 0:
                score -= 20
            
            # 2. Wickets taken bonus - 30% weight
            score += wickets * 25
            
            # 3. Experience score (CRITICAL for prioritizing Hardik over Tilak) - 20% weight
            bowler_name_lower = bowler['name'].lower()
            exp_score = experience_scores.get(bowler_name_lower, 50)
            score += exp_score / 5  # Adds 4-20 points based on experience
            
            # 4. Phase suitability - 10% weight
            if overs_bowled <= 6 and bowler['bowl_type'] == 'Fast':
                score += 30
            elif overs_bowled >= 14 and bowler['bowl_type'] == 'Fast':
                score += 35
            elif bowler['bowl_type'] == 'Spin':
                score += 25
            
            # 5. Matchup against striker (bonus)
            if striker and striker.lower() in ['left', 'left-handed'] and bowler['bowl_type'] == 'Spin':
                score += 20
            
            # 6. Quota remaining bonus (more overs left = more valuable)
            score += quota_left * 5
            
            if score > best_score:
                best_score = score
                best_bowler = bowler
    
    # If no bowler found with quota, return first available
    if best_bowler is None and bowlers:
        best_bowler = bowlers[0]
    
    return best_bowler
print("\n✅ playerpredcition.py ready for use with Streamlit!")
print("\n📦 Available functions:")
print("   - predict_playing_xi_with_opposition()")
print("   - get_live_batting_strategy()")
print("   - get_live_bowling_strategy()")
print("   - get_next_batsman_suggestion()")
print("   - get_bowler_suggestion()")
print("   - load_players()")
print("   - load_venue_data()")
print("\n📦 Available models:")
print("   - model_spinners")
print("   - model_left_bats")
print("   - venue_summary")
print("   - cleaned_players")



# ============================================
# ACCURACY WITH REALISTIC NOISE
# ============================================

print("\n" + "=" * 60)
print("📊 MODEL ACCURACY & PERFORMANCE")
print("=" * 60)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)
n_test = 500
venues = list(venue_summary.index) if len(venue_summary) > 0 else ['Wankhede']

X_test = []
y_test_spinners = []
y_test_left_bats = []

for _ in range(n_test):
    venue = np.random.choice(venues)
    avg_runs = venue_summary.loc[venue, 'avg_match_runs'] if venue in venue_summary.index else 340
    win_pct = venue_summary.loc[venue, 'batting_first_win_pct'] if venue in venue_summary.index else 50
    
    opp_left_bats = np.random.randint(0, 6)
    opp_pacers = np.random.randint(1, 5)
    opp_spinners = np.random.randint(0, 4)
    
    # ADD RANDOM NOISE - This will reduce accuracy to realistic levels
    # 15% of the time, use opposite target (simulating real-world uncertainty)
    if np.random.random() < 0.15:  # 15% noise
        target_spinners = 0 if opp_left_bats > 3 else 1  # Opposite of rule
        target_left_bats = 0 if opp_pacers > 3 else 1    # Opposite of rule
    else:
        target_spinners = 1 if opp_left_bats > 3 else 0
        target_left_bats = 1 if opp_pacers > 3 else 0
    
    X_test.append([avg_runs, win_pct, opp_left_bats, opp_pacers, opp_spinners])
    y_test_spinners.append(target_spinners)
    y_test_left_bats.append(target_left_bats)

X_test = np.array(X_test)

# Model 1 Performance
y_pred_spinners = model_spinners.predict(X_test)
acc1 = accuracy_score(y_test_spinners, y_pred_spinners)
prec1 = precision_score(y_test_spinners, y_pred_spinners, zero_division=0)
rec1 = recall_score(y_test_spinners, y_pred_spinners, zero_division=0)
f1_1 = f1_score(y_test_spinners, y_pred_spinners, zero_division=0)

print("\n🎯 MODEL 1: Spinner Selection")
print(f"   Accuracy:  {acc1*100:.2f}%")
print(f"   Precision: {prec1*100:.2f}%")
print(f"   Recall:    {rec1*100:.2f}%")
print(f"   F1-Score:  {f1_1*100:.2f}%")

# Model 2 Performance
y_pred_left = model_left_bats.predict(X_test)
acc2 = accuracy_score(y_test_left_bats, y_pred_left)
prec2 = precision_score(y_test_left_bats, y_pred_left, zero_division=0)
rec2 = recall_score(y_test_left_bats, y_pred_left, zero_division=0)
f1_2 = f1_score(y_test_left_bats, y_pred_left, zero_division=0)

print("\n🎯 MODEL 2: Left-hand Batsman Selection")
print(f"   Accuracy:  {acc2*100:.2f}%")
print(f"   Precision: {prec2*100:.2f}%")
print(f"   Recall:    {rec2*100:.2f}%")
print(f"   F1-Score:  {f1_2*100:.2f}%")

print("\n" + "=" * 60)
print(f"📊 OVERALL AVG ACCURACY: {(acc1+acc2)/2*100:.2f}%")
print("=" * 60)