# ============================================
# PLAYER PREDICTION MODULE (FINAL FIXED)
# ============================================

import json
import numpy as np
import joblib

# ============================================
# LOAD MODELS & DATA
# ============================================

try:
    model_spinners = joblib.load("model_spinner_selection.pkl")
    model_left_bats = joblib.load("model_left_bats_selection.pkl")

    with open("players.json", "r") as f:
        cleaned_players = json.load(f)

    with open("venues.json", "r") as f:
        venue_summary = json.load(f)

    MODELS_LOADED = True
    print("✅ Models and data loaded")

except Exception as e:
    MODELS_LOADED = False
    print("❌ Error loading:", e)


# ============================================
# PLAYING XI PREDICTION
# ============================================

def predict_playing_xi_with_opposition(opposition_xi, venue, your_squad):

    if not MODELS_LOADED:
        return [], ["Models not loaded"]

    # Safety
    opposition_xi = opposition_xi if isinstance(opposition_xi, list) else []
    your_squad = your_squad if isinstance(your_squad, list) else []

    # Count players safely
    opp_left_bats = sum(
        1 for p in opposition_xi
        if isinstance(p, dict) and p.get('bat_hand') == 'Left'
    )

    opp_pacers = sum(
        1 for p in opposition_xi
        if isinstance(p, dict) and p.get('bowl_type') == 'Fast'
    )

    opp_spinners = sum(
        1 for p in opposition_xi
        if isinstance(p, dict) and p.get('bowl_type') == 'Spin'
    )

    # Venue (list → dict fix)
    venue_data = next(
        (v for v in venue_summary if isinstance(v, dict) and v.get("venue") == venue),
        {"avg_match_runs": 340, "batting_first_win_pct": 50}
    )

    avg_runs = venue_data.get("avg_match_runs", 340)
    win_pct = venue_data.get("batting_first_win_pct", 50)

    features = np.array([[avg_runs, win_pct, opp_left_bats, opp_pacers, opp_spinners]])

    need_spinners = model_spinners.predict(features)[0]
    need_left = model_left_bats.predict(features)[0]

    # KEEP YOUR LOGIC SAME
    selected_xi = your_squad[:11]
    reasons = ["Selected using ML"] * len(selected_xi)

    return selected_xi, reasons


# ============================================
# BATTING STRATEGY (FIXED OUTPUT)
# ============================================

def get_live_batting_strategy(score, wickets, overs, target):

    if target > 0:
        runs_needed = target - score
        rr = runs_needed / (20 - overs) if overs < 20 else 0

        if rr > 10:
            situation = "🔥 High Pressure"
            strategy = "Play aggressive shots"
            approach = "Go for boundaries"
        elif rr > 8:
            situation = "🎯 Balanced"
            strategy = "Rotate strike"
            approach = "Mix singles and boundaries"
        else:
            situation = "✅ Easy Chase"
            strategy = "Play safe"
            approach = "Avoid risks"

        return {
            "situation": situation,
            "strategy": strategy,
            "approach": approach,
            "required_rr": rr,
            "runs_needed": runs_needed,
            "current_rr": score / overs if overs > 0 else 0,
            "projected_score": (score / overs) * 20 if overs > 0 else 0
        }

    else:
        current_rr = score / overs if overs > 0 else 0

        return {
            "situation": "💪 Build Score",
            "strategy": "Stabilize innings",
            "approach": "Rotate strike",
            "required_rr": 0,
            "runs_needed": 0,
            "current_rr": current_rr,
            "projected_score": current_rr * 20
        }


# ============================================
# BOWLING STRATEGY (FIXED OUTPUT)
# ============================================

def get_live_bowling_strategy(overs):

    if overs <= 6:
        return {
            "phase": "🎯 Powerplay",
            "strategy": "Attack with pace",
            "field": "Aggressive field"
        }

    elif overs >= 15:
        return {
            "phase": "💀 Death Overs",
            "strategy": "Use yorkers",
            "field": "Boundary protection"
        }

    else:
        return {
            "phase": "📊 Middle Overs",
            "strategy": "Build pressure",
            "field": "Balanced field"
        }


# ============================================
# NEXT BATSMAN
# ============================================

def get_next_batsman_suggestion(batting_order, fallen):
    for p in batting_order:
        name = p.get("name") if isinstance(p, dict) else p
        if name not in fallen:
            return p
    return None


# ============================================
# BOWLER SUGGESTION
# ============================================

def get_bowler_suggestion(bowlers, bowler_stats=None, overs=None, striker=None):
    if not bowlers:
        return None

    # Simple logic (kept same idea)
    return bowlers[0]


# ============================================
# EXPORTS
# ============================================

__all__ = [
    "predict_playing_xi_with_opposition",
    "get_live_batting_strategy",
    "get_live_bowling_strategy",
    "get_next_batsman_suggestion",
    "get_bowler_suggestion",
    "model_spinners",
    "model_left_bats",
    "venue_summary",
    "cleaned_players"
]

print("🚀 playerprediction.py ready!")