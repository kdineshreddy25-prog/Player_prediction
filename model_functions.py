# app.py - Cricket XI Predictor Pro
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os

# ============================================
# IMPORT YOUR TRAINED MODELS FROM model_functions.py
# ============================================

try:
    from model_functions import (
        predict_playing_xi_with_opposition,
        get_live_batting_strategy,
        get_live_bowling_strategy,
        get_next_batsman_suggestion,
        get_bowler_suggestion,
        model_spinners,
        model_left_bats,
        venue_summary,
        cleaned_players
    )
    MODELS_LOADED = True
    st.success("✅ Successfully loaded your trained ML models!")
except ImportError as e:
    st.error(f"❌ Cannot import from model_functions.py: {e}")
    st.info("Make sure model_functions.py is in the same directory")
    MODELS_LOADED = False
    st.stop()

st.set_page_config(page_title="Cricket XI Predictor Pro", page_icon="🏏", layout="wide")

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f0f2f5 0%, #e0e5ec 100%); }
    
    .premium-header {
        background: linear-gradient(135deg, #1a472a 0%, #2e8b57 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .premium-header h1 { color: white !important; margin: 0; }
    .premium-header p { color: white !important; }
    
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }
    .section-header h3 { color: white !important; margin: 0; }
    
    .player-card {
        background: white;
        border-radius: 10px;
        padding: 0.6rem;
        margin: 0.4rem 0;
        border-left: 4px solid #2e8b57;
    }
    .player-name { font-weight: 700; color: #000000 !important; }
    .player-card span { color: #000000 !important; }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
    }
    .stat-value { font-size: 1.5rem; font-weight: bold; color: #2e8b57 !important; }
    .stat-label { color: #000000 !important; }
    
    .prediction-box {
        background: white;
        border-radius: 10px;
        padding: 0.6rem;
        margin: 0.3rem 0;
        border-left: 4px solid #2e8b57;
    }
    .prediction-number { font-size: 1.3rem; font-weight: bold; color: #2e8b57 !important; }
    .prediction-box div, .prediction-box span { color: #000000 !important; }
    
    .live-strategy-box {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .live-strategy-box h4, .live-strategy-box p, .live-strategy-box strong { color: #000000 !important; }
    
    .wicket-card {
        background: #ffebee;
        border-radius: 8px;
        padding: 0.4rem;
        margin: 0.2rem 0;
        border-left: 3px solid #f44336;
    }
    .wicket-card b, .wicket-card span { color: #000000 !important; }
    
    .bowler-card {
        background: #fff3e0;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #ff9800;
    }
    .bowler-card b, .bowler-card span { color: #000000 !important; }
    
    .suggestion-big {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1a472a !important;
        text-align: center;
        padding: 0.6rem;
        background: #e8f5e9;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #2e8b57;
    }
    
    .role-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        color: white !important;
    }
    .role-batsman { background: #2196f3; }
    .role-bowler { background: #f44336; }
    .role-allrounder { background: #ff9800; }
    .role-wicketkeeper { background: #9c27b0; }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a472a, #2e8b57);
        color: white !important;
        border: none;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #333 !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        color: #2e8b57 !important;
        font-weight: bold !important;
    }
    
    .stMetric label, .stMetric div { color: #000000 !important; }
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        color: #000000 !important;
        background: white !important;
    }
    label, .stMarkdown, .stCaption { color: #000000 !important; }
    
    .css-1d391kg, .css-12oz5g7 { background: linear-gradient(180deg, #1a472a 0%, #0d2818 100%); }
    .sidebar .stMarkdown, .sidebar .stSelectbox label { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD PLAYER DATA
# ============================================

@st.cache_data
def load_players_json():
    try:
        with open('players.json', 'r') as f:
            players_array = json.load(f)
        players_dict = {}
        for player in players_array:
            if player.get('fullname'):
                players_dict[player['fullname']] = {
                    'role': player.get('role', 'Batsman'),
                    'bat_hand': player.get('bat_hand', 'Right'),
                    'bowl_type': player.get('bowl_type', 'None'),
                    'country': player.get('country_name', '')
                }
        return players_dict
    except:
        return {}

# ============================================
# SESSION STATE
# ============================================

if 'squad' not in st.session_state:
    st.session_state.squad = []
if 'opposition' not in st.session_state:
    st.session_state.opposition = []
if 'squad_size' not in st.session_state:
    st.session_state.squad_size = 15
if 'squad_set' not in st.session_state:
    st.session_state.squad_set = False
if 'predicted_xi' not in st.session_state:
    st.session_state.predicted_xi = None
if 'ml_reasons' not in st.session_state:
    st.session_state.ml_reasons = []

# ============================================
# MAIN APP
# ============================================

def main():
    st.markdown("""
    <div class="premium-header">
        <h1>🏏 CRICKET XI PREDICTOR PRO</h1>
        <p>Powered by Your Trained ML Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    players_db = load_players_json()
    
    if not players_db:
        st.error("❌ players.json not found!")
        return
    
    with st.sidebar:
        st.markdown("### ⚙️ Match Settings")
        
        if venue_summary is not None and len(venue_summary) > 0:
            venue = st.selectbox("📍 Venue", venue_summary.index.tolist())
        else:
            venue = st.text_input("📍 Venue", "Wankhede Stadium, Mumbai")
        
        st.markdown("---")
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.squad)}/{st.session_state.squad_size}</div>
            <div class="stat-label">Squad Size</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.opposition)}/11</div>
            <div class="stat-label">Opposition</div>
        </div>
        """, unsafe_allow_html=True)
        
        if MODELS_LOADED:
            st.success("✅ Your ML Models Active")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Team Builder", "🏏 Batting Strategy", "🎯 Bowling Strategy"])
    
    # ============================================
    # TAB 1: TEAM BUILDER
    # ============================================
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header"><h3>🏏 BUILD YOUR SQUAD</h3></div>', unsafe_allow_html=True)
            
            if not st.session_state.squad_set:
                size = st.selectbox("Squad Size (11-20)", list(range(11, 21)), index=4)
                if st.button("Start Building"):
                    st.session_state.squad_size = size
                    st.session_state.squad_set = True
                    st.rerun()
            
            if st.session_state.squad_set and len(st.session_state.squad) < st.session_state.squad_size:
                remaining = st.session_state.squad_size - len(st.session_state.squad)
                st.progress(len(st.session_state.squad) / st.session_state.squad_size)
                st.caption(f"Add {remaining} more players")
                
                search = st.text_input("🔍 Search Player")
                if search:
                    matches = [p for p in players_db.keys() if search.lower() in p.lower()][:10]
                    for player in matches:
                        if player not in [p['name'] for p in st.session_state.squad]:
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"{player} - {players_db[player]['role']} | {players_db[player]['bat_hand']}")
                            with col_b:
                                if st.button("Add", key=f"add_{player}"):
                                    st.session_state.squad.append({'name': player, **players_db[player]})
                                    st.rerun()
            
            if st.session_state.squad:
                for p in st.session_state.squad:
                    role_class = f"role-{p['role'].lower()}"
                    st.markdown(f"""
                    <div class="player-card">
                        <span class="player-name">✅ {p['name']}</span>
                        <span class="role-badge {role_class}">{p['role']}</span>
                        <span> | {p['bat_hand']} bat | {p['bowl_type']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                if st.button("Clear Squad"):
                    st.session_state.squad = []
                    st.session_state.squad_set = False
                    st.rerun()
        
        with col2:
            st.markdown('<div class="section-header"><h3>🎯 OPPOSITION XI</h3></div>', unsafe_allow_html=True)
            st.caption(f"Add 11 players ({len(st.session_state.opposition)}/11)")
            st.progress(len(st.session_state.opposition) / 11)
            
            if len(st.session_state.opposition) < 11:
                search = st.text_input("🔍 Search Opposition", key="opp_search")
                if search:
                    matches = [p for p in players_db.keys() if search.lower() in p.lower()][:10]
                    for player in matches:
                        if player not in [p['name'] for p in st.session_state.opposition]:
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"{player} - {players_db[player]['role']}")
                            with col_b:
                                if st.button("Add", key=f"opp_{player}"):
                                    st.session_state.opposition.append({'name': player, **players_db[player]})
                                    st.rerun()
            
            for i, p in enumerate(st.session_state.opposition, 1):
                st.write(f"{i}. {p['name']} ({p['role']} | {p['bat_hand']})")
            
            if len(st.session_state.opposition) == 11:
                st.success("✅ Opposition complete!")
            if st.button("Clear Opposition"):
                st.session_state.opposition = []
                st.rerun()
        
        # ============================================
        # CALL YOUR TRAINED MODEL FOR PREDICTION
        # ============================================
        st.markdown("---")
        if st.button("🚀 GENERATE PLAYING XI (USING YOUR ML MODEL)", use_container_width=True):
            if len(st.session_state.squad) < 11:
                st.error(f"Need {11 - len(st.session_state.squad)} more players")
            elif len(st.session_state.opposition) < 11:
                st.error(f"Need {11 - len(st.session_state.opposition)} more opposition")
            else:
                with st.spinner("🧠 Your ML Model is analyzing..."):
                    time.sleep(1)
                    
                    # CALL YOUR TRAINED MODEL
                    predicted_xi, reasons = predict_playing_xi_with_opposition(
                        opposition_xi=st.session_state.opposition,
                        venue=venue,
                        your_squad=st.session_state.squad,
                        model_spinners=model_spinners,
                        model_left_bats=model_left_bats,
                        venue_summary=venue_summary
                    )
                    
                    if predicted_xi and len(predicted_xi) == 11:
                        st.session_state.predicted_xi = predicted_xi
                        st.session_state.ml_reasons = reasons
                        st.success("✅ Your ML Model has selected the playing XI!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"Model returned {len(predicted_xi) if predicted_xi else 0} players")
        
        # DISPLAY RESULTS
        if st.session_state.predicted_xi and len(st.session_state.predicted_xi) == 11:
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>🏆 YOUR ML MODEL'S PLAYING XI</h3>", unsafe_allow_html=True)
            
            if st.session_state.ml_reasons:
                with st.expander("🤖 ML Model Decision Insights"):
                    for reason in st.session_state.ml_reasons[:5]:
                        st.info(reason)
            
            for i, p in enumerate(st.session_state.predicted_xi, 1):
                role_class = f"role-{p['role'].lower()}"
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span class="prediction-number">#{i}</span>
                        <span style="font-weight: bold;">{p['name']}</span>
                        <span class="role-badge {role_class}">{p['role']}</span>
                        <span>| {p['bat_hand']} bat | {p['bowl_type']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            left_count = len([p for p in st.session_state.predicted_xi if p.get('bat_hand') == 'Left'])
            pacer_count = len([p for p in st.session_state.predicted_xi if p.get('bowl_type') == 'Fast'])
            spinner_count = len([p for p in st.session_state.predicted_xi if p.get('bowl_type') == 'Spin'])
            
            with col1:
                st.metric("🔄 Left-hand Bats", left_count)
            with col2:
                st.metric("➡️ Right-hand Bats", 11 - left_count)
            with col3:
                st.metric("⚡ Fast Bowlers", pacer_count)
            with col4:
                st.metric("🔄 Spin Bowlers", spinner_count)
    
    # ============================================
    # TAB 2: BATTING STRATEGY - USING YOUR MODEL
    # ============================================
    with tab2:
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🏏 LIVE BATTING STRATEGY</h3>", unsafe_allow_html=True)
        
        if st.session_state.predicted_xi:
            innings = st.radio("Innings", ["1st Innings (Batting First)", "2nd Innings (Chasing)"], horizontal=True)
            
            col1, col2 = st.columns(2)
            with col1:
                score = st.number_input("Current Score", value=100)
                overs = st.number_input("Overs Completed", value=10.0)
            with col2:
                wickets = st.number_input("Wickets Fallen", value=3)
                target = st.number_input("Target", value=180) if "Chasing" in innings else 0
            
            col1, col2 = st.columns(2)
            with col1:
                striker = st.text_input("Striker (on strike)")
            with col2:
                non_striker = st.text_input("Non-Striker")
            
            # Fallen wickets tracking
            st.markdown("### 📋 Fallen Wickets")
            fallen_wickets = []
            if wickets > 0:
                cols = st.columns(3)
                for i in range(wickets):
                    with cols[i % 3]:
                        name = st.text_input(f"Wicket {i+1}", key=f"fall_{i}")
                        if name:
                            fallen_wickets.append(name)
                            st.markdown(f'<div class="wicket-card"><b>OUT:</b> {name}</div>', unsafe_allow_html=True)
            
            if st.button("🎯 GET BATTING STRATEGY", use_container_width=True):
                with st.spinner("Your ML Model analyzing batting situation..."):
                    time.sleep(0.5)
                    
                    # CALL YOUR BATTING STRATEGY FUNCTION
                    strategy = get_live_batting_strategy(score, wickets, overs, target)
                    
                    st.markdown(f"""
                    <div class="live-strategy-box">
                        <h4>{strategy['situation']}</h4>
                        <p><strong>Strategy:</strong> {strategy['strategy']}</p>
                        <p><strong>Approach:</strong> {strategy['approach']}</p>
                        <p><strong>Risk Level:</strong> {strategy.get('risk', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get next batsman suggestion
                    if striker and non_striker and wickets < 10:
                        next_batsman = get_next_batsman_suggestion(
                            st.session_state.predicted_xi, 
                            fallen_wickets, 
                            striker, 
                            non_striker
                        )
                        if next_batsman:
                            st.markdown(f"""
                            <div class="suggestion-big">
                                🎯 NEXT BATSMAN: {next_batsman['name']} ({next_batsman['role']} | {next_batsman['bat_hand']} hand)
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please generate Playing XI first using your ML model")
    
    # ============================================
    # TAB 3: BOWLING STRATEGY - USING YOUR MODEL
    # ============================================
    with tab3:
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>🎯 LIVE BOWLING STRATEGY</h3>", unsafe_allow_html=True)
        
        if st.session_state.predicted_xi:
            innings = st.radio("Innings", ["1st Innings (Bowling First)", "2nd Innings (Defending)"], horizontal=True)
            
            col1, col2 = st.columns(2)
            with col1:
                opp_score = st.number_input("Opposition Score", value=80)
                overs_bowled = st.number_input("Overs Bowled", value=8.0)
            with col2:
                wkts = st.number_input("Wickets Taken", value=2)
                target_defend = st.number_input("Target to Defend", value=160) if "Defending" in innings else 0
            
            col1, col2 = st.columns(2)
            with col1:
                opp_striker = st.text_input("Opposition Striker")
            with col2:
                opp_non_striker = st.text_input("Opposition Non-Striker")
            
            # Bowler Performance Tracking
            st.markdown("### 📊 Bowler Performance Tracking")
            bowlers = [p for p in st.session_state.predicted_xi if p['role'] in ['Bowler', 'All-rounder']]
            bowler_stats = {}
            
            for b in bowlers:
                with st.expander(f"🎯 {b['name']} ({b['bowl_type']})"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        overs_b = st.number_input("Overs", key=f"overs_{b['name']}", value=0.0, step=0.5)
                    with col_b:
                        runs_b = st.number_input("Runs", key=f"runs_{b['name']}", value=0)
                    with col_c:
                        wkts_b = st.number_input("Wickets", key=f"wkts_{b['name']}", value=0)
                    
                    if overs_b > 0:
                        economy = runs_b / overs_b
                        st.caption(f"Economy: {economy:.2f} | Quota Left: {4 - overs_b:.1f}")
                    
                    bowler_stats[b['name']] = {'overs': overs_b, 'runs': runs_b, 'wickets': wkts_b}
            
            # Opposition Fallen Wickets
            st.markdown("### 📋 Opposition Fallen Wickets")
            opp_fallen = []
            if wkts > 0:
                cols = st.columns(3)
                for i in range(wkts):
                    with cols[i % 3]:
                        name = st.text_input(f"Out Batsman {i+1}", key=f"bowl_fall_{i}")
                        if name:
                            opp_fallen.append(name)
                            st.markdown(f'<div class="wicket-card"><b>OUT:</b> {name}</div>', unsafe_allow_html=True)
            
            if st.button("🎯 GET BOWLING STRATEGY", use_container_width=True):
                with st.spinner("Your ML Model analyzing bowling situation..."):
                    time.sleep(0.5)
                    
                    # CALL YOUR BOWLING STRATEGY FUNCTION
                    strategy = get_live_bowling_strategy(opp_score, wkts, overs_bowled, target_defend)
                    
                    st.markdown(f"""
                    <div class="live-strategy-box">
                        <h4>{strategy['situation']}</h4>
                        <p><strong>Strategy:</strong> {strategy['strategy']}</p>
                        <p><strong>Field Setting:</strong> {strategy.get('field_setting', 'N/A')}</p>
                        <p><strong>{strategy.get('wicket_advice', '')}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get bowler suggestion
                    if opp_striker:
                        best_bowler = get_bowler_suggestion(bowlers, bowler_stats, overs_bowled, opp_striker)
                        if best_bowler:
                            st.markdown(f"""
                            <div class="suggestion-big">
                                🎯 RECOMMENDED BOWLER: {best_bowler['name']} ({best_bowler['bowl_type']})
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Bowler performance summary
                    st.markdown("### 📊 Bowler Performance Summary")
                    for b in bowlers:
                        stats = bowler_stats.get(b['name'], {})
                        overs_b = stats.get('overs', 0)
                        runs_b = stats.get('runs', 0)
                        wkts_b = stats.get('wickets', 0)
                        economy = runs_b / overs_b if overs_b > 0 else 0
                        quota_left = 4 - overs_b
                        
                        st.markdown(f"""
                        <div class="bowler-card">
                            <b>{b['name']}</b> ({b['bowl_type']})<br>
                            Overs: {overs_b} | Runs: {runs_b} | Wickets: {wkts_b} | Economy: {economy:.2f} | Quota Left: {quota_left:.1f}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please generate Playing XI first using your ML model")

if __name__ == "__main__":
    main()