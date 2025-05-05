import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report

# Streamlit config
st.set_page_config(page_title="O7 Over 2.5 Predictor", layout="centered")

if "screen" not in st.session_state:
    st.session_state.screen = "home"

# === Utility ===
def init_stats():
    return {
        "partite": 0, "vittorie": 0, "pareggi": 0, "sconfitte": 0,
        "gol_fatti": 0, "gol_subiti": 0, "over35": 0,
        "gol_fatti_casa": 0, "gol_subiti_casa": 0, "over35_casa": 0, "partite_casa": 0,
        "gol_fatti_trasferta": 0, "gol_subiti_trasferta": 0, "over35_trasferta": 0, "partite_trasferta": 0
    }

@st.cache_data
def load_matches():
    urls = [
        "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.1.json",
        "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/en.1.json",
        "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/en.1.json"
    ]
    matches = []
    for url in urls:
        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
            league = url.split('/')[-1].split('.')[0]
            for match in data['matches']:
                match['league'] = league
            matches.extend(data['matches'])
        except:
            continue
    return matches

def build_rankings(matches):
    table = defaultdict(lambda: defaultdict(lambda: {"pt": 0, "gf": 0, "gs": 0}))
    for match in matches:
        if 'score' not in match or not match['score'].get('ft'):
            continue
        l = match['league']
        t1, t2 = match['team1'], match['team2']
        g1, g2 = match['score']['ft']
        if g1 > g2:
            table[l][t1]['pt'] += 3
        elif g1 < g2:
            table[l][t2]['pt'] += 3
        else:
            table[l][t1]['pt'] += 1
            table[l][t2]['pt'] += 1
        table[l][t1]['gf'] += g1
        table[l][t1]['gs'] += g2
        table[l][t2]['gf'] += g2
        table[l][t2]['gs'] += g1

    rankings = {}
    for league, teams in table.items():
        sorted_teams = sorted(teams.items(), key=lambda x: (-x[1]['pt'], -(x[1]['gf'] - x[1]['gs'])))
        rankings[league] = {team: rank+1 for rank, (team, _) in enumerate(sorted_teams)}
    return rankings

def get_recent_stats(matches, team, n=5):
    stats = []
    for match in matches:
        if 'score' not in match or not match['score'].get('ft'):
            continue
        if team not in (match['team1'], match['team2']):
            continue
        try:
            d = datetime.strptime(match['date'], "%Y-%m-%d")
        except:
            continue
        g1, g2 = match['score']['ft']
        scored = g1 if match['team1'] == team else g2
        conceded = g2 if match['team1'] == team else g1
        stats.append((d, scored, conceded))
    stats.sort(reverse=True)
    return stats[:n]

def get_btts_ratio(stats):
    return sum(1 for s in stats if s[1] > 0 and s[2] > 0) / len(stats) if stats else 0

def extract_features(matches, rankings):
    data = []
    for match in matches:
        if 'score' not in match or not match['score'].get('ft'):
            continue
        try:
            t1, t2 = match['team1'], match['team2']
            l = match['league']
            g1, g2 = match['score']['ft']
            total_goals = g1 + g2
            casa, ospite = init_stats(), init_stats()
            for m in matches:
                if m == match or 'score' not in m or not m['score'].get('ft'):
                    continue
                team1, team2 = m['team1'], m['team2']
                s1, s2 = m['score']['ft']
                tot = s1 + s2
                if t1 in (team1, team2):
                    casa['partite'] += 1
                    casa['gol_fatti'] += s1 if team1 == t1 else s2
                    casa['gol_subiti'] += s2 if team1 == t1 else s1
                    casa['over35'] += int(tot > 3)
                if t2 in (team1, team2):
                    ospite['partite'] += 1
                    ospite['gol_fatti'] += s2 if team2 == t2 else s1
                    ospite['gol_subiti'] += s1 if team2 == t2 else s2
                    ospite['over35'] += int(tot > 3)

            if casa['partite'] < 3 or ospite['partite'] < 3:
                continue

            media_gol_casa = (casa['gol_fatti'] + casa['gol_subiti']) / casa['partite']
            media_gol_ospite = (ospite['gol_fatti'] + ospite['gol_subiti']) / ospite['partite']
            btts_t1 = get_btts_ratio(get_recent_stats(matches, t1, 10))
            btts_t2 = get_btts_ratio(get_recent_stats(matches, t2, 10))
            form_t1 = get_recent_stats(matches, t1, 5)
            form_t2 = get_recent_stats(matches, t2, 5)

            data.append({
                "media_gol_casa": media_gol_casa,
                "media_gol_ospite": media_gol_ospite,
                "over35_casa_totale": casa['over35'] / casa['partite'],
                "over35_ospite_totale": ospite['over35'] / ospite['partite'],
                "avg_goal_match": ((media_gol_casa + media_gol_ospite) / 2) / casa['partite'],
                "form_gol_casa": sum(g for _, g, _ in form_t1) / 5,
                "form_gol_ospite": sum(g for _, g, _ in form_t2) / 5,
                "btts_ratio_casa": btts_t1,
                "btts_ratio_ospite": btts_t2,
                "ranking_casa": rankings[l].get(t1, 20),
                "ranking_ospite": rankings[l].get(t2, 20),
                "label": int(total_goals > 2.5)
            })
        except:
            continue
    return pd.DataFrame(data)

def train_model(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    search = RandomizedSearchCV(xgb, param_grid, n_iter=10, cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_

def predict_matches(model, matches, rankings, date):
    results = []
    for match in matches:
        if match.get('date') != date or 'score' in match:
            continue
        try:
            t1, t2 = match['team1'], match['team2']
            l = match['league']
            stats_t1 = get_recent_stats(matches, t1, 5)
            stats_t2 = get_recent_stats(matches, t2, 5)
            if len(stats_t1) < 3 or len(stats_t2) < 3:
                continue
            btts_t1 = get_btts_ratio(get_recent_stats(matches, t1, 10))
            btts_t2 = get_btts_ratio(get_recent_stats(matches, t2, 10))
            casa, ospite = init_stats(), init_stats()
            for m in matches:
                if 'score' not in m or not m['score'].get('ft'):
                    continue
                team1, team2 = m['team1'], m['team2']
                s1, s2 = m['score']['ft']
                tot = s1 + s2
                if t1 in (team1, team2):
                    casa['partite'] += 1
                    casa['gol_fatti'] += s1 if team1 == t1 else s2
                    casa['gol_subiti'] += s2 if team1 == t1 else s1
                    casa['over35'] += int(tot > 3)
                if t2 in (team1, team2):
                    ospite['partite'] += 1
                    ospite['gol_fatti'] += s2 if team2 == t2 else s1
                    ospite['gol_subiti'] += s1 if team2 == t2 else s2
                    ospite['over35'] += int(tot > 3)
            if casa['partite'] < 3 or ospite['partite'] < 3:
                continue
            media_gol_casa = (casa['gol_fatti'] + casa['gol_subiti']) / casa['partite']
            media_gol_ospite = (ospite['gol_fatti'] + ospite['gol_subiti']) / ospite['partite']
            row = pd.DataFrame([{
                "media_gol_casa": media_gol_casa,
                "media_gol_ospite": media_gol_ospite,
                "over35_casa_totale": casa['over35'] / casa['partite'],
                "over35_ospite_totale": ospite['over35'] / ospite['partite'],
                "avg_goal_match": ((media_gol_casa + media_gol_ospite) / 2) / casa['partite'],
                "form_gol_casa": sum(g for _, g, _ in stats_t1) / 5,
                "form_gol_ospite": sum(g for _, g, _ in stats_t2) / 5,
                "btts_ratio_casa": btts_t1,
                "btts_ratio_ospite": btts_t2,
                "ranking_casa": rankings[l].get(t1, 20),
                "ranking_ospite": rankings[l].get(t2, 20),
            }])
            prob = model.predict_proba(row)[0][1]
            if prob >= 0.75:
                results.append(f"{t1} vs {t2} → OVER 2.5 (Prob: {prob:.2f})")
        except:
            continue
    return results

# === UI ===
if st.session_state.screen == "home":
    st.markdown("<h1 style='text-align: center; color: #007BFF;'>WELCOME TO O7</h1>", unsafe_allow_html=True)
    date_input = st.text_input("Input a match date (YYYY-MM-DD)", key="date_input")
    if st.button("START") and date_input:
        st.session_state.date_chosen = date_input.strip()
        st.session_state.screen = "loading"
        st.experimental_rerun()

elif st.session_state.screen == "loading":
    st.markdown("<h2 style='text-align: center;'>OVER 2.5...</h2>", unsafe_allow_html=True)
    with st.spinner("Training model and generating predictions..."):
        matches = load_matches()
        rankings = build_rankings(matches)
        df = extract_features(matches, rankings)
        model = train_model(df)
        st.session_state.preds = predict_matches(model, matches, rankings, st.session_state.date_chosen)
        st.session_state.screen = "results"
        st.experimental_rerun()

elif st.session_state.screen == "results":
    st.markdown("<h2 style='text-align: center;'>PREDICTIONS</h2>", unsafe_allow_html=True)
    if st.session_state.preds:
        for line in st.session_state.preds:
            st.markdown(f"<p style='text-align:center;font-size:20px'>{line}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center;'>No matches found for OVER 2.5</p>", unsafe_allow_html=True)
    if st.button("BACK"):
        st.session_state.screen = "home"
        st.experimental_rerun()
