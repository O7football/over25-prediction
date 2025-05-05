import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === Streamlit Page Config ===
st.set_page_config(page_title="O7 Predictor", layout="centered")

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
        "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/fr.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/fr.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/at.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/at.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/au.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/be.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/de.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/de.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/eg.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.3.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.4.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/es.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/es.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/gr.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/it.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/it.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/nl.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/pt.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/sco.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/tr.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2025/ar.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2025/br.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2025/cn.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2025/co.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2025/copa.l.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2025/jp.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2025/mls.json",
  
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/at.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/de.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/de.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/en.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/en.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/es.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/fr.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/it.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/nl.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/pt.1.json",
  
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/at.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/de.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/de.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/en.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/en.2.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/es.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/fr.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/it.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/nl.1.json",
  "https://raw.githubusercontent.com/openfootball/football.json/master/2022-23/pt.1.json"
    ]
    all_matches = []
    for url in urls:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            league = url.split('/')[-1].split('.')[0]
            for match in data['matches']:
                match['league'] = league
            all_matches.extend(data['matches'])
    return all_matches

def build_rankings(matches):
    league_tables = defaultdict(lambda: defaultdict(lambda: {"pt": 0, "gf": 0, "gs": 0}))
    for match in matches:
        if 'score' not in match or not match['score'].get('ft'):
            continue
        l = match['league']
        t1, t2 = match['team1'], match['team2']
        g1, g2 = match['score']['ft']
        if g1 > g2:
            league_tables[l][t1]['pt'] += 3
        elif g1 < g2:
            league_tables[l][t2]['pt'] += 3
        else:
            league_tables[l][t1]['pt'] += 1
            league_tables[l][t2]['pt'] += 1
        league_tables[l][t1]['gf'] += g1
        league_tables[l][t1]['gs'] += g2
        league_tables[l][t2]['gf'] += g2
        league_tables[l][t2]['gs'] += g1

    rankings = {}
    for league, teams in league_tables.items():
        sorted_teams = sorted(teams.items(), key=lambda x: (-x[1]['pt'], -(x[1]['gf'] - x[1]['gs'])))
        rankings[league] = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}
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

        t1, t2 = match['team1'], match['team2']
        l = match['league']
        g1, g2 = match['score']['ft']
        total_goals = g1 + g2

        casa = init_stats()
        ospite = init_stats()

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
                if team1 == t1:
                    casa['over35_casa'] += int(tot > 3)
                    casa['partite_casa'] += 1
            if t2 in (team1, team2):
                ospite['partite'] += 1
                ospite['gol_fatti'] += s2 if team2 == t2 else s1
                ospite['gol_subiti'] += s1 if team2 == t2 else s2
                ospite['over35'] += int(tot > 3)
                if team2 == t2:
                    ospite['over35_trasferta'] += int(tot > 3)
                    ospite['partite_trasferta'] += 1

        if casa['partite'] < 3 or ospite['partite'] < 3:
            continue

        media_gol_casa = (casa['gol_fatti'] + casa['gol_subiti']) / casa['partite']
        media_gol_ospite = (ospite['gol_fatti'] + ospite['gol_subiti']) / ospite['partite']
        partite_casa = casa['partite_casa'] or 1
        partite_ospite = ospite['partite_trasferta'] or 1

        form_t1 = get_recent_stats(matches, t1, 5)
        form_t2 = get_recent_stats(matches, t2, 5)

        btts_t1 = get_btts_ratio(get_recent_stats(matches, t1, 10))
        btts_t2 = get_btts_ratio(get_recent_stats(matches, t2, 10))

        features = {
            "media_gol_casa": media_gol_casa,
            "media_gol_ospite": media_gol_ospite,
            "over35_casa_totale": casa['over35'] / casa['partite'],
            "over35_ospite_totale": ospite['over35'] / ospite['partite'],
            "over35_casa_homeonly": casa['over35_casa'] / partite_casa,
            "over35_ospite_awayonly": ospite['over35_trasferta'] / partite_ospite,
            "avg_goal_match": ((media_gol_casa + media_gol_ospite) / 2) / casa['partite'],
            "form_gol_casa": sum(g for _, g, _ in form_t1) / 5,
            "form_gol_ospite": sum(g for _, g, _ in form_t2) / 5,
            "btts_ratio_casa": btts_t1,
            "btts_ratio_ospite": btts_t2,
            "ranking_casa": rankings[l].get(t1, 20),
            "ranking_ospite": rankings[l].get(t2, 20),
            "label": int(total_goals > 2.5)
        }
        data.append(features)
    return pd.DataFrame(data)

def predict_upcoming_matches(model, matches, date_input, rankings):
    predictions = []
    for match in matches:
        if match.get('date') != date_input or 'score' in match and match['score'].get('ft'):
            continue

        t1, t2 = match['team1'], match['team2']
        l = match['league']

        casa = init_stats()
        ospite = init_stats()

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
                if team1 == t1:
                    casa['over35_casa'] += int(tot > 3)
                    casa['partite_casa'] += 1
            if t2 in (team1, team2):
                ospite['partite'] += 1
                ospite['gol_fatti'] += s2 if team2 == t2 else s1
                ospite['gol_subiti'] += s1 if team2 == t2 else s2
                ospite['over35'] += int(tot > 3)
                if team2 == t2:
                    ospite['over35_trasferta'] += int(tot > 3)
                    ospite['partite_trasferta'] += 1

        if casa['partite'] < 3 or ospite['partite'] < 3:
            continue

        media_gol_casa = (casa['gol_fatti'] + casa['gol_subiti']) / casa['partite']
        media_gol_ospite = (ospite['gol_fatti'] + ospite['gol_subiti']) / ospite['partite']
        partite_casa = casa['partite_casa'] or 1
        partite_ospite = ospite['partite_trasferta'] or 1

        form_t1 = get_recent_stats(matches, t1, 5)
        form_t2 = get_recent_stats(matches, t2, 5)
        btts_t1 = get_btts_ratio(get_recent_stats(matches, t1, 10))
        btts_t2 = get_btts_ratio(get_recent_stats(matches, t2, 10))

        features = {
            "media_gol_casa": media_gol_casa,
            "media_gol_ospite": media_gol_ospite,
            "over35_casa_totale": casa['over35'] / casa['partite'],
            "over35_ospite_totale": ospite['over35'] / ospite['partite'],
            "over35_casa_homeonly": casa['over35_casa'] / partite_casa,
            "over35_ospite_awayonly": ospite['over35_trasferta'] / partite_ospite,
            "avg_goal_match": ((media_gol_casa + media_gol_ospite) / 2) / casa['partite'],
            "form_gol_casa": sum(g for _, g, _ in form_t1) / 5,
            "form_gol_ospite": sum(g for _, g, _ in form_t2) / 5,
            "btts_ratio_casa": btts_t1,
            "btts_ratio_ospite": btts_t2,
            "ranking_casa": rankings[l].get(t1, 20),
            "ranking_ospite": rankings[l].get(t2, 20),
        }

        X_pred = pd.DataFrame([features])
        prediction = model.predict(X_pred)[0]
        prob = model.predict_proba(X_pred)[0][1]

        if prediction == 1 and prob >= 0.75:
            predictions.append((t1, t2, prob))
    return predictions

# === Streamlit UI ===
st.markdown("<h1 style='text-align: center; color: #007BFF;'>WELCOME TO O7</h1>", unsafe_allow_html=True)

date_input = st.text_input(" ", placeholder="INPUT YYYY-MM-DD")
start_col = st.columns([1, 1, 1])
with start_col[1]:
    run = st.button("START", use_container_width=True)

if run:
    with st.spinner("Training model and making predictions..."):
        matches = load_matches()
        rankings = build_rankings(matches)
        df = extract_features(matches, rankings)

        if df.empty:
            st.error("Not enough data to train the model.")
        else:
            X = df.drop(columns=["label"])
            y = df["label"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            results = predict_upcoming_matches(model, matches, date_input.strip(), rankings)

            if results:
                st.success("Predictions Complete")
                for t1, t2, prob in results:
                    st.markdown(f"**{t1} vs {t2}** => OVER 2.5 prob: **{prob:.2f}**")
            else:
                st.warning("No strong predictions (prob >= 0.0) found for the selected date.")
