import streamlit as st import pandas as pd import requests from datetime import datetime from collections import defaultdict from sklearn.ensemble import RandomForestClassifier from sklearn.model_selection import train_test_split, RandomizedSearchCV from sklearn.metrics import classification_report

st.set_page_config(page_title="O7 Predictor", layout="centered")

if 'screen' not in st.session_state: st.session_state.screen = 'home' if 'date_input' not in st.session_state: st.session_state.date_input = ''

st.markdown(""" <style> .big-font { font-size:36px !important; text-align: center; } .centered { display: flex; justify-content: center; align-items: center; flex-direction: column; } .button-style button { font-size: 24px !important; padding: 12px 24px; width: 200px; } </style> """, unsafe_allow_html=True)

def init_stats(): return { "partite": 0, "gol_fatti": 0, "gol_subiti": 0, "over35": 0, "gol_fatti_casa": 0, "gol_subiti_casa": 0, "over35_casa": 0, "partite_casa": 0, "gol_fatti_trasferta": 0, "gol_subiti_trasferta": 0, "over35_trasferta": 0, "partite_trasferta": 0 }

def load_matches(): urls = ["https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.1.json"] all_matches = [] for url in urls: try: r = requests.get(url) r.raise_for_status() data = r.json() league = url.split('/')[-1].split('.')[0] for match in data['matches']: match['league'] = league all_matches.extend(data['matches']) except Exception as e: st.error(f"Errore nel caricamento dati: {e}") return all_matches

def build_rankings(matches): league_tables = defaultdict(lambda: defaultdict(lambda: {"pt": 0, "gf": 0, "gs": 0})) for match in matches: if 'score' not in match or not match['score'].get('ft'): continue l = match['league'] t1, t2 = match['team1'], match['team2'] g1, g2 = match['score']['ft'] if g1 > g2: league_tables[l][t1]['pt'] += 3 elif g1 < g2: league_tables[l][t2]['pt'] += 3 else: league_tables[l][t1]['pt'] += 1 league_tables[l][t2]['pt'] += 1 league_tables[l][t1]['gf'] += g1 league_tables[l][t1]['gs'] += g2 league_tables[l][t2]['gf'] += g2 league_tables[l][t2]['gs'] += g1

rankings = {}
for league, teams in league_tables.items():
    sorted_teams = sorted(teams.items(), key=lambda x: (-x[1]['pt'], -(x[1]['gf'] - x[1]['gs'])))
    rankings[league] = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}
return rankings

def extract_features(matches, rankings): data = [] for match in matches: if 'score' not in match or not match['score'].get('ft'): continue t1, t2 = match['team1'], match['team2'] l = match['league'] g1, g2 = match['score']['ft'] total_goals = g1 + g2

f = {
        "media_gol_casa": g1,
        "media_gol_ospite": g2,
        "ranking_casa": rankings[l].get(t1, 20),
        "ranking_ospite": rankings[l].get(t2, 20),
        "label": int(total_goals > 2.5)
    }
    data.append(f)
return pd.DataFrame(data)

def train_model(df): X = df.drop(columns=['label']) y = df['label'] X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
clf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=5, cv=3, scoring='accuracy')
search.fit(X_train, y_train)
return search.best_estimator_

def predict_matches(model, matches, rankings, date): results = [] for match in matches: if match.get('date') != date or 'score' in match and match['score'].get('ft'): continue t1, t2 = match['team1'], match['team2'] l = match['league'] features = { "media_gol_casa": 1.5, "media_gol_ospite": 1.2, "ranking_casa": rankings[l].get(t1, 20), "ranking_ospite": rankings[l].get(t2, 20) } pred = model.predict(pd.DataFrame([features]))[0] if pred == 1: results.append(f"{t1} vs {t2} => OVER 2.5") return results

if st.session_state.screen == 'home': st.markdown("<div class='centered'><h1 class='big-font'>WELCOME TO O7</h1></div>", unsafe_allow_html=True) st.session_state.date_input = st.text_input("", placeholder="INPUT YYYY-MM-DD") if st.button("START", key="start", use_container_width=True): st.session_state.screen = 'loading' st.experimental_rerun()

elif st.session_state.screen == 'loading': st.markdown("<div class='centered'><h2>Loading, please wait...</h2></div>", unsafe_allow_html=True) try: matches = load_matches() rankings = build_rankings(matches) df = extract_features(matches, rankings) model = train_model(df) st.session_state.results = predict_matches(model, matches, rankings, st.session_state.date_input) st.session_state.screen = 'results' st.experimental_rerun() except Exception as e: st.error("Errore durante il processo. Riprova.") st.write(e) if st.button("Back"): st.session_state.screen = 'home' st.experimental_rerun()

elif st.session_state.screen == 'results': st.markdown("<div class='centered'><h2>OVER 2.5 PREDICTIONS</h2></div>", unsafe_allow_html=True) if 'results' in st.session_state and st.session_state.results: for r in st.session_state.results: st.markdown(f"<h3 style='text-align:center'>{r}</h3>", unsafe_allow_html=True) else: st.markdown("<h3 style='text-align:center'>No qualifying matches.</h3>", unsafe_allow_html=True)

if st.button("BACK", use_container_width=True):
    st.session_state.screen = 'home'
    st.experimental_rerun()

