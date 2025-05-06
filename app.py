import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import date

# === INIT ===
st.set_page_config(page_title="Football Over 2.5 Predictor", layout="wide")

def init_team_stats():
    return {
        "games": 0, "goals_for": 0, "goals_against": 0, "over35": 0,
        "home_games": 0, "home_over35": 0,
        "away_games": 0, "away_over35": 0
    }

def load_all_matches(urls):
    all_matches = []
    for url in urls:
        try:
            res = requests.get(url)
            if res.status_code == 200:
                all_matches.extend(res.json().get("matches", []))
        except Exception as e:
            st.warning(f"Error loading {url}: {e}")
    return all_matches

def update_stats(stats, match, team_name, is_home):
    team1, team2 = match["team1"], match["team2"]
    g1, g2 = match["score"]["ft"]
    total = g1 + g2

    stats["games"] += 1
    if team_name == team1:
        stats["goals_for"] += g1
        stats["goals_against"] += g2
    else:
        stats["goals_for"] += g2
        stats["goals_against"] += g1

    stats["over35"] += int(total > 3)

    if is_home:
        stats["home_games"] += 1
        stats["home_over35"] += int(total > 3)
    else:
        stats["away_games"] += 1
        stats["away_over35"] += int(total > 3)

def safe_div(a, b):
    return a / b if b else 0

def extract_features(matches):
    rows = []
    for match in matches:
        if "score" not in match or not match["score"].get("ft"):
            continue

        home, away = match["team1"], match["team2"]
        g1, g2 = match["score"]["ft"]
        total_goals = g1 + g2

        home_stats, away_stats = init_team_stats(), init_team_stats()

        for past in matches:
            if past == match or "score" not in past or not past["score"].get("ft"):
                continue

            if home in (past["team1"], past["team2"]):
                update_stats(home_stats, past, home, home == past["team1"])
            if away in (past["team1"], past["team2"]):
                update_stats(away_stats, past, away, away == past["team1"])

        if home_stats["games"] < 3 or away_stats["games"] < 3:
            continue

        row = {
            "home_avg_goals": safe_div(home_stats["goals_for"] + home_stats["goals_against"], home_stats["games"]),
            "away_avg_goals": safe_div(away_stats["goals_for"] + away_stats["goals_against"], away_stats["games"]),
            "home_over35_rate": safe_div(home_stats["over35"], home_stats["games"]),
            "away_over35_rate": safe_div(away_stats["over35"], away_stats["games"]),
            "home_over35_home": safe_div(home_stats["home_over35"], home_stats["home_games"]),
            "away_over35_away": safe_div(away_stats["away_over35"], away_stats["away_games"]),
            "avg_match_goal": safe_div((home_stats["goals_for"] + away_stats["goals_for"]), 2),
            "label": int(total_goals > 2.5)
        }
        rows.append(row)
    return pd.DataFrame(rows)

def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, classification_report(y_test, model.predict(X_test), output_dict=True)

def predict_date_matches(model, matches, date_str):
    predictions = []

    for match in matches:
        if match.get("date") != date_str or match.get("score"):
            continue

        home, away = match["team1"], match["team2"]
        home_stats, away_stats = init_team_stats(), init_team_stats()

        for past in matches:
            if "score" not in past or not past["score"].get("ft"):
                continue
            if home in (past["team1"], past["team2"]):
                update_stats(home_stats, past, home, home == past["team1"])
            if away in (past["team1"], past["team2"]):
                update_stats(away_stats, past, away, away == past["team1"])

        if home_stats["games"] < 3 or away_stats["games"] < 3:
            continue

        row = pd.DataFrame([{
            "home_avg_goals": safe_div(home_stats["goals_for"] + home_stats["goals_against"], home_stats["games"]),
            "away_avg_goals": safe_div(away_stats["goals_for"] + away_stats["goals_against"], away_stats["games"]),
            "home_over35_rate": safe_div(home_stats["over35"], home_stats["games"]),
            "away_over35_rate": safe_div(away_stats["over35"], away_stats["games"]),
            "home_over35_home": safe_div(home_stats["home_over35"], home_stats["home_games"]),
            "away_over35_away": safe_div(away_stats["away_over35"], away_stats["away_games"]),
            "avg_match_goal": safe_div((home_stats["goals_for"] + away_stats["goals_for"]), 2)
        }])

        pred = model.predict(row)[0]
        prob = model.predict_proba(row)[0][1]

        predictions.append({
            "Match": f"{home} vs {away}",
            "Prediction": "OVER 2.5" if pred else "NO OVER",
            "Confidence": round(prob, 2)
        })

    return pd.DataFrame(predictions)

# === STREAMLIT GUI ===
st.title("Football Over 2.5 Goal Predictor")
st.markdown("Predict if a match will have more than 2.5 goals based on historical data.")

with st.spinner("Loading match data..."):
    urls = [
        "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/it.1.json",
        "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/en.1.json",
        "https://raw.githubusercontent.com/openfootball/football.json/master/2024-25/es.1.json",
    ]
    matches = load_all_matches(urls)
    df = extract_features(matches)

if df.empty:
    st.error("Not enough data to train the model.")
else:
    model, report = train_model(df)
    st.success("Model trained successfully.")

    # Date picker
    selected_date = st.date_input("Select a match date", value=date.today())
    selected_date_str = selected_date.strftime("%Y-%m-%d")

    if st.button("Predict Matches"):
        results = predict_date_matches(model, matches, selected_date_str)
        if not results.empty:
            st.subheader(f"Predictions for {selected_date_str}")
            st.dataframe(results)
        else:
            st.info("No matches found for this date or not enough past data.")

    if st.checkbox("Show model evaluation report"):
        st.json(report)
        
