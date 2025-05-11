import streamlit as st
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

st.set_page_config(page_title="Predittore Over 2.5", layout="wide")
st.title("Modello Predittivo Over 2.5 – Random Forest")

# --- CONFIG ---
FINESTRE = [3, 5, 10, 15]
URLS = [
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
  "https://raw.githubusercontent.com/openfootball/football.json/master/2025/mls.json"
]

@st.cache_data(show_spinner=False)
def fetch_matches():
    all_matches = []
    for url in URLS:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for match in data.get("matches", []):
                if "score" in match and match["score"].get("ft"):
                    match["league"] = data.get("name", "")
                    all_matches.append(match)
        except Exception as e:
            st.warning(f"Errore su {url}: {e}")
    return all_matches

def safe_div(a, b):
    return a / b if b else 0

def calcola_punti(team, matches):
    win, draw = 0, 0
    for m in matches:
        if team not in (m["team1"], m["team2"]):
            continue
        ft = m.get("score", {}).get("ft", [0, 0])
        if not ft:
            continue
        g1, g2 = ft
        if m["team1"] == team:
            if g1 > g2: win += 1
            elif g1 == g2: draw += 1
        elif m["team2"] == team:
            if g2 > g1: win += 1
            elif g2 == g1: draw += 1
    return win * 3 + draw

def gol_per_finestra(team, matches, finestre=FINESTRE):
    risultati = {f"gol_{f}": 0 for f in finestre}
    partite = []
    for m in matches:
        if team not in (m["team1"], m["team2"]):
            continue
        ft = m.get("score", {}).get("ft")
        if not ft:
            continue
        try:
            d = datetime.strptime(m["date"], "%Y-%m-%d")
            g = ft[0] if m["team1"] == team else ft[1]
            partite.append((d, g))
        except:
            continue
    partite.sort(reverse=True)
    for f in finestre:
        risultati[f"gol_{f}"] = sum(g for _, g in partite[:f])
    return risultati

def calcola_forma_agile(gol_data):
    score = 0
    if safe_div(gol_data['gol_3'], 3) >= 1.5:
        score += 2
    if safe_div(gol_data['gol_5'], 5) >= 1.4:
        score += 1
    if safe_div(gol_data['gol_10'], 10) >= 1.3:
        score += 1
    return score

def ultima_over35(team, matches):
    partite = []
    for m in matches:
        if team not in (m['team1'], m['team2']):
            continue
        ft = m.get('score', {}).get('ft')
        if not ft:
            continue
        try:
            d = datetime.strptime(m['date'], "%Y-%m-%d")
            partite.append((d, sum(ft)))
        except:
            continue
    if not partite:
        return 0
    partite.sort(reverse=True)
    return 1 if partite[0][1] > 3 else 0

@st.cache_data(show_spinner=False)
def build_dataset(matches):
    records = []
    for m in matches:
        ft = m.get('score', {}).get('ft')
        if not ft:
            continue
        total_goals = sum(ft)
        label = 1 if total_goals >= 3 else 0

        team1, team2 = m['team1'], m['team2']
        date_m = datetime.strptime(m['date'], "%Y-%m-%d")
        past_matches = [
            x for x in matches if x.get('score', {}).get('ft') and datetime.strptime(x['date'], "%Y-%m-%d") < date_m
        ]

        if not past_matches:
            continue

        f_team1 = gol_per_finestra(team1, past_matches)
        f_team2 = gol_per_finestra(team2, past_matches)
        punti1 = calcola_punti(team1, past_matches)
        punti2 = calcola_punti(team2, past_matches)
        overcorr1 = ultima_over35(team1, past_matches)
        overcorr2 = ultima_over35(team2, past_matches)
        forma1 = calcola_forma_agile(f_team1)
        forma2 = calcola_forma_agile(f_team2)

        features = {
            **{f"{k}_casa": v for k, v in f_team1.items()},
            **{f"{k}_ospite": v for k, v in f_team2.items()},
            "punti_casa": punti1,
            "punti_ospite": punti2,
            "overcorr_casa": overcorr1,
            "overcorr_ospite": overcorr2,
            "forma_agile_casa": forma1,
            "forma_agile_ospite": forma2,
            "over25_label": label
        }
        records.append(features)

    df = pd.DataFrame(records)
    return df.dropna()

def train_model(df):
    X = df.drop(columns=["over25_label"])
    y = df["over25_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    st.subheader("Performance del Modello")
    st.markdown(f"**AUC ROC**: `{auc:.4f}`")

    importanze = clf.feature_importances_
    df_importanza = pd.DataFrame({
        "feature": X.columns,
        "importanza": importanze
    }).sort_values(by="importanza", ascending=False)

    st.markdown("**Importanza delle Feature**")
    st.dataframe(df_importanza.reset_index(drop=True))

    return clf, X.columns.tolist()

def predict_next_matches(model, all_matches, feature_names, data_analisi):
    upcoming = [
        m for m in all_matches 
        if not m.get("score", {}).get("ft") and m["date"] >= data_analisi.strftime("%Y-%m-%d")
    ]

    if not upcoming:
        st.warning("Nessuna partita trovata per la data selezionata.")
        return

    risultati = []

    for m in upcoming:
        team1, team2 = m["team1"], m["team2"]
        date_m = datetime.strptime(m["date"], "%Y-%m-%d")
        past = [
            x for x in all_matches 
            if x.get("score", {}).get("ft") and datetime.strptime(x["date"], "%Y-%m-%d") < date_m
        ]

        if not past:
            continue

        f_team1 = gol_per_finestra(team1, past)
        f_team2 = gol_per_finestra(team2, past)
        punti1 = calcola_punti(team1, past)
        punti2 = calcola_punti(team2, past)
        overcorr1 = ultima_over35(team1, past)
        overcorr2 = ultima_over35(team2, past)
        forma1 = calcola_forma_agile(f_team1)
        forma2 = calcola_forma_agile(f_team2)

        feat = {
            **{f"{k}_casa": v for k, v in f_team1.items()},
            **{f"{k}_ospite": v for k, v in f_team2.items()},
            "punti_casa": punti1,
            "punti_ospite": punti2,
            "overcorr_casa": overcorr1,
            "overcorr_ospite": overcorr2,
            "forma_agile_casa": forma1,
            "forma_agile_ospite": forma2
        }

        if not set(feature_names).issubset(feat.keys()):
            continue

        row = np.array([feat[k] for k in feature_names]).reshape(1, -1)
        prob = model.predict_proba(row)[0, 1]
        risultati.append({
            "data": m["date"],
            "match": f"{team1} vs {team2}",
            "probabilità Over 2.5": round(prob, 2),
            "decisione": "GIOCA" if prob >= 0.7 else "NON GIOCARE"
        })

    if risultati:
        st.subheader(f"Previsioni Over 2.5 dal {data_analisi.strftime('%Y-%m-%d')}")
        df_result = pd.DataFrame(risultati)
        st.dataframe(df_result)
    else:
        st.info("Nessuna previsione disponibile.")

def main():
    st.sidebar.header("Parametri")
    use_today = st.sidebar.radio("Seleziona la data per l'analisi:", ["Oggi", "Data personalizzata"])

    if use_today == "Oggi":
        data_analisi = datetime.now()
    else:
        data_analisi = st.sidebar.date_input("Scegli la data", datetime.now())

    st.info("Caricamento e preparazione dei dati...")

    all_matches = fetch_matches()
    if not all_matches:
        st.stop()

    df = build_dataset(all_matches)
    if df.empty:
        st.error("Dataset vuoto. Controlla i dati disponibili.")
        st.stop()

    model, feature_names = train_model(df)

    st.success("Modello pronto! Ora puoi analizzare le partite future.")
    predict_next_matches(model, all_matches, feature_names, data_analisi)

if __name__ == "__main__":
    main()
    
