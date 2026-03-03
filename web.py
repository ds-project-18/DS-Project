import streamlit as st

# --- Seiten-Konfiguration ---
st.set_page_config(page_title="DS Projekt: Inflation", layout="wide")

# --- Header ---
st.title("Media, Inflation and Public Attention in Germany")
st.markdown("**Ein Data Science Projekt zur Quantifizierung von Medienpräsenz und Suchverhalten**")
[cite_start]st.markdown("*Projektteam: Ali - Olith - Ajub - Mohamed | Friday, Week 1*") # [cite: 3, 4]

st.divider()

# --- BEREITS BEANTWORTETE FRAGE 3 ---
[cite_start]st.header("Q3: How closely does Google search interest for 'Inflation' track the official monthly inflation rate in Germany over time?") # [cite: 65]
st.success("**Ergebnis: Sehr eng (Korrelation: 0.89).**")
st.markdown("""
Unsere Datenanalyse zeigt eine extrem starke positive Korrelation zwischen dem Google-Suchinteresse und der offiziellen HVPI-Inflationsrate von Eurostat. 
* **Die Schock-Phase (2021-2022):** Ein fast synchroner, steiler Anstieg beider Metriken. Die Bevölkerung reagierte sofort auf steigende Preise mit einem stark erhöhten Informationsbedürfnis.
* **Die Gewöhnungsphase (ab Ende 2022):** Beide Kurven fallen nach dem Überschreiten der 10%-Marke wieder ab, wobei das Suchinteresse dem Rückgang der offiziellen Rate eng folgt.
""")

# Hier lädt Python euer gespeichertes Bild direkt in die Webseite!
try:
    st.image("Inflation.png", caption="Deutschland: Suchinteresse nach 'Inflation' vs. Offizielle Inflationsrate", use_container_width=True)
except FileNotFoundError:
    st.error("⚠️ Bild 'Inflation.png' wurde nicht gefunden. Bitte stelle sicher, dass es im selben Ordner wie diese app.py Datei liegt!")

st.divider()

# --- DIE RESTLICHEN 8 FORSCHUNGSFRAGEN (Als aufklappbare Boxen) ---
st.header("Weitere Forschungsfragen des Projekts")

[cite_start]with st.expander("Q1: How strong is the relationship between the daily number of German news articles mentioning 'Inflation' and the Google Trends search index for 'Inflation' in Germany?"): # [cite: 63]
    st.info("**Platzhalter-Ergebnis:** Hier kommt später der Korrelationswert der News-API rein.")

[cite_start]with st.expander("Q2: To what extent does increased media coverage of inflation influence Google search interest for related terms such as 'Energy costs' and 'Cost of living' in Germany?"): # [cite: 64]
    st.info("**Platzhalter-Ergebnis:** Hier fügen wir die Analyse der verwandten Suchbegriffe ein.")

[cite_start]with st.expander("Q4: To what extent does media coverage on day t explain variation in Google search interest on day t+1?"): # [cite: 71]
    st.info("**Platzhalter-Ergebnis:** Hier kommt die Lag-Analyse (Tag t vs. Tag t+1) rein.")

[cite_start]with st.expander("Q5: How long does elevated Google search interest persist following major peaks in media coverage?"): # [cite: 72]
    st.info("**Platzhalter-Ergebnis:** Hier definieren wir die 'Halbwertszeit' der öffentlichen Aufmerksamkeit.")

[cite_start]with st.expander("Q6: How do temporal lags between media coverage and public search interest vary over time?"): # [cite: 73]
    st.info("**Platzhalter-Ergebnis:** Hier analysieren wir, ob die Menschen im Laufe der Jahre schneller auf News reagieren.")

[cite_start]with st.expander("Q7: How does public search interest differ during months with high inflation rates (e.g., above 5%)?"): # [cite: 74]
    st.info("**Platzhalter-Ergebnis:** Vergleich der durchschnittlichen Suchanfragen in Hoch- vs. Niedriginflationsmonaten.")

[cite_start]with st.expander("Q8: How does the explanatory power of media coverage intensity compare to that of the official inflation rate in accounting for variations in Google search interest?"): # [cite: 75]
    st.info("**Platzhalter-Ergebnis:** Ergebnisse der multiplen Regressionsanalyse (Medien vs. Reale Zahlen).")

[cite_start]with st.expander("Q9: How does the relationship between inflation rates and Google search interest change during periods of high versus low media coverage?"): # [cite: 76]
    st.info("**Platzhalter-Ergebnis:** Analyse des 'Verstärker-Effekts' durch die Medien.")