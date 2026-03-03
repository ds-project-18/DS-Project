import streamlit as st

# --- Konfiguration der Webseite ---
st.set_page_config(page_title="DS Projekt: Inflation & Medien", page_icon="📈", layout="centered")

# --- Haupttitel und Team ---
st.title("📊 Media, Inflation and Public Attention in Germany")
st.subheader("Ein Data Science Projekt")

# Team-Info
st.markdown("**Projektteam:** Ali - Olith - Ajub - Mohamed")

st.divider()

# --- Allgemeine Projektbeschreibung ---
st.header("Worum geht es in diesem Projekt?")
st.write(
    "In diesem Projekt analysieren und quantifizieren wir die Beziehung zwischen "
    "der Intensität der Medienberichterstattung, den offiziellen Inflationsraten und "
    "dem öffentlichen Web-Suchverhalten in Deutschland."
)
st.write(
    "Wir möchten herausfinden: Reagieren die Menschen mit ihren Google-Suchen direkt auf "
    "steigende Preise, oder werden sie eher durch die Berichterstattung der Medien dazu getrieben?"
)

# --- Datenquellen ---
st.subheader("Unsere Datenquellen")
st.write("Um unsere 9 Forschungsfragen zu beantworten, kombinieren wir drei Schnittstellen:")

st.markdown(
    """
    * 🔍 **Google Trends API (pytrends):** Zur Messung des Suchinteresses für Begriffe wie "Inflation" oder "Cost of living".
    * 📰 **World News API:** Zur Erfassung der Häufigkeit von Nachrichtenartikeln über spezifische Zeiträume.
    * 💶 **Eurostat Statistics API:** Zum Abruf der offiziellen monatlichen Inflationsrate in Deutschland.
    """
)

st.divider()

st.info("🔜 Im nächsten Schritt fügen wir hier die Antworten auf unsere Forschungsfragen und die interaktiven Graphen ein.")