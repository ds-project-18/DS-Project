import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="DS Project: Inflation & Media", page_icon="📈", layout="centered")

# --- Main Title and Team ---
st.title("📊 Media, Inflation and Public Attention in Germany")
st.subheader("A Data Science Project")

# Team Info (HIER EURE NAMEN EINTRAGEN)
st.markdown("**Project Team:** Ali - Ajub - Olith - Mohamed")

st.divider()

# --- General Project Description ---
st.header("What is this project about?")
st.write(
    "In this project, we analyze and quantify the relationship between "
    "media coverage intensity, official inflation rates, and "
    "public web search behavior in Germany."
)
st.write(
    "We want to find out: Do people react directly to rising prices with their Google searches, "
    "or are they rather driven by media coverage?"
)

# --- Data Sources ---
st.subheader("Our Data Sources")
st.write("To answer our research questions, we combine three APIs:")

st.markdown(
    """
    * 🔍 **Google Trends API (pytrends):** To measure search interest for terms like 'Inflation' and 'Cost of living'.
    * 📰 **World News API:** To capture the frequency of news articles over specific time periods.
    * 💶 **Eurostat Statistics API:** To retrieve the official monthly inflation rate in Germany.
    """
)

st.divider()

st.info("🔜 In the next step, we will add the answers to our research questions and the interactive graphs here.")