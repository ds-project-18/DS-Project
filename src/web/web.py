"""
Main Streamlit Application: Media, Public Interest & Inflation in Germany (2022-2024)
This dashboard analyzes the psychological and macroeconomic effects of the inflation crisis 
by correlating official economic data (Destatis) with media volume (GDELT) and public panic (Google Trends).
"""

# --- IMPORTS ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path
from scipy.stats import pearsonr

# --- PAGE SETUP ---
# Configure the page title, icon, and default layout mode (wide)
st.set_page_config(page_title="Inflation Impact Analysis", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
# Injecting custom CSS to enhance typography and UI elements for a more professional look
st.markdown("""
    <style>
    /* Styling for the top metric cards */
    .stMetric {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
    }

    /* Increase readability for paragraphs and lists */
    .stMarkdown p, .stMarkdown li, .stAlert p {
        font-size: 1.08rem !important;
        line-height: 1.65 !important;
    }

    h1, h2, h3 {
        line-height: 1.25 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- THEME-SAFE COLORS ---
# Centralized color palette ensuring consistency across all Plotly charts
COLORS = {
    "news_fill": "rgba(181, 101, 29, 0.22)",      
    "news_line": "rgba(181, 101, 29, 0.95)",
    "inflation": "#3b82f6",
    "energy": "#f59e0b",
    "cost": "#22c55e",
    "official": "#ef4444",
    "food_fill": "rgba(34, 197, 94, 0.20)",
    "food_line": "#22c55e",
    "decay": "rgba(239, 68, 68, 0.12)"
}

layout_template = "streamlit"

# --- UI LABEL MAPPING ---
# Dictionary to map raw DataFrame column names to clean, human-readable UI labels
LABELS = {
    "news_count": "News Count",
    "inflation_rate": "Inflation Rate (%)",
    "energy_price_index": "Energy Price Index (index points)",
    "food_price_index": "Food Price Index (index points)",
    "unemployment_rate": "Unemployment Rate (%)",
    "Inflation": "Google Trends: Inflation",
    "Energiekosten": "Google Trends: Energy Costs",
    "Lebenshaltungskosten": "Google Trends: Cost of Living",
    "news_count_lag1": "News Count (t-1)"
}

def clean_label(name):
    """Returns the mapped UI label or a title-cased fallback if not found in LABELS."""
    return LABELS.get(name, name.replace("_", " ").title())

def add_trendline_and_corr(fig, x, y, x_name, y_name, color="#94a3b8"):
    """
    Calculates the Pearson correlation coefficient and linear regression line (y = mx + b),
    then overlays the trendline and statistical annotation onto an existing Plotly figure.
    """
    # Drop NaNs to ensure polyfit and pearsonr don't fail
    tmp = pd.DataFrame({x_name: x, y_name: y}).dropna()
    
    if len(tmp) > 2:
        r = tmp[x_name].corr(tmp[y_name])
        m, b = np.polyfit(tmp[x_name], tmp[y_name], 1)
        
        # Generate points for the trendline
        x_line = np.linspace(tmp[x_name].min(), tmp[x_name].max(), 100)
        y_line = m * x_line + b
        
        # Add the dotted trendline trace
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Trendline (r = {r:.2f})",
            line=dict(color=color, dash="dash")
        ))
        
        # Add correlation text annotation in the top-left corner
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            text=f"Pearson r = {r:.2f}",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.05)"
        )
    return fig

# --- DATA LOADING & CLEANING ---
# Establish absolute paths to ensure the app runs correctly regardless of the execution directory
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

@st.cache_data
def load_data():
    """
    Loads raw CSVs, standardizes date columns, merges all data sources into a single 
    monthly DataFrame, and engineers new analytical features.
    Cached via Streamlit to prevent reloading data on every UI interaction.
    """
    try:
        # Load all datasets
        df_inflation = pd.read_csv(DATA_DIR / "inflation.csv")
        df_energy = pd.read_csv(DATA_DIR / "energy.csv")
        df_food = pd.read_csv(DATA_DIR / "food.csv")
        df_labour = pd.read_csv(DATA_DIR / "labour.csv")
        df_news = pd.read_csv(DATA_DIR / "news_monthly.csv")
        df_trends = pd.read_csv(DATA_DIR / "trends_monthly.csv")

        def find_and_convert_date(df):
            """Helper to robustly locate and convert various date column naming conventions."""
            for col in df.columns:
                if col.lower() in ['month', 'time', 'date', 'datum', 'period']:
                    df['Date'] = pd.to_datetime(df[col])
                    return df
            raise KeyError(f"Could not find a date column. Your columns are: {list(df.columns)}")

        # Standardize 'Date' column across all dataframes
        df_inflation = find_and_convert_date(df_inflation)
        df_energy = find_and_convert_date(df_energy)
        df_food = find_and_convert_date(df_food)
        df_labour = find_and_convert_date(df_labour)
        df_news = find_and_convert_date(df_news)
        df_trends = find_and_convert_date(df_trends)

        # Aggregate daily/weekly labour data into monthly averages
        df_labour = df_labour.groupby('Date')['unemployment_rate'].mean().reset_index()

        # Iteratively merge all datasets on the 'Date' column using outer joins 
        # to ensure no time periods are dropped prematurely
        df = pd.merge(df_inflation[['Date', 'inflation_rate']], df_energy[['Date', 'energy_price_index']], on='Date', how='outer')
        df = pd.merge(df, df_food[['Date', 'food_price_index']], on='Date', how='outer')
        df = pd.merge(df, df_labour[['Date', 'unemployment_rate']], on='Date', how='outer')
        df = pd.merge(df, df_news[['Date', 'news_count']], on='Date', how='outer')
        df = pd.merge(df, df_trends[['Date', 'Inflation', 'Energiekosten', 'Lebenshaltungskosten']], on='Date', how='outer')
        
        # Sort chronologically and drop rows where official inflation data is missing
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.dropna(subset=['inflation_rate']).reset_index(drop=True)
        
        # Extract year as string for dropdown filters
        df['Year'] = df['Date'].dt.year.astype(str)
        
        # --- FEATURE ENGINEERING ---
        # 1. Calculate Lagged Spillover (Shift news back by 1 month to represent t-1)
        df['news_count_lag1'] = df['news_count'].shift(1)
        
        # 2. Categorize prior month media volume
        avg_news = df['news_count'].mean()
        df['prev_month_news_level'] = np.where(df['news_count_lag1'] > avg_news, 'High News Prior Month', 'Low News Prior Month')
        df.loc[df['news_count_lag1'].isna(), 'prev_month_news_level'] = None 

        # 3. Define specific historical crisis phases based on the 2022 timeline
        df['Phase'] = 'Baseline'
        df.loc[(df['Date'] >= '2022-09-01') & (df['Date'] <= '2022-11-01'), 'Phase'] = 'Peak Crisis'
        df.loc[(df['Date'] > '2022-11-01') & (df['Date'] <= '2023-03-01'), 'Phase'] = 'Decay Period'

        # 4. Psychological Threshold Tagging (Above/Below 5% Inflation)
        df['Inflation_Level'] = np.where(df['inflation_rate'] > 5.0, '> 5% (High)', '<= 5% (Low)')

        return df
        
    except Exception as e:
        # Graceful error handling for the UI
        st.error(f"🚨 Error loading data: {e}")
        st.stop()

# Initialize data and global UI variables
df_main = load_data()
available_years = ["All Years"] + sorted(df_main['Year'].unique().tolist())
bg_color = 'rgba(128, 128, 128, 0.3)'

# --- DYNAMIC SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("👨‍💻 Team")
    st.info("""
    **Data Science Project 2026**
    
    *Members:*
    - Ali
    - Ajub
    - Olith
    - Mohamed 
    """)
    st.markdown("---")
    
    # State-controlled navigation menu
    page = st.radio("📌 Navigation", [
        "📖 Executive Summary",
        "📊 Interactive Dashboard", 
        "🎯 Project Summary"
    ])

# --- MAIN TITLE ---
st.title("📊 Media, Public Interest & Inflation in Germany")

# ==========================================
# PAGE 1: EXECUTIVE SUMMARY
# ==========================================
if page == "📖 Executive Summary":
    st.header("📖 Executive Summary: About This Project")
    with st.container(border=True):
        st.markdown("""
        **The Context:** Starting in early 2022, global exogenous shocks (primarily the European energy crisis and supply chain disruptions) triggered historic inflation rates in Germany, peaking at over 11% in autumn 2022. This period placed immense financial pressure on households and businesses alike.

        **Our Research Focus:** While traditional economics only looks at numbers, this project bridges the gap between **hard economic indicators** (inflation rates, energy prices, food prices, unemployment) and **public psychology**. We analyze:
        1. **The Media's Role:** How did news coverage shape public anxiety?
        2. **Public Attention:** How did the German population react, measured through their real-time Google search behavior for terms like "Inflation", "Energy costs", and "Cost of living"?
        3. **Macroeconomic Interactions:** Did actual price hikes drive public panic, or was it primarily media-induced? And how resilient was the labor market during this storm?

        **Methodology:** We aggregate multiple datasets into a monthly format to perform our analysis:
        * **[Statistisches Bundesamt (Destatis)](https://www.destatis.de/EN/Home/_node.html)**: For official hard economic indicators (Inflation Rate, Energy Price Index, Food Price Index, Unemployment Rate).
        * **[Google Trends](https://trends.google.com/)**: To measure real-time public attention and search behavior for specific anxiety-driven terms.
        * **[The GDELT Project](https://www.gdeltproject.org/)**: To analyze the global database of news events and quantify the exact volume of media reporting over time.
        """)
    st.info("👈 **Please use the navigation menu on the left to explore the Interactive Dashboard!**")


# ==========================================
# PAGE 2: INTERACTIVE DASHBOARD (WITH TABS)
# ==========================================
elif page == "📊 Interactive Dashboard":
    
    # KPI / Metric Row
    st.markdown("### ⚡ Crisis at a Glance (2022 - 2024)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Inflation Rate", f"{df_main['inflation_rate'].max():.1f}%", "Oct 2022", delta_color="inverse")
    m2.metric("Peak Energy Index", f"{df_main['energy_price_index'].max():.1f} index pts", "Nov 2022", delta_color="inverse")
    m3.metric("Highest News Volume", f"{df_main['news_count'].max():.0f} articles", "Sep 2022", delta_color="inverse")
    m4.metric("Avg Unemployment", f"{df_main['unemployment_rate'].mean():.1f}%", "Highly Stable", delta_color="normal")
    st.markdown("---")

    # Set up logical tabs corresponding to the research questions
    tab1, tab2, tab3, tab4 = st.tabs([
        "📰 Q1 - Q4: Media & Attention", 
        "📈 Q5 - Q6: Inflation vs. Trends", 
        "🛒 Q7 - Q8: Energy & Food Prices",
        "🌐 Q9: Macro Interactions"
    ])

    # --- TAB 1: MEDIA & PUBLIC ATTENTION ---
    with tab1:
        # Dynamic filter for current tab
        y1 = st.selectbox("📅 Filter Timeline for Tab 1:", available_years, key="y1")
        dff1 = df_main[df_main['Year'] == y1] if y1 != "All Years" else df_main.copy()
        
        # QUESTION 1
        with st.container(border=True):
            st.subheader("Q1: Relationship between news articles mentioning 'Inflation' and Google Trends for 'Inflation'")
            col1_1, col1_2 = st.columns(2)

            with col1_1:
                # Scatter plot representing correlation
                fig1a = px.scatter(
                    dff1,
                    x="news_count",
                    y="Inflation",
                    color="Year" if y1 == "All Years" else None,
                    title="1a: News Count vs Search Interest",
                    labels={
                        "news_count": clean_label("news_count") + " (articles/month)",
                        "Inflation": clean_label("Inflation") + " (index)"
                    }
                )
                fig1a = add_trendline_and_corr(fig1a, dff1["news_count"], dff1["Inflation"], "news_count", "Inflation")
                fig1a.update_layout(height=430)
                st.plotly_chart(fig1a, use_container_width=True, theme="streamlit")

            with col1_2:
                # Dual-axis line chart for simultaneous timeline tracking
                fig1b = go.Figure()
                fig1b.add_trace(go.Scatter(
                    x=dff1['Date'], y=dff1['news_count'],
                    name="News Count", fill='tozeroy', fillcolor=COLORS["news_fill"], line=dict(color=COLORS["news_line"], width=2)
                ))
                fig1b.add_trace(go.Scatter(
                    x=dff1['Date'], y=dff1['Inflation'],
                    name="Search Interest: Inflation", yaxis='y2', line=dict(color=COLORS["inflation"], width=3)
                ))
                fig1b.update_layout(
                    title="1b: Simultaneous Peaks over Time",
                    yaxis_title="News Count (articles/month)",
                    yaxis2=dict(title="Google Trends Index", overlaying='y', side='right'),
                    height=430
                )
                st.plotly_chart(fig1b, use_container_width=True, theme="streamlit")

            st.success("**Answer:** There is a strong positive relationship between media coverage and public search interest. Graph 1a explicitly calculates the live mathematical correlation: more news reliably equals more searches. Graph 1b illustrates this over time, showing how public search behavior closely follows the volume of media reporting. Media acts as a major driver of public awareness.") 
        
        # QUESTION 2
        with st.container(border=True):
            st.subheader("Q2: Does media coverage influence search interest for related terms ('Energy costs' & 'Cost of living')?")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                # Combined Bar/Line chart mapping news spikes to specific term searches
                fig2a = go.Figure()
                fig2a.add_trace(go.Bar(x=dff1['Date'], y=dff1['news_count'], name="News Articles", marker_color=bg_color))
                fig2a.add_trace(go.Scatter(x=dff1['Date'], y=dff1['Energiekosten'], name="Trends: Energy Costs", yaxis='y2', line=dict(color='#ff7f0e', width=3)))
                fig2a.add_trace(go.Scatter(x=dff1['Date'], y=dff1['Lebenshaltungskosten'], name="Trends: Cost of Living", yaxis='y2', line=dict(color='#2ca02c', width=3)))
                fig2a.update_layout(title="2a: News Volume vs. Related Search Terms", yaxis_title="News Count", yaxis2=dict(title="Google Trends Index", overlaying='y', side='right'))
                st.plotly_chart(fig2a, use_container_width=True, theme="streamlit")
            
            with col2_2:
                # Treemap calculating proportional distribution of the search terms
                trends_sum = dff1[['Inflation', 'Energiekosten', 'Lebenshaltungskosten']].sum().reset_index()
                trends_sum.columns = ['Search Term', 'Total Volume']
                fig2b = px.treemap(
                    trends_sum, 
                    path=['Search Term'], 
                    values='Total Volume', 
                    title="2b: Overall Share of Search Interest (Treemap)", 
                    color='Search Term', 
                    color_discrete_map={'Inflation': '#3b82f6', 'Energiekosten': '#f59e0b', 'Lebenshaltungskosten': '#22c55e'}
                )
                fig2b.update_traces(textinfo="label+value+percent root")
                st.plotly_chart(fig2b, use_container_width=True, theme="streamlit")
                
            st.success("**Answer:** The influence is highly evident. Graph 2a shows that spikes in media coverage perfectly align with surges in searches for specific personal consequences. Graph 2b uses a hierarchical Treemap to show the exact proportions: 'Cost of living' (Lebenshaltungskosten) and general 'Inflation' dominate the overall volume, but specific pain points like 'Energy costs' take a substantial chunk that is primarily driven by these news peaks.")

        # QUESTION 3
        with st.container(border=True):
            st.subheader("Q3: To what extent does media coverage on month t explain variation in search interest on month t+1?")
            col4_1, col4_2 = st.columns(2)
            with col4_1:
                # Heatmap to visualize clusters between t-1 news and t search
                fig3a = px.density_heatmap(dff1, x="news_count_lag1", y="Inflation", nbinsx=15, nbinsy=15,
                                  title="3a: Density Heatmap (Lagged Spillover Effect)",
                                  labels={"news_count_lag1": "News Volume (Month t-1)", "Inflation": "Search Panic (Month t)"},
                                  color_continuous_scale="Oranges")
                st.plotly_chart(fig3a, use_container_width=True, theme="streamlit")
                
            with col4_2:
                # Violin plot showing distribution density shifts based on prior month media levels
                df_violin = dff1.dropna(subset=['prev_month_news_level'])
                if len(df_violin) > 0:
                    fig3b = px.violin(df_violin, x='prev_month_news_level', y='Inflation', color='prev_month_news_level',
                                      box=True, points="all",
                                      title="3b: Search Interest Distribution (Violin Plot)",
                                      labels={'Inflation': 'Search Interest (Month t)', 'prev_month_news_level': 'Prior Month Media Level'})
                    st.plotly_chart(fig3b, use_container_width=True, theme="streamlit")
                else:
                    st.info("Not enough data to construct a Violin Plot for this specific year.")
                
            st.success("**Answer:** Media coverage shows a moderate spillover effect on search interest. Graph 3a suggests a weak positive relationship between prior news volume and subsequent search, though the pattern is dispersed. Graph 3b shows that higher prior media coverage shifts search interest upward, but with considerable variability.")

        # QUESTION 4
        with st.container(border=True):
            st.subheader("Q4: How long does elevated Google search interest persist following major peaks in media coverage?")
            col5_1, col5_2 = st.columns(2)
            with col5_1:
                # Timeline with highlighted vertical rectangle indicating the 'Decay Phase'
                fig4a = go.Figure()
                fig4a.add_trace(go.Scatter(
                    x=dff1['Date'], y=dff1['news_count'], name="News Count: Inflation",
                    fill='tozeroy', fillcolor=COLORS["news_fill"], line=dict(color=COLORS["news_line"], width=2)
                ))
                fig4a.add_trace(go.Scatter(
                    x=dff1['Date'], y=dff1['Inflation'], name="Search Interest: Inflation",
                    yaxis='y2', line=dict(color=COLORS["inflation"], width=3)
                ))
                fig4a.update_layout(
                    title="4a: Decay of Public Interest Timeline",
                    yaxis_title="News Count (articles/month)",
                    yaxis2=dict(title="Google Trends Index", overlaying='y', side='right',range=[0, 100]),
                    height=430, xaxis=dict(title="Date", range=[dff1['Date'].min(), dff1['Date'].max()])
                )

                # Dynamically render the decay rect only if it falls within the selected date range
                if dff1['Date'].min() <= pd.to_datetime('2023-03-01') and dff1['Date'].max() >= pd.to_datetime('2022-11-01'):
                    fig4a.add_vrect(
                        x0="2022-11-01", x1="2023-03-01",
                        fillcolor=COLORS["decay"], opacity=1, line_width=0,
                        annotation_text="Decay Phase", annotation_font_color="red"
                    )
                st.plotly_chart(fig4a, use_container_width=True, theme="streamlit")
            
            with col5_2:
                # Box plot isolating data points from 'Baseline', 'Peak', and 'Decay' phases
                df_phase = dff1[dff1['Phase'] != 'Baseline']
                if len(df_phase) > 0:
                    fig4b = px.box(df_phase, x="Phase", y="Inflation", color="Phase",
                                   points="all", title="4b: Search Interest Distribution (Peak vs Decay)")
                    fig4b.update_traces(boxmean=True)
                    st.plotly_chart(fig4b, use_container_width=True, theme="streamlit")
                else:
                    st.info("No active 'Peak' or 'Decay' phase falls within this selected timeline.")
                    
            st.success("**Answer:** The primary news peak occurred between September and November 2022. As shown in Graph 4a, search interest began to decay quickly after the media peak subsided. Elevated search interest persisted for roughly 3 to 4 months. Graph 4b clearly displays the underlying data points alongside the median and mean, demonstrating that during this decay period, public anxiety drops substantially, showcasing a rapid 'habituation effect'.")

    # --- TAB 2: INFLATION VS TRENDS ---
    with tab2:
        y2 = st.selectbox("📅 Filter Timeline for Tab 2:", available_years, key="y2")
        dff2 = df_main[df_main['Year'] == y2] if y2 != "All Years" else df_main.copy()
        
        # QUESTION 5
        with st.container(border=True):
            st.subheader("Q5: How closely does Google search interest for 'Inflation' track the official monthly inflation rate?")
            col3_1, col3_2 = st.columns(2)
            with col3_1:
                # Overlay hard economic data (official rate) against psychological data (Trends)
                fig5a = go.Figure()
                fig5a.add_trace(go.Scatter(x=dff2['Date'], y=dff2['inflation_rate'], name="Official Inflation Rate (%)", line=dict(color='#d62728', width=4)))
                fig5a.add_trace(go.Scatter(x=dff2['Date'], y=dff2['Inflation'], name="Trends: Inflation", yaxis='y2', line=dict(color='#1f77b4', dash='dot')))
                fig5a.update_layout(title="5a: Trajectories over Time", yaxis_title="Inflation Rate (%)", yaxis2=dict(title="Google Trends", overlaying='y', side='right'))
                st.plotly_chart(fig5a, use_container_width=True, theme="streamlit")
            with col3_2:
                # Bubble chart where bubble size = News volume, showing how media drives the correlation
                df_q5b = dff2.dropna(subset=['inflation_rate', 'Inflation', 'news_count'])
                if len(df_q5b) > 0:
                    fig5b = px.scatter(
                        df_q5b, x="inflation_rate", y="Inflation", color="Year" if y2 == "All Years" else None,
                        size="news_count", title="5b: Official Rate vs Search Interest (Bubble = News Volume)",
                        labels={"inflation_rate": "Inflation Rate (%)", "Inflation": "Google Trends: Inflation (index)", "news_count": "News Count"}
                    )
                    fig5b = add_trendline_and_corr(fig5b, df_q5b["inflation_rate"], df_q5b["Inflation"], "inflation_rate", "Inflation")
                    fig5b.update_layout(height=430)
                    st.plotly_chart(fig5b, use_container_width=True, theme="streamlit")          
            st.success("**Answer:** They track each other exceptionally well over the full timeline. Both metrics peaked synchronously around October 2022. However, Graph 5a shows that during the disinflation phase in 2023, public search interest dropped slightly faster than the actual inflation rate. Graph 5b uses a Bubble Chart to highlight a critical finding: the moments of highest correlation (top right) occur precisely when the media news volume (bubble size) is at its largest.")

        # QUESTION 6
        with st.container(border=True):
            st.subheader("Q6: How does public search interest differ during months with high inflation rates (above 5%)?")
            col7_1, col7_2 = st.columns(2)
            with col7_1:
                # Grouped bar chart validating the 5% psychological threshold
                df_q6 = dff2.groupby('Inflation_Level')[['Inflation', 'Energiekosten']].mean().reset_index()
                if len(df_q6) > 0:
                    fig6a = px.bar(df_q6, x='Inflation_Level', y=['Inflation', 'Energiekosten'], barmode='group',
                                  title="6a: Average Search Interest Comparison")
                    st.plotly_chart(fig6a, use_container_width=True, theme="streamlit")
            with col7_2:
                # Overlaid Histogram showcasing distribution shift above the 5% mark
                fig6b = px.histogram(dff2, x="Inflation", color="Inflation_Level", marginal="box",
                                     title="6b: Distribution of Search Volumes", barmode="overlay", opacity=0.7)
                st.plotly_chart(fig6b, use_container_width=True, theme="streamlit")
            st.success("**Answer:** Graph 6a proves a massive threshold effect: During months with an inflation rate above 5%, the average search interest for 'Inflation' jumps significantly, and searches for 'Energiekosten' nearly triple. Graph 6b confirms this: the entire distribution of search volume shifts drastically to the higher end during high-inflation months, showing that macroeconomic topics completely dominate public consciousness once the 5% mark is crossed.")

    # --- TAB 3: ENERGY & FOOD PRICES ---
    with tab3:
        y3 = st.selectbox("📅 Filter Timeline for Tab 3:", available_years, key="y3")
        dff3 = df_main[df_main['Year'] == y3] if y3 != "All Years" else df_main.copy()
        
        # QUESTION 7
        with st.container(border=True):
            st.subheader("Q7: How do changes in energy prices influence search interest for 'inflation' and 'energy costs'?")
            col6_1, col6_2 = st.columns(2)
            
            search_map = {"Inflation": "Inflation", "Energy Costs": "Energiekosten"}

            with col6_1:
                # Padding hack to align charts horizontally with the selectbox in the right column
                st.markdown("<div style='height: 76px;'></div>", unsafe_allow_html=True)

                fig7a = go.Figure()
                fig7a.add_trace(go.Scatter(x=dff3['Date'], y=dff3['energy_price_index'], name="Energy Price Index", line=dict(color='#d62728', width=3)))
                fig7a.add_trace(go.Scatter(x=dff3['Date'], y=dff3['Inflation'], name="Search: Inflation", yaxis='y2', line=dict(color='#1f77b4', dash='dot')))
                fig7a.add_trace(go.Scatter(x=dff3['Date'], y=dff3['Energiekosten'], name="Search: Energy Costs", yaxis='y2', line=dict(color='#ff7f0e', dash='dot')))
                fig7a.update_layout(title="7a: Energy Index vs Search Trends", yaxis_title="Price Index", yaxis2=dict(title="Google Trends", overlaying='y', side='right'), height=450)
                st.plotly_chart(fig7a, use_container_width=True, theme="streamlit")

            with col6_2:
                # Dynamic user-driven scatter plot with marginal histograms for distribution checks
                selected_label = st.selectbox("Choose search term:", list(search_map.keys()), key="q7b_search")
                selected_column = search_map[selected_label]

                fig7b = px.scatter(
                    dff3, x="energy_price_index", y=selected_column, color_discrete_sequence=['#d62728'],
                    marginal_x="histogram", marginal_y="histogram", title=f"7b: Energy vs {selected_label}",
                    labels={"energy_price_index": "Energy Price Index", selected_column: f"Trends: {selected_label}"}
                )
                fig7b.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), opacity=0.8)
                fig7b.update_layout(height=450)
                st.plotly_chart(fig7b, use_container_width=True, theme="streamlit")
                
            st.success("**Answer:** The plots show that sharp increases in energy prices—especially in late 2022—coincide with strong spikes in search interest for both “inflation” and “energy costs.” However, this relationship is short-lived. After the initial surge, search interest declines significantly even though energy prices remain relatively high. The scatter plots further confirm that there is no strong long-term correlation, indicating that public attention is driven more by sudden price shocks than by sustained energy price levels.")
            
        # QUESTION 8
        with st.container(border=True):
            st.subheader("Q8: How strongly do fluctuations in food prices explain variations in search interest?")
            col8_1, col8_2 = st.columns(2)

            with col8_1:
                search_map_q8 = {"Cost of Living": "Lebenshaltungskosten", "Inflation": "Inflation"}
                selected_label_q8 = st.selectbox("Choose search term:", list(search_map_q8.keys()), key="q8a_search")
                selected_column_q8 = search_map_q8[selected_label_q8]

                # Density contour to detect concentration points (often refutes linear assumptions)
                fig8a = px.density_contour(
                    dff3, x="food_price_index", y=selected_column_q8, title=f"8a: Food Prices vs {selected_label_q8}",
                    labels={"food_price_index": "Food Price Index", selected_column_q8: f"Trends: {selected_label_q8}"}
                )
                fig8a.update_traces(contours_coloring="fill", contours_showlabels=True)
                fig8a.add_trace(go.Scatter(x=dff3["food_price_index"], y=dff3[selected_column_q8], mode="markers", marker=dict(color="white", size=5, line=dict(color="black", width=1)), showlegend=False))
                fig8a.update_layout(height=450)
                st.plotly_chart(fig8a, use_container_width=True, theme="streamlit")

            with col8_2:
                # Visualization of the core "Crisis Fatigue" / "Habituation" divergence
                st.markdown("<div style='height: 76px;'></div>", unsafe_allow_html=True)
                fig8b = go.Figure()
                fig8b.add_trace(go.Scatter(x=dff3['Date'], y=dff3['food_price_index'], name="Food Price Index", fill='tozeroy', marker_color='rgba(44, 160, 44, 0.2)', line=dict(color='#2ca02c')))
                fig8b.add_trace(go.Scatter(x=dff3['Date'], y=dff3['Lebenshaltungskosten'], name="Trends: Cost of Living", yaxis='y2', line=dict(color='#1f77b4', width=3)))
                fig8b.update_layout(title="8b: The Divergence over Time", yaxis_title="Food Price Index", yaxis2=dict(title="Google Trends", overlaying='y', side='right'), height=450)
                st.plotly_chart(fig8b, use_container_width=True, theme="streamlit")
                
            st.success("**Answer:** The plots indicate a weak and inconsistent relationship between food prices and search interest for “cost of living.” While food prices show a steady upward trend, search interest fluctuates without a clear pattern. The density plot shows no strong clustering along a trend line, and the time series highlights a divergence between rising prices and unstable search behavior. Overall, food price changes alone do not strongly explain variations in search interest.")

    # --- TAB 4: MACRO INTERACTIONS ---
    with tab4:
        y4 = st.selectbox("📅 Filter Timeline for Tab 4:", available_years, key="y4")
        dff4 = df_main[df_main['Year'] == y4] if y4 != "All Years" else df_main.copy()
        
        # QUESTION 9
        with st.container(border=True):
            st.subheader("Q9: How do economic indicators interact with media coverage in shaping public attention?")
            
            # --- CORRELATION MATRIX CALCULATIONS ---
            metrics = ['news_count', 'energy_price_index', 'food_price_index', 'unemployment_rate', 'Inflation']
            clean_df = dff4[metrics].dropna()
            clean_df_renamed = clean_df.rename(columns={m: clean_label(m) for m in metrics})
            corr = clean_df_renamed.corr(numeric_only=True)
            
            indicators = [m for m in metrics if m != 'news_count']
            r_vals, p_vals = [], []
            
            # Iteratively calculate Pearson r and strictly capture the p-value for statistical significance checking
            for col in indicators:
                valid_subset = clean_df[['news_count', col]].dropna()
                # Safety check: Pearsonr requires at least 3 points to calculate a meaningful p-value
                if len(valid_subset) > 2:
                    r, p = pearsonr(valid_subset['news_count'], valid_subset[col])
                else:
                    r, p = 0.0, 1.0 
                r_vals.append(r)
                p_vals.append(p)
            
            # Build the results dataframe for the UI
            stats_df = pd.DataFrame({
                'Indicator': [clean_label(i) for i in indicators], 
                'Correlation with Media (r)': r_vals,
                'p_value': p_vals
            })
            # Assess 95% confidence interval (alpha = 0.05)
            stats_df['Significant? (95% Confidence)'] = stats_df['p_value'].apply(lambda x: '✅ Yes (p < 0.05)' if x < 0.05 else '❌ No')
            
            col9_1, col9_2 = st.columns(2)
            
            with col9_1:
                # Render full multi-variable heatmap
                fig9a = px.imshow(corr.round(2), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', 
                                 title="9a: Full Correlation Heatmap (Pearson r)")
                st.plotly_chart(fig9a, use_container_width=True, theme="streamlit")
            
            with col9_2:
                # Render a bar chart isolating the relationship between Media and everything else
                fig9b = px.bar(
                    stats_df, 
                    x='Indicator', y='Correlation with Media (r)', color='Correlation with Media (r)',
                    color_continuous_scale='RdBu_r', title="9b: What drives the Media Cycle?",
                    hover_data={'p_value': ':.4f', 'Significant? (95% Confidence)': True}
                )
                fig9b.update_layout(xaxis_title=None)
                st.plotly_chart(fig9b, use_container_width=True, theme="streamlit")
            
            st.markdown("---")
            st.markdown("#### Exact Statistical Confidence (p-values)")
            
            # Display formatted statistical dataframe table below charts
            st.dataframe(stats_df[['Indicator', 'Correlation with Media (r)', 'p_value', 'Significant? (95% Confidence)']].style.format({
                'Correlation with Media (r)': '{:.2f}',
                'p_value': '{:.4f}'
            }), use_container_width=True, hide_index=True)
                
            st.success("""
            **Answer & Statistical Confidence:** We used the **Pearson correlation algorithm** to measure the linear relationship over the entire timeline. The explicitly calculated p-values reveal a fascinating truth:
            
            1. **Media & Public Panic (Google Trends):** This is the strongest, highly significant relationship. The news cycle directly dictates public search behavior.
            2. **Energy Prices:** Surprisingly, there is **no overall linear correlation** (Not Significant). Why? Because the energy crisis was a sudden, temporary shock in late 2022. Over the entire 3-year timeline, energy prices do not move linearly with the news cycle.
            3. **Food & Unemployment:** Both show a statistically significant **negative** correlation. This represents a clear divergence: while media interest and news volume decayed rapidly after the peak of late 2022, food prices continued to rise relentlessly. The media simply lost interest in the ongoing, slow-moving pain of grocery inflation.
            
            **Conclusion:** The media cycle is not driven by continuous economic reality (like steady food inflation). It is strictly event-driven. It perfectly mirrors public panic, but mathematically decouples from actual, long-term macroeconomic indicators once the initial "shock" wears off.
            """)

# ==========================================
# PAGE 3: PROJECT SUMMARY
# ==========================================
elif page == "🎯 Project Summary":
    st.header("🎯 Final Project Conclusion & Summary")
    with st.container(border=True):
        st.markdown("""
        Based on our comprehensive analysis of the 9 research questions and the exact statistical calculations shown in our Dashboard, we can draw the following core conclusions regarding the German inflation crisis (2022-2024):
        
        ### 1. Media as a key Amplifier (Q1 - Q4)
        Public concern does not simply arise on its own; it is heavily directed by the media. We found a highly significant positive correlation between the volume of news articles and public search interest. When the media heavily reports on inflation, the public immediately researches related topics. Some evidence also points to a short-term spillover effect into subsequent periods.
        
        ### 2. Evidence of a Threshold Effect (Q5 - Q6)
        Public attention closely tracks the official inflation rate. Our data proves a clear **threshold effect**: months where inflation exceeded 5% saw dramatically higher, panic-level search volumes (e.g., searches for energy costs nearly tripled). Below this psychological threshold, macroeconomic topics largely fade from the public's daily consciousness.
        
        ### 3. The Habituation Effect: Prices decoupling from Panic (Q7 - Q8)
        Our most surprising finding is that actual living costs eventually decouple from public panic. While the initial explosion of energy prices in 2022 caused an immediate shock, the long-term correlation to the media cycle is statistically insignificant. Even more striking: Food prices rose continuously throughout 2022 and 2023, yet public searches for "cost of living" dropped off, resulting in a significant negative correlation. This proves a **psychological habituation effect**: people adapt to slow, continuous price pain (groceries), but only panic during sudden, unpredictable media shocks.
        
        ### 4. The Synthesis (Q9)
        Overall, the findings suggest that public attention is more closely aligned with media dynamics and salient economic shocks than with steady underlying trends. Some indicators, such as unemployment, show limited alignment with media coverage and search behavior.
        """)
        
        st.info("💡 **Takeaway:** Economic crises appear to be shaped not only by underlying economic conditions but also by media dynamics and public perception.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Data Science Project 2026</p>", unsafe_allow_html=True)