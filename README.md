#![Python](https://img.shields.io/badge/python-3.10+-blue.svg) ![Streamlit](https://img.shields.io/badge/streamlit-1.31-red.svg) ![Status](https://img.shields.io/badge/status-final--project-green.svg)

# Media, Public Interest and Inflation in Germany (2022-2024)

## 1. Topic and Research Questions:

This project examines the relationship between media coverage of inflation, public attention, and economic indicators in Germany between 2022 and 2024. The objective is to analyze how news volume, Google search beavior, and official economic data interact. 

The research questions are: 

### Q1. How strong is the relationship between the daily number of German
news articles mentioning “Inflation” and the Google Trends search
index for “Inflation” in Germany?

### Q2. To what extent does increased media coverage of inflation influence
Google search interest for related terms such as “Energy costs” and
“Cost of living” in Germany?

### Q3. How closely does Google search interest for “Inflation” track the
official monthly inflation rate in Germany over time?1. 

### Q4. To what extent does media coverage on day t explain variation in
Google search interest on day t+1?

### Q5. How long does elevated Google search interest persist following major
peaks in media coverage?

### Q6. How do changes in energy prices influence Google search interest for
“inflation” and “energy costs” in Germany during the period
2022–2024?

### Q7. How does public search interest differ during months with high
inflation rates (e.g., above 5)

### Q8. How strongly do fluctuations in food prices explain variations in
Google search interest for “cost of living” in Germany during the
period 2022–2024?

### Q9. How do economic indicators (energy prices, food prices, and
unemployment) interact with media coverage in shaping public
attention to inflation in Germany?

###  Key Findings
* **Media Influence:** Strong correlation between news volume and public search interest.
* **Psychological Threshold:** Public panic surges once inflation crosses the 5% mark.
* **Habituation Effect:** Search interest decays faster than actual prices, indicating "crisis fatigue."
  
## 2. Data 

The project uses data from the following sources: 
- European commission (Eurosat) - inflation rate, energy price index, food   price index, unemployment rate 
- Google trends (pytrends) - search interest for inflation, cost of living and energycost
- GNews - monthly news coverage for inflation

All data is aggregated at a monthly level, with one observation per month from 2022 to 2024. 

## 3. Data pipeline

Acquisition: 
- Data collected via API access 

Processing:
-Data cleaned, aggregated to a monthly level using pandas in jupyter notebooks.

Storage and loading: 
-Processed datasets are stored as CSV files in the repository 

## 4. Implementation/Deployment: 

The website was built using Streamlit as the main framework, with plotly for interactive visualizations and pandas for data handling. The application is implemented in a single file (src\web\web.py). Preprocessed CSV files are stored in the repository (data\processed) and loaded using a central data-loading function. These datasets are merged and prepared when the app starts. The project is deployed via Streamlit community Cloud, which is connected to the GitHub repository and automatically redeploys the app whenever changes are pushed to the main branch. 

## 5. How to use: 

- open the app via the provided link (https://data-science-project-26.streamlit.app/)
- Navigate through the side bar 
- Naviagte through the different tabs 
- Use filters (e.g. year selection) to explore the data 

## 6. LLM usage: 

A Large Langugage Model (Gemini, ChatGPT) was used for support of the follwoing tasks:

Data Visualization Code: Assisting in writing and troubleshooting the Python code for our Plotly graphs and Streamlit dashboard.

Code Structuring: Helping to format and structure our codebase cleanly and professionally.

English Language Assistance: Refining and translating our texts to ensure a natural and professional flow for the final presentation.

## 7. Code Quality
- Functions and variables are named using snake_case (e.g. load_data(), find_and_convert_date())
- Module-level constants use UPPER_SNAKE_CASE (e.g. COLORS, LABELS)
- Lines are kept under 88 characters
- Imports are grouped into standard library, third-party, and local modules
- All non-trivial functions have docstrings explaining their purpose, arguments and return values
- Inline comments are added wherever the logic is not immediately self-evident, for example to explain the lagged spillover calculation or the pearsonr safety guard


## 8. How to Run:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run src/web/web.py
