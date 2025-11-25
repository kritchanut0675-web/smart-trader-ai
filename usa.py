import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from deep_translator import GoogleTranslator
import feedparser
from bs4 import BeautifulSoup
import nltk
import urllib.parse

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- 1. Setup & Design ---
st.set_page_config(
    page_title="Smart Trader AI : Pro Max",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# --- CSS Styling ---
st.markdown("""
    <style>
        /* Main Theme */
        body { background-color: #050505; color: #fff; }
        .stApp { background: radial-gradient(circle at 10% 20%, #000000 0%, #1a1a1a 90%); }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        }

        /* Search Box Styling */
        div[data-testid="stTextInput"] input { 
            border-radius: 12px !important; 
            background-color: rgba(255,255,255,0.1) !important; 
            color: #fff !important; 
            border: 2px solid #00E676 !important; 
            font-size: 1.1rem;
        }

        /* Entry Strategy Cards */
        .entry-card { padding: 20px; border-radius: 16px; margin-bottom: 15px; position: relative; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); }
        .ec-tier1 { background: linear-gradient(135deg, rgba(0, 229, 255, 0.1) 0%, rgba(0,0,0,0) 100%); border-left: 5px solid #00E5FF; }
        .ec-tier2 { background: linear-gradient(135deg, rgba(255, 214, 0, 0.1) 0%, rgba(0,0,0,0) 100%); border-left: 5px solid #FFD600; }
        .ec-tier3 { background: linear-gradient(135deg, rgba(255, 23, 68, 0.15) 0%, rgba(0,0,0,0) 100%); border-left: 5px solid #FF1744; }
        .ec-price { font-size: 1.8rem; font-weight: bold; margin: 10px 0; color: #fff; }
        .ec-title-1 { color: #00E5FF; font-weight: bold; }
        .ec-title-2 { color: #FFD600; font-weight: bold; }
        .ec-title-3 { color: #FF1744; font-weight: bold; }

        /* Sentiment Cards */
        .sentiment-card { padding: 15px; border-radius: 15px; margin-bottom: 15px; background: rgba(255,255,255,0.05); border: 1px solid #333; }
        .badge-pos { background: #00E676; color: #000; padding: 4px 10px; border-radius: 15px; font-weight: bold; font-size: 0.8rem; }
        .badge-neg { background: #FF1744; color: #fff; padding: 4px 10px; border-radius: 15px; font-weight: bold; font-size: 0.8rem; }
        .badge-neu { background: #FFD600; color: #000; padding: 4px 10px; border-radius: 15px; font-weight: bold; font-size: 0.8rem; }

        /* S/R Tags */
        .sr-tag { padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; display: inline-block; margin-left: 10px; }
        .sr-strong { background: rgba(0, 230, 118, 0.2); color: #00E676; border: 1px solid #00E676; }
        .sr-weak { background: rgba(255, 255, 255, 0.1); color: #aaa; border: 1px solid #555; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Data & Analysis Functions ---

@st.cache_data(ttl=1800)
def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df, ticker
    except: return pd.DataFrame(), None

def get_fundamentals_safe(ticker):
    """Safely get fundamentals without crashing"""
    try:
        return ticker.info
    except:
        return {}

def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = [ (df['Open'][0] + df['Close'][0]) / 2 ]
    for i in range(1, len(df)):
        ha_open.append( (ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2 )
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    return ha_df

def identify_levels(df, window=5, tolerance=0.02):
    levels = []
    try:
        for i in range(window, len(df) - window):
            is_support = df['Low'][i] == df['Low'][i-window:i+window+1].min()
            is_resistance = df['High'][i] == df['High'][i-window:i+window+1].max()
            if is_support: levels.append({'price': df['Low'][i], 'type': 'Support', 'touches': 1})
            elif is_resistance: levels.append({'price': df['High'][i], 'type': 'Resistance', 'touches': 1})
        
        levels.sort(key=lambda x: x['price'])
        merged = []
        if not levels: return []
        curr = levels[0]
        for next_lvl in levels[1:]:
            if abs(next_lvl['price'] - curr['price']) / curr['price'] < tolerance:
                curr['price'] = (curr['price'] * curr['touches'] + next_lvl['price'] * next_lvl['touches']) / (curr['touches'] + next_lvl['touches'])
                curr['touches'] += next_lvl['touches']
            else:
                merged.append(curr)
                curr = next_lvl
        merged.append(curr)
        
        final = []
        current_price = df['Close'].iloc[-1]
        for lvl in merged:
            price = lvl['price']
            is_psy = False
            if price > 100: is_psy = (abs(price % 100) < 1) or (abs(price % 1000) < 10)
            
            if lvl['touches'] >= 3 or (lvl['touches'] >= 2 and is_psy): strength, desc = "Strong", "🔥🔥 แข็งแกร่ง"
            else: strength, desc = "Minor", "☁️ เบาบาง"
            
            if abs(price - current_price)/current_price > 0.5 and stren
