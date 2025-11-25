import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import feedparser
import nltk
import urllib.parse

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- 1. Setup & Design ---
st.set_page_config(
    page_title="Smart Trader AI : Premium",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD' # เปลี่ยน Default เป็น USD

def set_symbol(sym):
    st.session_state.symbol = sym

# --- 2. Premium CSS & UI Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
        
        /* Global Font */
        html, body, [class*="css"] {
            font-family: 'Kanit', sans-serif;
        }

        /* Background */
        .stApp {
            background: radial-gradient(circle at top center, #1a1a2e 0%, #000000 100%);
            color: #fff;
        }

        /* --- 🎯 Fix Input Box (Black Text on White) --- */
        div[data-testid="stTextInput"] input { 
            background-color: #ffffff !important; 
            color: #000000 !important; 
            font-weight: 600 !important;
            border: 2px solid #00E5FF !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }
        div[data-testid="stTextInput"] label {
            color: #00E5FF !important;
            font-size: 1.1rem !important;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* Sidebar Buttons */
        div.stButton > button {
            width: 100%;
            background: transparent;
            border: 1px solid #333;
            color: #aaa;
            border-radius: 12px;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            border-color: #00E5FF;
            color: #00E5FF;
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
        }

        /* Status Badges */
        .status-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
            display: inline-block;
        }
        .badge-up { background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid #00E676; }
        .badge-down { background: rgba(255, 23, 68, 0.15); color: #FF1744; border: 1px solid #FF1744; }

        /* Entry Strategy Cards */
        .entry-box {
            background: linear-gradient(145deg, #111, #161616);
            border-radius: 15px;
            padding: 20px;
            border-left: 5px solid #555;
            margin-bottom: 15px;
        }
        .eb-1 { border-left-color: #00E5FF; }
        .eb-2 { border-left-color: #FFD600; }
        .eb-3 { border-left-color: #FF1744; }

    </style>
""", unsafe_allow_html=True)

# --- 3. Data Functions (Robust) ---

@st.cache_data(ttl=300) # ลดเวลา Cache ลงเหลือ 5 นาทีเพื่อให้ราคาอัปเดตไวขึ้น
def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        # Force download to ensure fresh data
        df = ticker.history(period=period, interval=interval)
        return df
    except: return pd.DataFrame()

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

def calculate_trade_setup(df):
    try:
        close = df['Close'].iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        
        # ATR Calculation
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        if close > ema50 and ema50 > ema200:
            trend = "Uptrend 🟢"
            signal = "BUY / LONG"
            color = "#00E676"
            sl = close - (1.5 * atr)
            tp = close + (2.5 * atr)
        elif close < ema50 and ema50 < ema200:
            trend = "Downtrend 🔴"
            signal = "SELL / SHORT"
            color = "#FF1744"
            sl = close + (1.5 * atr)
            tp = close - (2.5 * atr)
        else:
            trend = "Sideways 🟡"
            signal = "WAIT"
            color = "#888"
            sl = close - atr
            tp = close + atr
            
        return {'trend': trend, 'signal': signal, 'color': color, 'entry': close, 'sl': sl, 'tp': tp, 'atr': atr}
    except: return None

# --- Bloomberg News (Google RSS) ---
@st.cache_data(ttl=3600)
def get_news(symbol):
    news_list = []
    clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
    try:
        q = urllib.parse.quote(f"site:bloomberg.com {clean_sym}")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for item in feed.entries[:4]:
             news_list.append({'title': item.title, 'link': item.link, 'summary': item.description, 'source': 'Bloomberg'})
    except: pass
    return news_list

# --- 4. Sidebar ---
with
