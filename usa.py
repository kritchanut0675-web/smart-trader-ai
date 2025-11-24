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
from newspaper import Article, Config
import nltk

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- 1. ตั้งค่าหน้าเว็บ & Session State ---
st.set_page_config(
    page_title="Smart Trader AI : Assistant",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# CSS Styling
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        
        /* Input & Button */
        div[data-testid="stTextInput"] input {
            font-size: 20px !important; height: 50px !important;
            border-radius: 12px !important; background-color: #1b1b1b !important;
            color: #fff !important; border: 1px solid #333 !important;
        }
        div[data-testid="stButton"] button {
            height: 50px !important; font-size: 20px !important;
            border-radius: 12px !important; width: 100% !important;
            background-color: #2962FF !important; color: white !important;
            border: none !important; font-weight: bold !important;
        }
        
        /* Stats Cards */
        .stat-card {
            background-color: #1E1E1E; padding: 15px; border-radius: 10px;
            text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            height: 100%; transition: transform 0.2s;
        }
        .stat-card:hover { transform: translateY(-3px); }
        .stat-label { font-size: 0.85rem; color: #aaa; margin-bottom: 5px; text-transform: uppercase; }
        .stat-value { font-size: 1.3rem; font-weight: bold; }
        
        .high-card { border-top: 3px solid #00E5FF; } .high-val { color: #00E5FF; }
        .low-card { border-top: 3px solid #FF4081; } .low-val { color: #FF4081; }
        .beta-card { border-top: 3px solid #E040FB; } .beta-val { color: #E040FB; }
        .div-card { border-top: 3px solid #00E676; } .div-val { color: #00E676; }

        /* Guru & News & AI Report */
        .guru-card {
            background: linear-gradient(135deg, #1a237e 0%, #000000 100%);
            padding: 20px; border-radius: 15px; border: 1px solid #304FFE;
            margin-bottom: 20px; box-shadow: 0 4px 15px rgba(48, 79, 254, 0.3);
        }
        .ai-report-box {
            background: #111; border-left: 5px solid #00E676; padding: 20px;
            border-radius: 10px; line-height: 1.8; font-size: 1.05rem; color: #e0e0e0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        .news-content { font-size: 1rem; line-height: 1.7; color: #ddd; background: #1a1a1a; padding: 15px; border-radius: 10px; }
        
        button[data-baseweb="tab"] { font-size: 1.1rem !important; padding: 15px !important; flex: 1; }
        .stButton button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Functions ---

def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty and symbol.endswith("-THB"):
            base = symbol.replace("-THB", "-USD")
            df = yf.Ticker(base).history(period=period, interval=interval)
            usd = yf.Ticker("THB=X").history(period="1d")['Close'].iloc[-1]
            if not df.empty: df[['Open','High','Low','Close']] *= usd
        return df, ticker
    except: return pd.DataFrame(), None

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

def analyze_ai_signal(df):
    close = df['Close'].iloc[-1]
    ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
    rsi = df['RSI'].iloc[-1]
    if close > ema200:
        if rsi < 30: return "🟢 เข้าซื้อ (Strong Buy)", "#00E676", "เทรนด์ขาขึ้น + ย่อตัวหนัก"
        elif rsi < 50: return "🟢 ทยอยสะสม (Buy)", "#66BB6A", "เทรนด์ขาขึ้น ราคายังไม่แพง"
        elif rsi > 70: return "🔴 ระวังแรงเทขาย", "#FF1744", "ราคา Overbought สูงเกินไป"
        else: return "🟡 ถือรันเทรนด์", "#FFD600", "แนวโน้มยังดี ถือต่อได้"
    else:
        if rsi > 70: return "🔴 ขาย/Short", "#D50000", "เทรนด์ขาลง + ราคาดีดสูงเกินไป"
        else: return "🟠 เลี่ยงการเทรด", "#FF9100", "ราคายังอยู่ใต้เส้นค่าเฉลี่ย 200 วัน"

def analyze_levels(df):
    levels = []
    for i in range(2, df.shape[0]-2):
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1]:
            levels.append({'p': df['Low'][i], 't': 'Support'})
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1]:
            levels.append({'p': df['High'][i], 't': 'Resistance'})
    levels.sort(key=lambda x: x['p'])
    clusters = []
    threshold = df['Close'].mean() * 0.015
    for l in levels:
        if not clusters: clusters.append({'p': l['p'], 'c': 1, 't': l['t']}); continue
        if abs(l['p'] - clusters[-1]['p']) < threshold:
            clusters[-1]['c'] += 1
            clusters[-1]['p'] = (clusters[-1]['p'] * (clusters[-1]['c']-1) + l['p']) / clusters[-1]['c']
        else: clusters.append({'p': l['p'], 'c': 1, 't': l['t']})
    results = []
    for c in clusters:
        label = "แข็งแกร่ง 🔥" if c['c'] >= 3 else "ปกติ"
        results.append({'price': c['p'], 'type': c['t'], 'label': label, 'score': c['c']})
    return results

def get_guru_insight(ticker, price):
    try:
        info = ticker.info
        name = info.get('longName', 'Unknown')
        sector = info.get('sector', '-')
        target = info.get('targetMeanPrice', 0)
        rec = info.get('recommendationKey', '-').upper().replace('_', ' ')
        pe = info.get('trailingPE', 0)
        beta = info.get('beta', 0)
        div_yield = info.get('dividendYield', 0)
        high52 = info.get('fiftyTwoWeekHigh', 0)
        low52 = info.get('fiftyTwoWeekLow', 0)

        insight = f"**{name}** ({sector})\n\n"
        if target and target > 0:
            upside = ((target - price) / price) * 100
            if upside > 0: insight += f"🎯 **Target:** Upside **{upside:.1f}%** (เป้า {target:,.2f}) "
            else: insight += f"⚠️ **Target:** Overvalued (เป้า {target:,.2f}) "
        
        if pe > 0:
            if pe < 15: insight += f"💎 **P/E:** ต่ำ ({pe:.2f}) Value Stock"
            elif pe > 50: insight += f"🚀 **P/E:** สูง ({pe:.2f}) Growth Stock"
        
        return insight, rec, target, pe, beta, div_yield, high52, low52
    except: return "No Data", "-", 0, 0, 0, 0, 0, 0

# --- News ---
@st.cache_data(ttl=3600) 
def fetch_content(url, backup=""):
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 10
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text
        if len(text) < 150: return backup if backup else "เนื้อหาถูกจำกัดสิทธิ์"
        return text[:4000]
    except: return backup if backup else "ไม่สามารถดึงเนื้อห
