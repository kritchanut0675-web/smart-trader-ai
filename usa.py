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
import requests
import datetime
import re # นำเข้า module สำหรับจัดการข้อความ (Regex)

# Import translation library
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- CONFIGURATION ---
FINNHUB_KEY = "d4l5ku1r01qt7v18ll40d4l5ku1r01qt7v18ll4g" 

# --- 1. Setup & Design ---
st.set_page_config(
    page_title="Smart Trader AI : Ultra Black",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# --- 2. Ultra Black CSS (Big & Clear) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
        
        /* BLACK BACKGROUND */
        .stApp { background-color: #000000 !important; color: #ffffff; }

        /* INPUT BOX */
        div[data-testid="stTextInput"] input { 
            background-color: #ffffff !important; color: #000000 !important; 
            font-weight: 700 !important; font-size: 1.5rem !important; height: 60px !important;
            border: 3px solid #00E5FF !important; border-radius: 15px !important;
            padding: 10px 20px !important; box-shadow: 0 0 15px rgba(0, 229, 255, 0.5);
        }

        /* CARDS */
        .glass-card {
            background: rgba(20, 20, 20, 0.6); backdrop-filter: blur(10px);
            border-radius: 25px; border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 35px; margin-bottom: 30px; box-shadow: 0 0 20px rgba(255, 255, 255, 0.05);
        }
        
        /* BIG S/R TABLE */
        .sr-container { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
        .sr-row { display: flex; justify-content: space-between; align-items: center; padding: 15px 25px; border-radius: 15px; font-size: 1.5rem; font-weight: bold; }
        .res-row { background: linear-gradient(90deg, rgba(255, 23, 68, 0.1), rgba(0,0,0,0)); border-left: 8px solid #FF1744; color: #FF1744; }
        .sup-row { background: linear-gradient(90deg, rgba(0, 230, 118, 0.1), rgba(0,0,0,0)); border-left: 8px solid #00E676; color: #00E676; }
        .curr-row { background: #222; border: 1px solid #555; color: #fff; justify-content: center; font-size: 1.8rem; text-shadow: 0 0 10px white; }

        /* STATS BOX */
        .stat-box { background: #0a0a0a; border-radius: 20px; padding: 25px; text-align: center; border: 1px solid #333; margin-bottom: 15px; }
        .stat-label { color: #888; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px; }
        .stat-value { font-size: 2.5rem; font-weight: 800; color: #fff; }

        /* FUNDAMENTAL BOX */
        .fund-box { 
            background: #111; border: 1px solid #444; border-radius: 15px; padding: 20px; 
            margin-bottom: 10px; position: relative; overflow: hidden;
        }
        .fund-title { font-size: 1rem; color: #aaa; margin-bottom: 5px; text-transform: uppercase; }
        .fund-val { font-size: 1.8rem; font-weight: bold; color: #fff; }
        .fund-desc { font-size: 0.9rem; color: #888; margin-top: 5px; }

        /* NEWS CARD (UPDATED) */
        .news-card { 
            padding: 25px; margin-bottom: 20px; 
            background: #111; border-radius: 15px; 
            border-left: 6px solid #888; 
            transition: transform 0.2s; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .news-card:hover { transform: translateY(-2px); background: #161616; }
        .nc-pos { border-left-color: #00E676; }
        .nc-neg { border-left-color: #FF1744; }
        .nc-neu { border-left-color: #FFD600; }
        
        .news-summary {
            font-size: 1rem; color: #ccc; margin-top: 10px; line-height: 1.6;
            background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;
        }

        /* AI VERDICT CARD */
        .ai-card {
            background: linear-gradient(145deg, #111, #0d0d0d); border: 2px solid #00E5FF;
            border-radius: 20px; padding: 30px; position: relative; box-shadow: 0 0 30px rgba(0, 229, 255, 0.1);
        }
        .ai-score-circle {
            width: 100px; height: 100px; border-radius: 50%; border: 5px solid #00E5FF;
            display: flex; align-items: center; justify-content: center;
            font-size: 2.5rem; font-weight: bold; color: #00E5FF; margin: 0 auto 20px auto;
        }

        /* BUTTONS */
        div.stButton > button {
            font-size: 1.1rem !important; padding: 15px !important; border-radius: 15px !important;
            background: #111; border: 1px solid #333; color: #fff;
        }
        div.stButton > button:hover { background: #00E5FF; color: #000 !important; font-weight: bold; }
        
        button[data-baseweb="tab"] { font-size: 1.1rem !important; font-weight: 600 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. Data & Analysis Functions ---

@st.cache_data(ttl=300)
def get_market_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info
    except: return None

@st.cache_data(ttl=15)
def get_bitkub_ticker():
    try:
        url = "https://api.bitkub.com/api/market/ticker"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except: return None

# --- NEW: Helper to clean HTML tags ---
def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def get_finnhub_news(symbol):
    try:
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=2)
        clean_sym = symbol.split("-")[0]
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_sym}&from={yesterday}&to={today}&token={FINNHUB_KEY}"
        r = requests.get(url)
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[:5]
        return []
    except: return []

# --- UPDATED: News Analysis with Summary Translation ---
@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    news_list = []
    translator = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    
    # 1. Finnhub (US/Crypto)
    finnhub_news = get_finnhub_news(symbol)
    if finnhub_news:
        for item in finnhub_news:
            title = item.get('headline', '')
            summary = item.get('summary', '') # Finnhub has summary
            link = item.get('url', '#')
            
            blob = TextBlob(title)
            score = blob.sentiment.polarity
            if score > 0.05: sentiment, color, icon = "ข่าวดี (Positive)", "nc-pos", "🚀"
            elif score < -0.05: sentiment, color, icon = "ข่าวร้าย (Negative)", "nc-neg", "🔻"
            else: sentiment, color, icon = "ทั่วไป (Neutral)", "nc-neu", "⚖️"
            
            # Translate Title & Summary
            title_th = title
            summary_th = summary
            if translator:
                try: 
                    title_th = translator.translate(title)
                    if summary:
                        summary_th = translator.translate(summary)
                except: pass
                
            news_list.append({
                'title_th': title_th, 'summary_th': summary_th, 'link': link, 
                'sentiment': sentiment, 'class': color, 'icon': icon, 'score': score, 'source': 'Finnhub'
            })
            
    # 2. Google RSS (Backup/Thai Stocks)
    if len(news_list) < 3:
        try:
            clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
            q = urllib.parse.quote(f"site:bloomberg.com {clean_sym} market")
            rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            if len(feed.entries) == 0:
                q = urllib.parse.quote(f"{clean_sym} finance news")
                rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(rss_url)

            for item in feed.entries[:5]:
                title = item.title
                # Extract Summary from RSS description
                raw_summary = getattr(item, 'summary', '') or getattr(item, 'description', '')
                summary = clean_html(raw_summary)[:300] + "..." if len(raw_summary) > 300 else clean_html(raw_summary)

                blob = TextBlob(title)
                score = blob.sentiment.polarity
                
                if score > 0.05: sentiment, color, icon = "ข่าวดี (Positive)", "nc-pos", "🚀"
                elif score < -0.05: sentiment, color, icon = "ข่าวร้าย (Negative)", "nc-neg", "🔻"
                else: sentiment, color, icon = "ทั่วไป (Neutral)", "nc-neu", "⚖️"

                # Translate Title & Summary
                title_th = title
                summary_th = summary
                if translator:
                    try: 
                        title_th = translator.translate(title)
                        if summary:
                            summary_th = translator.translate(summary)
                    except: pass

                news_list.append({
                    'title_th': title_th, 'summary_th': summary_th, 'link': item.link, 
                    'sentiment': sentiment, 'class': color, 'icon': icon, 'score': score, 'source': 'Google News'
                })
        except: pass
        
    return news_list[:10]

# --- S/R & AI Logic (Same as before) ---
def generate_dynamic_insight(price, pivots, dynamics):
    ema200 = dynamics['EMA 200']
    ema20 = dynamics['EMA 20']
    
    if price > ema200:
        if price > ema20: trend_msg, trend_color = "Bullish Strong (แกร่งมาก)", "#00E676"
        else: trend_msg, trend_color = "Bullish Retrace (ย่อตัวในขาขึ้น)", "#00E676"
    else:
        if price < ema20: trend_msg, trend_color = "Bearish Strong (ลงหนัก)", "#FF1744"
        else: trend_msg, trend_color = "Bearish Correction (ดีดตัวในขาลง)", "#FF1744"

    all_levels = {**pivots, **{k:v for k,v in dynamics.items() if k != 'Current'}}
    nearest_name = ""
    nearest_price = 0
    min_dist = float('inf')
    
    for name, lvl_price in all_levels.items():
        dist = abs(price - lvl_price)
        if dist < min_dist:
            min_dist = dist
            nearest_name = name
            nearest_price = lvl_price
            
    dist_pct = (min_dist / price) * 100
    if dist_pct < 0.8: action_msg = f"⚠️ ราคากำลังทดสอบแนวสำคัญ **{nearest_name}** ({nearest_price:,.2f})"
    else: action_msg = f"🏃 ราคามีพื้นที่วิ่ง (Room to run) ไปหา **{nearest_name}** ({nearest_price:,.2f})"

    return trend_msg, trend_color, action_msg

def calculate_bitkub_ai_levels(high24, low24, last_price):
    pp = (high24 + low24 + last_price) / 3
    rng = high24 - low24
    r1 = (2 * pp) - low24
    s1 = (2 * pp) - high24
    r2 = pp + rng
    s2 = pp - rng
    fib_high = low24 + (rng * 0.618)
    fib_low = low24 + (rng * 0.382)
    
    mid_point = (high24 + low24) / 2
    if last_price > mid_point: status, status_color = "BULLISH (กระทิง)", "#00E676"
    else: status, status_color = "BEARISH (หมี)", "#FF1744"
        
    return {
        "levels": [
            {"name": "🚀 R2 (Breakout)", "price": r2, "type": "res"},
            {"name": "🛑 R1 (Resist)", "price": r1, "type": "res"},
            {"name": "⚖️ PIVOT (จุดหมุน)", "price": pp, "type": "neu"},
            {"name": "🛡️ S1 (Support)", "price": s1, "type": "sup"},
            {"name": "💎 S2 (Bottom)", "price": s2, "type": "sup"}
        ],
        "fib": {"top": fib_high, "bot": fib_low},
        "status": status, "color": status_color
    }

def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = [ (df['Open'][0] + df['Close'][0]) / 2 ]
    for i in range(1, len(df)): ha_open.append( (ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2 )
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    return ha_df

def calculate_technical_setup(df):
    try:
        close = df['Close'].iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0,0).rolling(14).mean() / abs(df['Close'].diff().where(lambda x: x<0,0)).rolling(14).mean()))).iloc[-1]

        if close > ema50 and ema50 > ema200
