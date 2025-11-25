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

# --- Config & Setup ---
st.set_page_config(
    page_title="Smart Trader AI : Pro Max",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# CSS Design
st.markdown("""
    <style>
        body { background-color: #050505; color: #fff; }
        .stApp { background: radial-gradient(circle at 10% 20%, #000000 0%, #1a1a1a 90%); }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            margin-bottom: 15px;
        }
        .big-font { font-size: 20px !important; font-weight: bold; }
        .search-box input { 
            font-size: 1.2rem; 
            padding: 10px; 
            border: 2px solid #00E5FF !important; 
            border-radius: 10px !important; 
        }
        /* Button Styling */
        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: bold;
            border: 1px solid #333;
        }
        div.stButton > button:hover {
            border-color: #00E5FF;
            color: #00E5FF;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# --- 1. Robust Data Fetching (Anti-Crash) ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent Rate Limit
def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamentals(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info
    except:
        return {} # Return empty dict if fails (avoids crash)

# --- 2. Technical Analysis Functions ---
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

def identify_levels(df):
    # Simplified logic for brevity, robust against empty data
    if df.empty: return []
    levels = []
    # Simple logic: Pivot points or local min/max could go here
    # Using simple local min/max for demo
    window = 5
    for i in range(window, len(df) - window):
        if df['Low'][i] == df['Low'][i-window:i+window+1].min():
            levels.append({'price': df['Low'][i], 'type': 'Support', 'strength': 'Minor', 'desc': 'Support'})
        elif df['High'][i] == df['High'][i-window:i+window+1].max():
            levels.append({'price': df['High'][i], 'type': 'Resistance', 'strength': 'Minor', 'desc': 'Resistance'})
    return levels[-5:] # Return last 5 levels

def calculate_trade_setup(df):
    if df.empty: return None
    close = df['Close'].iloc[-1]
    ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
    
    # ATR
    tr = np.max([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=0)
    atr = pd.Series(tr).rolling(14).mean().iloc[-1]
    
    if close > ema50:
        return {'signal': 'BUY / LONG', 'color': '#00E676', 'entry': close, 'sl': close - 1.5*atr, 'tp': close + 2*atr, 'atr': atr, 'trend': 'Uptrend'}
    else:
        return {'signal': 'SELL / SHORT', 'color': '#FF1744', 'entry': close, 'sl': close + 1.5*atr, 'tp': close - 2*atr, 'atr': atr, 'trend': 'Downtrend'}

# --- 3. Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=60)
    st.title("Smart Trader AI")
    st.caption("Developed by: KRITCHANUT VONGRAT")
    
    st.markdown("### 🚀 Quick Select (เลือกเหรียญด่วน)")
    
    st.markdown("**🇹🇭 Thai Crypto (กระดานไทย)**")
    c1, c2 = st.columns(2)
    if c1.button("BTC-THB"): set_symbol("BTC-THB")
    if c2.button("ETH-THB"): set_symbol("ETH-THB")
    
    st.markdown("**🌎 Global Crypto**")
    c3, c4 = st.columns(2)
    if c3.button("BTC-USD"): set_symbol("BTC-USD")
    if c4.button("ETH-USD"): set_symbol("ETH-USD")
    
    st.markdown("**📈 Others**")
    c5, c6 = st.columns(2)
    if c5.button("Gold (XAU)"): set_symbol("GC=F")
    if c6.button("Oil (WTI)"): set_symbol("CL=F")

    st.markdown("---")
    st.subheader("⚙️ Settings")
    chart_type = st.selectbox("Chart Type", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Period", ["1mo", "3mo", "6mo", "1y", "5y"], value="1y")

# --- 4. Main Search & Header ---
st.markdown("### 🔍 ค้นหาหุ้น / เหรียญ (Smart Search)")

# Search Layout
col_search, col_btn = st.columns([4, 1])
with col_search:
    # Use a clear text input
    sym_input = st.text_input(
        "ระบุชื่อย่อ (Ticker)", 
        value=st.session_state.symbol,
        placeholder="เช่น BTC-USD, OR.BK, DELTA.BK, PTT.BK, TSLA",
        label_visibility="collapsed"
    )
with col_btn:
    if st.button("วิเคราะห์ทันที ⚡", use_container_width=True):
        st.session_state.symbol = sym_input
        st.rerun()

# Instruction hints
st.caption("💡 **Tip:** หุ้นไทยให้เติม `.BK` (เช่น `PTT.BK`, `KBANK.BK`) | คริปโตไทยใช้ `-THB` (เช่น `BTC-THB`)")

# --- 5. Processing & Display ---
symbol = st.session_state.symbol.upper()
interval = "1d"

if symbol:
    # Fetch Data
    with st.spinner(f'กำลังดึงข้อมูล {symbol} (Cache enabled)...'):
        df = get_data(symbol, period, interval)
        info = get_fundamentals(symbol) # Won't crash if fails
    
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล '{symbol}' หรือ API ถูกจำกัดการเข้าถึงชั่วคราว กรุณาลองใหม่ในภายหลัง")
    else:
        # Calculate Indicators
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = curr_price - prev_price
        pct = (change / prev_price) * 100
        
        # Header Display
        color = '#00E676' if change >= 0 else '#FF1744'
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-left: 10px solid {color};">
                <h1 style="margin:0;">{symbol}</h1>
                <h2 style="margin:0; font-size: 3rem; color: {color};">{curr_price:,.2f}</h2>
                <p style="font-size: 1.2rem; color: #aaa;">{change:+,.2f} ({pct:+.2f}%)</p>
                <p style="font-size: 0.9rem; color: #666;">{info.get('longName', 'Unknown Name')}</p>
            </div>
        """, unsafe_allow_html=True)

        # Tabs
        tabs = st.tabs(["📈 Chart", "🎯 AI Setup", "📊 Stats"])
        
        # Tab 1: Chart
        with tabs[0]:
            fig = go.Figure()
            if chart_type == "Heikin Ashi":
                plot_df = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=plot_df['HA_Open'], high=plot_df['HA_High'], 
                                             low=plot_df['HA_Low'], close=plot_df['HA_Close'], name="HA"))
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                             low=df['Low'], close=df['Close'], name="Price"))
                
            fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Trade Setup
        with tabs[1]:
            setup = calculate_trade_setup(df)
            if setup:
                st.markdown(f"""
                    <div style="display: flex; gap: 10px;">
                        <div class="glass-card" style="flex:1; text-align:center;">
                            <h3>Signal</h3>
                            <h2 style="color:{setup['color']}">{setup['signal']}</h2>
                        </div>
                         <div class="glass-card" style="flex:1; text-align:center;">
                            <h3>Trend</h3>
                            <h2>{setup['trend']}</h2>
                        </div>
                    </div>
                    <div style="display: flex; gap: 10px; margin-top: 10px;">
                        <div class="glass-card" style="flex:1; text-align:center; border: 1px solid #2979FF;">
                            <small>ENTRY</small>
                            <div class="big-font" style="color:#2979FF">{setup['entry']:,.2f}</div>
                        </div>
                        <div class="glass-card" style="flex:1; text-align:center; border: 1px solid #FF1744;">
                            <small>STOP LOSS</small>
                            <div class="big-font" style="color:#FF1744">{setup['sl']:,.2f}</div>
                        </div>
                        <div class="glass-card" style="flex:1; text-align:center; border: 1px solid #00E676;">
                            <small>TAKE PROFIT</small>
                            <div class="big-font" style="color:#00E676">{setup['tp']:,.2f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Tab 3: Stats/Fundamentals
        with tabs[2]:
            # Safe display that won't crash if 'info' is empty
            if info:
                c1, c2, c3 = st.columns(3)
                c1.metric("Market Cap", f"{info.get('marketCap', 0):,}")
                c2.metric("52 Week High", f"{info.get('fiftyTwoWeekHigh', 0):,.2f}")
                c3.metric("52 Week Low", f"{info.get('fiftyTwoWeekLow', 0):,.2f}")
                st.write(f"**Business Summary:** {info.get('longBusinessSummary', 'No description available')[:500]}...")
            else:
                st.warning("⚠️ ไม่สามารถดึงข้อมูลพื้นฐาน (Fundamentals) ได้ในขณะนี้เนื่องจากข้อจำกัดของ API (Rate Limit) แต่กราฟราคายังทำงานได้ปกติ")
