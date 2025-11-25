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
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00E5FF;'>💎 SMART AI</h2>", unsafe_allow_html=True)
    st.caption("Premium Edition by KRITCHANUT")
    st.markdown("---")
    
    # เอาปุ่ม BTC-THB, ETH-THB ออกตามสั่ง
    st.markdown("### 🌎 Global Assets")
    c1, c2 = st.columns(2)
    if c1.button("BTC-USD"): set_symbol("BTC-USD")
    if c2.button("ETH-USD"): set_symbol("ETH-USD")
    
    st.markdown("### 🌟 Popular")
    c3, c4 = st.columns(2)
    if c3.button("Gold (XAU)"): set_symbol("GC=F")
    if c4.button("Oil (WTI)"): set_symbol("CL=F")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    chart_type = st.selectbox("Chart Style", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Period", options=["1mo", "3mo", "6mo", "1y", "5y"], value="1y")

# --- 5. Main Content ---

# Header Section
st.markdown("<h3 style='margin-bottom: 5px;'>🔍 ค้นหาเหรียญ / หุ้น (Premium Search)</h3>", unsafe_allow_html=True)

c_search, c_btn = st.columns([4, 1])
with c_search:
    # Input box is styled via CSS to be White with Black text
    sym_input = st.text_input("ระบุชื่อย่อ (เช่น BTC-USD, AAPL)", value=st.session_state.symbol, label_visibility="collapsed")
with c_btn:
    if st.button("วิเคราะห์ ⚡", use_container_width=True):
        st.session_state.symbol = sym_input
        st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner('💎 AI กำลังวิเคราะห์ข้อมูลระดับสูง...'):
        df = get_data(symbol, period, "1d")
        
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล '{symbol}' กรุณาลองตรวจสอบชื่อย่อใหม่")
    else:
        # Calculation
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = curr_price - prev_price
        pct = (change / prev_price) * 100
        last_date = df.index[-1].strftime('%d-%m-%Y')
        
        setup = calculate_trade_setup(df)

        # --- HERO SECTION (Beautiful Header) ---
        color_trend = "#00E676" if change >= 0 else "#FF1744"
        arrow = "▲" if change >= 0 else "▼"
        
        st.markdown(f"""
        <div class="glass-card" style="border-top: 5px solid {color_trend}; text-align: center;">
            <div style="font-size: 1.2rem; color: #aaa; letter-spacing: 2px; text-transform: uppercase;">ASSET ANALYSIS</div>
            <div style="font-size: 4rem; font-weight: 800; margin: 10px 0; background: -webkit-linear-gradient(45deg, #fff, {color_trend}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {symbol}
            </div>
            <div style="font-size: 3.5rem; font-weight: bold; color: {color_trend};">
                {curr_price:,.2f} 
            </div>
            <div style="color: #888; margin-bottom: 10px;">Price Date: {last_date}</div>
            <div class="status-badge {'badge-up' if change >= 0 else 'badge-down'}" style="font-size: 1.2rem;">
                {arrow} {abs(change):,.2f} ({pct:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- TABS ---
        tabs = st.tabs(["📈 Smart Chart", "🎯 AI Setup", "💰 Entry Strategy", "🧠 Sentiment", "📊 Stats"])

        # 1. CHART
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
            
            # Choose Chart Type
            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['HA_Open'], high=ha['HA_High'], low=ha['HA_Low'], close=ha['HA_Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # EMA
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF', width=1.5), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=200).mean(), line=dict(color='#FF9100', width=1.5), name='EMA 200'), row=1, col=1)
            
            # RSI
            rsi = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0,0).rolling(14).mean() / abs(df['Close'].diff().where(lambda x: x<0,0)).rolling(14).mean())))
            fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#E040FB'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            
            fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # 2. SETUP
        with tabs[1]:
            if setup:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <div style="color: #aaa;">SIGNAL</div>
                        <div style="font-size: 2rem; font-weight: bold; color: {setup['color']};">{setup['signal']}</div>
                        <hr style="border-color: #333;">
                        <div style="color: #aaa;">TREND</div>
                        <div style="font-size: 1.2rem;">{setup['trend']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="display: flex; gap: 10px; height: 100%;">
                        <div class="glass-card" style="flex: 1; text-align: center; border: 1px solid #FF1744;">
                            <div style="color: #FF1744; font-weight: bold;">STOP LOSS</div>
                            <div style="font-size: 1.8rem;">{setup['sl']:,.2f}</div>
                        </div>
                        <div class="glass-card" style="flex: 1; text-align: center; border: 1px solid #00E676;">
                            <div style="color: #00E676; font-weight: bold;">TAKE PROFIT</div>
                            <div style="font-size: 1.8rem;">{setup['tp']:,.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # 3. ENTRY STRATEGY
        with tabs[2]:
            st.markdown("### 💰 Smart Money Management")
            
            # Simple tiered logic based on current price & ATR
            t1 = curr_price * 0.995 # ย่อเล็กน้อย
            t2 = curr_price * 0.98  # ย่อลึก
            t3 = curr_price * 0.95  # จุด Panic
            
            st.markdown(f"""
            <div class="entry-box eb-1">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#00E5FF; font-weight:bold; font-size:1.2rem;">🔹 ไม้ที่ 1 : Probe Buy (20%)</span>
                    <span style="font-weight:bold; font-size:1.2rem;">{t1:,.2f}</span>
                </div>
                <small style="color:#aaa;">เข้าซื้อเพื่อทดสอบตลาด หรือเกาะแนวโน้มระยะสั้น</small>
            </div>
            
            <div class="entry-box eb-2">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#FFD600; font-weight:bold; font-size:1.2rem;">🔸 ไม้ที่ 2 : Accumulate (30%)</span>
                    <span style="font-weight:bold; font-size:1.2rem;">{t2:,.2f}</span>
                </div>
                <small style="color:#aaa;">จุดเข้าซื้อเพิ่มเมื่อราคาย่อตัวลงมา (Dip Buying)</small>
            </div>
            
            <div class="entry-box eb-3">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#FF1744; font-weight:bold; font-size:1.2rem;">🔻 ไม้ที่ 3 : Sniper Zone (50%)</span>
                    <span style="font-weight:bold; font-size:1.2rem;">{t3:,.2f}</span>
                </div>
                <small style="color:#aaa;">จุดซื้อของถูกเมื่อเกิด Panic Sell หรือแนวรับสำคัญมาก</small>
            </div>
            """, unsafe_allow_html=True)

        # 4. SENTIMENT
        with tabs[3]:
            st.markdown("### 📰 Bloomberg & Global News")
            news = get_news(symbol)
            if news:
                for n in news:
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 15px;">
                        <a href="{n['link']}" target="_blank" style="text-decoration:none; color:#fff;">
                            <h4 style="margin:0;">{n['title']}</h4>
                        </a>
                        <div style="font-size:0.8rem; color:#888; margin-top:5px;">Source: {n['source']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ไม่พบข่าวสำคัญในขณะนี้")

        # 5. STATS (FIX: ใช้ข้อมูลจาก DF แทน info เพื่อแก้ปัญหา Data unavailable)
        with tabs[4]:
            st.markdown("### 📊 Market Statistics")
            # คำนวณค่าจาก DataFrame ที่มีอยู่แล้ว (มั่นใจว่ามีข้อมูลแน่นอน)
            high_p = df['High'].max()
            low_p = df['Low'].min()
            vol_p = df['Volume'].sum()
            avg_p = df['Close'].mean()

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            
            c1.metric(f"Highest ({period})", f"{high_p:,.2f}")
            c2.metric(f"Lowest ({period})", f"{low_p:,.2f}")
            c3.metric(f"Avg Price", f"{avg_p:,.2f}")
            c4.metric(f"Total Volume", f"{vol_p:,.0f}")
            
            st.caption(f"หมายเหตุ: ข้อมูลสถิติคคำนวณจากกราฟย้อนหลัง {period}")
