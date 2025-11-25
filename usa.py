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
            
            # --- FIX: Syntax Error Fixed Here (Added :) ---
            if abs(price - current_price)/current_price > 0.5 and strength == "Minor":
                continue
            
            lvl['strength'], lvl['desc'] = strength, desc
            final.append(lvl)
        return final
    except: return []

def calculate_trade_setup(df):
    try:
        close = df['Close'].iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = np.max(ranges, axis=1).rolling(14).mean().iloc[-1]
        
        if close > ema50 and ema50 > ema200:
            trend = "Uptrend (ขาขึ้น)"
            signal = "LONG / BUY"
            color = "#00E676"
            sl = close - (1.5 * atr)
            tp = close + (2.5 * atr)
        elif close < ema50 and ema50 < ema200:
            trend = "Downtrend (ขาลง)"
            signal = "SHORT / SELL"
            color = "#FF1744"
            sl = close + (1.5 * atr)
            tp = close - (2.5 * atr)
        else:
            trend = "Sideways"
            signal = "WAIT"
            color = "#888"
            sl = close - atr
            tp = close + atr
            
        return {'trend': trend, 'signal': signal, 'color': color, 'entry': close, 'sl': sl, 'tp': tp, 'atr': atr}
    except: return None

def calculate_tiered_entries(df, sr_levels):
    try:
        current_price = df['Close'].iloc[-1]
        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = np.max(ranges, axis=1).rolling(14).mean().iloc[-1]
        
        supports = sorted([s['price'] for s in sr_levels if s['price'] < current_price], reverse=True)
        t1 = supports[0] if len(supports) > 0 else current_price * 0.98
        t2 = supports[1] if len(supports) > 1 else t1 - (2 * atr)
        t3 = supports[2] if len(supports) > 2 else t2 - (3 * atr)
        
        if (t1 - t2) < atr: t2 = t1 - atr
        if (t2 - t3) < atr: t3 = t2 - (1.5 * atr)

        return {'t1': t1, 't2': t2, 't3': t3, 'atr': atr}
    except: return None

# --- UPDATE: Bloomberg News Function ---
@st.cache_data(ttl=3600)
def get_bloomberg_news(symbol):
    news_list = []
    clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
    try:
        # 1. Search specifically for site:bloomberg.com
        # Using Google News RSS to filter for Bloomberg domain
        q = urllib.parse.quote(f"site:bloomberg.com {clean_sym}")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        for item in feed.entries[:5]:
             news_list.append({
                 'title': item.title, 
                 'link': item.link, 
                 'summary': item.description, 
                 'source': 'Bloomberg'
             })
             
        # 2. Fallback: If Bloomberg has no recent news, get general finance news
        if len(news_list) == 0:
             q_gen = urllib.parse.quote(f"{clean_sym} finance news")
             feed_gen = feedparser.parse(f"https://news.google.com/rss/search?q={q_gen}&hl=en-US&gl=US&ceid=US:en")
             for item in feed_gen.entries[:3]:
                 news_list.append({
                     'title': item.title, 
                     'link': item.link, 
                     'summary': item.description, 
                     'source': 'General News'
                 })
    except: pass
    return news_list

def analyze_sentiment_advanced(text, title):
    try:
        blob = TextBlob(text + " " + title)
        polarity = blob.sentiment.polarity
        if polarity > 0.05: cat, badge, icon = "Positive", "badge-pos", "🚀"
        elif polarity < -0.05: cat, badge, icon = "Negative", "badge-neg", "🔻"
        else: cat, badge, icon = "Neutral", "badge-neu", "⚖️"
        
        return {'title': title, 'summary': text[:200]+"...", 'cat': cat, 'badge': badge, 'icon': icon}
    except: return None

# --- 3. Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=60)
    st.title("Smart Trader AI")
    st.caption("Pro Max Edition")
    
    st.markdown("### 🏆 Top Assets")
    st.markdown("**🇹🇭 Thai Market**")
    c1, c2 = st.columns(2)
    if c1.button("BTC-THB"): set_symbol("BTC-THB")
    if c2.button("ETH-THB"): set_symbol("ETH-THB")
    
    st.markdown("**🌎 Global Market**")
    c3, c4 = st.columns(2)
    if c3.button("BTC-USD"): set_symbol("BTC-USD")
    if c4.button("ETH-USD"): set_symbol("ETH-USD")
    
    st.markdown("**📉 Commodities**")
    c5, c6 = st.columns(2)
    if c5.button("Gold"): set_symbol("GC=F")
    if c6.button("Oil"): set_symbol("CL=F")
    
    st.markdown("---")
    st.subheader("🛠 Configuration")
    chart_type = st.selectbox("รูปแบบกราฟ", ["Candlestick", "Heikin Ashi"], index=0)
    period = st.select_slider("ย้อนหลัง", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], value="1y")
    interval = st.selectbox("Timeframe", ["1d", "1wk", "1h"], index=0)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**KRITCHANUT VONGRAT**")

# --- 4. Main Interface ---
st.markdown("### 🔍 ค้นหาหุ้น / เหรียญ (Smart Search)")

# Search Bar Area
col_search, col_btn = st.columns([4, 1])
with col_search:
    sym_input = st.text_input("ชื่อย่อ (เช่น BTC-THB, PTT.BK, DELTA.BK)", value=st.session_state.symbol, label_visibility="collapsed")
with col_btn:
    if st.button("วิเคราะห์ทันที ⚡", use_container_width=True):
        st.session_state.symbol = sym_input
        st.rerun()
st.caption("💡 Tip: หุ้นไทยเติม `.BK` | คริปโตไทยใช้ `-THB`")

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner(f'🤖 AI กำลังประมวลผลกลยุทธ์สำหรับ {symbol}...'):
        df, ticker = get_data(symbol, period, interval)
    
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล '{symbol}' หรือ API ถูกจำกัดการเข้าถึง (ลองกด Refresh หรือเปลี่ยนคู่เหรียญ)")
    else:
        # Data Processing
        curr_price = df['Close'].iloc[-1]
        change = curr_price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        
        # Calculate Indicators
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        sr_levels = identify_levels(df)
        setup = calculate_trade_setup(df)
        entries = calculate_tiered_entries(df, sr_levels)
        
        # Try fetch info (won't crash if fails)
        info = get_fundamentals_safe(ticker)

        # --- Header ---
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-left: 10px solid {'#00E676' if change>=0 else '#FF1744'};">
                <h1 style="margin:0; font-size: 3rem;">{symbol}</h1>
                <h2 style="margin:0; font-size: 4rem; color: {'#00E676' if change>=0 else '#FF1744'};">{curr_price:,.2f}</h2>
                <p style="font-size: 1.5rem; color: #aaa;">{change:+,.2f} ({pct:+.2f}%)</p>
                <p style="color: #666;">{info.get('longName', '')}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # --- 6 Full Tabs Restored ---
        t1, t2, t3, t4, t5, t6 = st.tabs([
            "📈 Smart Chart", 
            "🛡️ S/R Levels", 
            "🎯 Trade Setup", 
            "📊 Fundamentals", 
            "🧠 AI Sentiment", 
            "💰 Entry Strategy"
        ])
        
        # Tab 1: Chart
        with t1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            if chart_type == "Heikin Ashi":
                plot_df = calculate_heikin_ashi(df)
                o, h, l, c_plot = plot_df['HA_Open'], plot_df['HA_High'], plot_df['HA_Low'], plot_df['HA_Close']
            else:
                o, h, l, c_plot = df['Open'], df['High'], df['Low'], df['Close']
                
            fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=l, close=c_plot, name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2979FF', width=1), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#FF9100', width=1), name='EMA 200'), row=1, col=1)
            
            # Draw SR Lines
            for lvl in sr_levels:
                if abs(lvl['price'] - curr_price)/curr_price < 0.2:
                    color_line = 'rgba(0,230,118,0.5)' if lvl['type']=='Support' else 'rgba(255,23,68,0.5)'
                    fig.add_hline(y=lvl['price'], line_dash='dash', line_color=color_line, row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AB47BC', width=1.5), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: S/R Levels
        with t2:
            st.markdown("### 🛡️ Support & Resistance Zones")
            c_res, c_sup = st.columns(2)
            res = sorted([l for l in sr_levels if l['price'] > curr_price], key=lambda x:x['price'])[:5]
            sup = sorted([l for l in sr_levels if l['price'] < curr_price], key=lambda x:x['price'], reverse=True)[:5]
            with c_res:
                st.error("🟥 RESISTANCE (แนวต้าน)")
                for r in reversed(res):
                    tag = "sr-strong" if r['strength']=="Strong" else "sr-weak"
                    st.markdown(f"<div style='border-bottom:1px solid #333; padding:10px; display:flex; justify-content:space-between;'><span>{r['price']:,.2f}</span><span class='sr-tag {tag}'>{r['desc']}</span></div>", unsafe_allow_html=True)
            with c_sup:
                st.success("🟩 SUPPORT (แนวรับ)")
                for s in sup:
                    tag = "sr-strong" if s['strength']=="Strong" else "sr-weak"
                    st.markdown(f"<div style='border-bottom:1px solid #333; padding:10px; display:flex; justify-content:space-between;'><span>{s['price']:,.2f}</span><span class='sr-tag {tag}'>{s['desc']}</span></div>", unsafe_allow_html=True)

        # Tab 3: Trade Setup
        with t3:
            st.markdown("### 🎯 AI Trade Plan")
            if setup:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 10px solid {setup['color']}; display:flex; justify-content:space-between; align-items:center;">
                    <div><h2 style="margin:0; color:{setup['color']}">{setup['signal']}</h2><p style="margin:0; color:#aaa;">Trend: {setup['trend']}</p></div>
                    <div style="text-align:right;"><h1>{setup['entry']:,.2f}</h1><small>Entry Price</small></div>
                </div>
                """, unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='glass-card' style='text-align:center; border:1px solid #FF1744;'><h3 style='color:#FF1744'>STOP LOSS</h3><h1>{setup['sl']:,.2f}</h1></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='glass-card' style='text-align:center; border:1px solid #00E676;'><h3 style='color:#00E676'>TAKE PROFIT</h3><h1>{setup['tp']:,.2f}</h1></div>", unsafe_allow_html=True)

        # Tab 4: Fundamentals
        with t4:
            st.markdown("### 📊 Fundamental Data")
            if info and 'marketCap' in info:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Market Cap", f"{info.get('marketCap', 0):,}")
                c2.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                c3.metric("High (52W)", f"{info.get('fiftyTwoWeekHigh', 0):,.2f}")
                c4.metric("Low (52W)", f"{info.get('fiftyTwoWeekLow', 0):,.2f}")
                st.info(f"ℹ️ **Business Summary:** {info.get('longBusinessSummary', 'No Data Available')[:600]}...")
            else:
                st.warning("⚠️ ข้อมูลพื้นฐานไม่สามารถดึงได้ในขณะนี้ (Data Unavailable / Rate Limited) แต่กราฟเทคนิคยังใช้งานได้ปกติ")

        # Tab 5: Sentiment
        with t5:
            st.markdown("### 🧠 AI Sentiment Analysis (Source: Bloomberg)")
            raw_news = get_bloomberg_news(symbol)
            if raw_news:
                for item in raw_news:
                    res = analyze_sentiment_advanced(item['summary'], item['title'])
                    if res:
                        st.markdown(f"""
                        <div class="sentiment-card">
                            <div style="display:flex; justify-content:space-between;">
                                <span class="{res['badge']}">{res['icon']} {res['cat']}</span>
                                <small style="color:#aaa;">Source: {item['source']}</small>
                            </div>
                            <h4 style="margin:10px 0;">{res['title']}</h4>
                            <p style="color:#aaa; font-size:0.9rem;">{res['summary']}</p>
                            <a href="{item['link']}" target="_blank" style="font-size:0.8rem;">🔗 อ่านข่าวต้นฉบับ</a>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("ไม่พบข่าวจาก Bloomberg หรือ API News ถูกจำกัดการเข้าถึงในขณะนี้")

        # Tab 6: Entry Strategy
        with t6:
            st.markdown("### 💰 Money Management (Tiered Entry)")
            if entries:
                st.markdown(f"""
                <div class="entry-card ec-tier1">
                    <div class="ec-title-1">🪵 ไม้แรก (Probe Buy - 20%)</div>
                    <div class="ec-price">{entries['t1']:,.2f}</div>
                    <small>เข้าซื้อเพื่อหยั่งเชิง บริเวณแนวรับแรก</small>
                </div>
                <div class="entry-card ec-tier2">
                    <div class="ec-title-2">🪵 ไม้สอง (Accumulate - 30%)</div>
                    <div class="ec-price">{entries['t2']:,.2f}</div>
                    <small>ซื้อเพิ่มเมื่อย่อตัว (Correction)</small>
                </div>
                <div class="entry-card ec-tier3">
                    <div class="ec-title-3">💎 ไม้หนัก (Sniper - 50%)</div>
                    <div class="ec-price">{entries['t3']:,.2f}</div>
                    <small>จุด All-in หรือ Panic Sell zone</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("ข้อมูลไม่เพียงพอสำหรับคำนวณ Entry Strategy")
