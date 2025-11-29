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

# Import translation library
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

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

        /* NEWS CARD */
        .news-card { padding: 20px; margin-bottom: 15px; background: #111; border-radius: 15px; border-left: 6px solid #888; transition: transform 0.2s; }
        .news-card:hover { transform: scale(1.01); background: #161616; }
        .nc-pos { border-left-color: #00E676; }
        .nc-neg { border-left-color: #FF1744; }
        .nc-neu { border-left-color: #FFD600; }

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

# --- NEW: AI Calculation for THB ---
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
    if last_price > mid_point:
        status = "BULLISH (กระทิง)"
        status_color = "#00E676"
    else:
        status = "BEARISH (หมี)"
        status_color = "#FF1744"
        
    return {
        "levels": [
            {"name": "🚀 R2 (Breakout)", "price": r2, "type": "res"},
            {"name": "🛑 R1 (Resist)", "price": r1, "type": "res"},
            {"name": "⚖️ PIVOT (จุดหมุน)", "price": pp, "type": "neu"},
            {"name": "🛡️ S1 (Support)", "price": s1, "type": "sup"},
            {"name": "💎 S2 (Bottom)", "price": s2, "type": "sup"}
        ],
        "fib": {"top": fib_high, "bot": fib_low},
        "status": status,
        "color": status_color
    }

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

        if close > ema50 and ema50 > ema200:
            trend = "Uptrend (ขาขึ้น)"
            signal = "BUY / LONG"
            color = "#00E676"
            score_trend = 2
        elif close < ema50 and ema50 < ema200:
            trend = "Downtrend (ขาลง)"
            signal = "SELL / SHORT"
            color = "#FF1744"
            score_trend = -2
        else:
            trend = "Sideways (ไซด์เวย์)"
            signal = "WAIT"
            color = "#888"
            score_trend = 0
        
        return {
            'trend': trend, 'signal': signal, 'color': color, 
            'entry': close, 'sl': close - (1.5*atr) if score_trend>=0 else close + (1.5*atr),
            'tp': close + (2.5*atr) if score_trend>=0 else close - (2.5*atr),
            'rsi': rsi, 'ema50': ema50, 'ema200': ema200
        }
    except: return None

def identify_sr_levels(df):
    levels = []
    try:
        window = 5
        for i in range(window, len(df) - window):
            if df['Low'][i] == df['Low'][i-window:i+window+1].min():
                levels.append({'price': df['Low'][i], 'type': 'Support'})
            elif df['High'][i] == df['High'][i-window:i+window+1].max():
                levels.append({'price': df['High'][i], 'type': 'Resistance'})
        levels.sort(key=lambda x: x['price'])
        filtered = []
        if levels:
            curr = levels[0]
            for next_lvl in levels[1:]:
                if (next_lvl['price'] - curr['price']) / curr['price'] > 0.02:
                    filtered.append(curr)
                    curr = next_lvl
            filtered.append(curr)
        return filtered
    except: return []

def calculate_pivot_points(df):
    try:
        prev = df.iloc[-2]
        high, low, close = prev['High'], prev['Low'], prev['Close']
        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        s1 = (2 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        return {"PP": pp, "R1": r1, "S1": s1, "R2": r2, "S2": s2}
    except: return None

def calculate_dynamic_levels(df):
    try:
        close = df['Close'].iloc[-1]
        ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        sma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        std = df['Close'].rolling(window=20).std().iloc[-1]
        bb_upper = sma20 + (2 * std)
        bb_lower = sma20 - (2 * std)
        return {
            "EMA 20": ema20, "EMA 50": ema50, "EMA 200": ema200,
            "BB Upper": bb_upper, "BB Lower": bb_lower,
            "Current": close
        }
    except: return None

@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    news_list = []
    clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
    translator = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    try:
        q = urllib.parse.quote(f"site:bloomberg.com {clean_sym} market")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        if len(feed.entries) == 0:
            q = urllib.parse.quote(f"{clean_sym} finance news")
            rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)

        for item in feed.entries[:5]:
            blob = TextBlob(item.title)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.05:
                sentiment = "ข่าวดี (Positive)"
                color_class = "nc-pos"
                icon = "🚀"
            elif sentiment_score < -0.05:
                sentiment = "ข่าวร้าย (Negative)"
                color_class = "nc-neg"
                icon = "🔻"
            else:
                sentiment = "ทั่วไป (Neutral)"
                color_class = "nc-neu"
                icon = "⚖️"

            title_th = item.title
            if translator:
                try: title_th = translator.translate(item.title)
                except: pass

            news_list.append({
                'title_th': title_th, 'link': item.link, 'sentiment': sentiment,
                'class': color_class, 'icon': icon, 'score': sentiment_score
            })
    except: pass
    return news_list

def generate_ai_analysis(df, setup, news_list):
    analysis_text = ""
    score = 50
    
    if setup['trend'] == "Uptrend (ขาขึ้น)":
        analysis_text += "📈 **ด้านเทคนิค:** กราฟยังคงรักษาทรงขาขึ้นได้อย่างแข็งแกร่ง ราคายืนเหนือเส้น EMA "
        score += 25
    elif setup['trend'] == "Downtrend (ขาลง)":
        analysis_text += "📉 **ด้านเทคนิค:** กราฟอยู่ในแนวโน้มขาลงชัดเจน ระวังแรงขายกดดัน "
        score -= 25
    else:
        analysis_text += "⚖️ **ด้านเทคนิค:** กราฟแกว่งตัวออกข้าง (Sideways) รอเลือกทาง "

    if setup['rsi'] > 70: score -= 5
    elif setup['rsi'] < 30: score += 5
    
    news_score = sum([n['score'] for n in news_list]) if news_list else 0
    if news_score > 0.2:
        analysis_text += "\n\n📰 **กระแสข่าว:** ภาพรวมข่าวสารเป็นบวก สนับสนุนราคา"
        score += 15
    elif news_score < -0.2:
        analysis_text += "\n\n📰 **กระแสข่าว:** มีข่าวเชิงลบกดดันตลาด ควรระวัง"
        score -= 15

    score = max(0, min(100, score))
    if score >= 75: verdict = "STRONG BUY"
    elif score >= 55: verdict = "BUY"
    elif score >= 45: verdict = "HOLD / WATCH"
    elif score >= 25: verdict = "SELL"
    else: verdict = "STRONG SELL"
    
    return analysis_text, score, verdict

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    st.caption("AI Edition (No Guru)")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    if c1.button("BTC-USD"): set_symbol("BTC-USD")
    if c2.button("ETH-USD"): set_symbol("ETH-USD")
    c3, c4 = st.columns(2)
    if c3.button("Gold"): set_symbol("GC=F")
    if c4.button("Oil"): set_symbol("CL=F")
    
    st.markdown("---")
    st.markdown("### 🇹🇭 Bitkub Rates (THB)")
    bk_data = get_bitkub_ticker()
    if bk_data:
        btc_thb = bk_data.get('THB_BTC', {}).get('last', 0)
        btc_chg = bk_data.get('THB_BTC', {}).get('percentChange', 0)
        c_btc = "#00E676" if btc_chg >= 0 else "#FF1744"
        
        eth_thb = bk_data.get('THB_ETH', {}).get('last', 0)
        eth_chg = bk_data.get('THB_ETH', {}).get('percentChange', 0)
        c_eth = "#00E676" if eth_chg >= 0 else "#FF1744"

        st.markdown(f"""
<div style="background:#111; padding:10px; border-radius:10px; margin-bottom:5px;">
<div style="display:flex; justify-content:space-between;">
<span style="font-size:0.9rem; color:#aaa;">BTC/THB</span>
<span style="color:{c_btc}; font-size:0.8rem;">{btc_chg:+.2f}%</span>
</div>
<div style="font-size:1.2rem; font-weight:bold; color:{c_btc};">{btc_thb:,.2f}</div>
</div>
<div style="background:#111; padding:10px; border-radius:10px;">
<div style="display:flex; justify-content:space-between;">
<span style="font-size:0.9rem; color:#aaa;">ETH/THB</span>
<span style="color:{c_eth}; font-size:0.8rem;">{eth_chg:+.2f}%</span>
</div>
<div style="font-size:1.2rem; font-weight:bold; color:{c_eth};">{eth_thb:,.2f}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.caption("กำลังโหลดข้อมูล Bitkub...")
        
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    chart_type = st.selectbox("Chart Style", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Period", options=["1mo", "3mo", "6mo", "1y", "5y"], value="1y")

# --- 5. Main Content ---

st.markdown("<h2 style='color:#00E5FF;'>🔍 Smart Search</h2>", unsafe_allow_html=True)
c_search, c_btn = st.columns([4, 1])
with c_search:
    sym_input = st.text_input("Symbol", value=st.session_state.symbol, label_visibility="collapsed")
with c_btn:
    st.write("")
    if st.button("วิเคราะห์ ⚡", use_container_width=True):
        st.session_state.symbol = sym_input
        st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner('💎 AI กำลังประมวลผลข้อมูล...'):
        df = get_market_data(symbol, period, "1d")
        info = get_stock_info(symbol)
        
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล '{symbol}'")
    else:
        curr_price = df['Close'].iloc[-1]
        change = curr_price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        
        setup = calculate_technical_setup(df)
        news_list = get_ai_analyzed_news_thai(symbol)
        sr_levels = identify_sr_levels(df)
        
        ai_text, ai_score, ai_verdict = generate_ai_analysis(df, setup, news_list)

        # --- HERO HEADER ---
        color_trend = "#00E676" if change >= 0 else "#FF1744"
        arrow = "▲" if change >= 0 else "▼"
        st.markdown(f"""
        <div class="glass-card" style="border-top: 6px solid {color_trend}; text-align: center; padding-top:40px;">
            <div style="font-size: 5rem; font-weight: 900; background: -webkit-linear-gradient(45deg, #fff, {color_trend}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{symbol}</div>
            <div style="font-size: 4.5rem; font-weight: 800; color: {color_trend};">{curr_price:,.2f}</div>
            <div style="background: {color_trend}20; padding: 10px 30px; border-radius: 30px; display: inline-block; border: 2px solid {color_trend};">
                <span style="font-size: 1.8rem; font-weight:bold; color:{color_trend};">{arrow} {abs(change):,.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- TABS ---
        tabs = st.tabs(["📈 Chart", "📊 Stats & Funda", "📰 AI News", "🎯 S/R & Setup", "💰 Entry", "🤖 AI Verdict", "🛡️ S/R Dynamics", "🇹🇭 Bitkub AI S/R"])

        # Tab 1: Chart
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['HA_Open'], high=ha['HA_High'], low=ha['HA_Low'], close=ha['HA_Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF', width=2), name='EMA 50'), row=1, col=1)
            rsi = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0,0).rolling(14).mean() / abs(df['Close'].diff().where(lambda x: x<0,0)).rolling(14).mean())))
            fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#E040FB', width=2), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Stats & Fundamentals
        with tabs[1]:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class="stat-box"><div class="stat-label">High</div><div class="stat-value" style="color:#00E676;">{df['High'].max():,.2f}</div></div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="stat-box"><div class="stat-label">Low</div><div class="stat-value" style="color:#FF1744;">{df['Low'].min():,.2f}</div></div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="stat-box"><div class="stat-label">Vol</div><div class="stat-value" style="color:#E040FB;">{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>""", unsafe_allow_html=True)
            st.markdown("---")
            if info:
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                summary = info.get('longBusinessSummary', 'No description available.')
                summary_th = summary
                if HAS_TRANSLATOR:
                    try:
                        translator = GoogleTranslator(source='auto', target='th')
                        summary_th = translator.translate(summary[:4500])
                    except: pass
                
                st.markdown(f"### 🏢 Business Profile: {sector} / {industry}")
                with st.expander("อ่านรายละเอียดธุรกิจ (Business Summary)", expanded=True):
                    st.write(summary_th)
            
                st.markdown("### 📊 Fundamental Valuation")
                pe = info.get('trailingPE')
                eps = info.get('trailingEps')
                peg = info.get('pegRatio')
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    eps_val = f"{eps:.2f}" if eps else "N/A"
                    eps_color = "#00E676" if eps and eps > 0 else "#FF1744"
                    st.markdown(f"""<div class="fund-box" style="border-left: 5px solid {eps_color};"><div class="fund-title">EPS (กำไรต่อหุ้น)</div><div class="fund-val">{eps_val}</div><div class="fund-desc">Earnings Per Share</div></div>""", unsafe_allow_html=True)
                with col_f2:
                    pe_val = f"{pe:.2f}" if pe else "N/A"
                    pe_status = "N/A"
                    pe_color = "#888"
                    if pe:
                        if pe < 15: pe_status, pe_color = "Undervalued (ถูก)", "#00E676"
                        elif pe > 30: pe_status, pe_color = "Overvalued (แพง)", "#FF1744"
                        else: pe_status, pe_color = "Fair Price (เหมาะสม)", "#FFD600"
                    st.markdown(f"""<div class="fund-box" style="border-left: 5px solid {pe_color};"><div class="fund-title">P/E Ratio</div><div class="fund-val">{pe_val}</div><div class="fund-desc" style="color:{pe_color}; font-weight:bold;">{pe_status}</div></div>""", unsafe_allow_html=True)
                with col_f3:
                    peg_val = f"{peg:.2f}" if peg else "N/A"
                    peg_status, peg_color = "", "#888"
                    if peg:
                        if peg < 1: peg_status, peg_color = "Growth is Cheap (น่าสนใจ)", "#00E676"
                        elif peg > 2: peg_status, peg_color = "Growth is Pricey (ตึงตัว)", "#FF1744"
                        else: peg_status, peg_color = "Reasonable (สมเหตุสมผล)", "#FFD600"
                    st.markdown(f"""<div class="fund-box" style="border-left: 5px solid {peg_color};"><div class="fund-title">PEG Ratio (เทียบการเติบโต)</div><div class="fund-val">{peg_val}</div><div class="fund-desc" style="color:{peg_color};">{peg_status}</div></div>""", unsafe_allow_html=True)
                st.caption("*หมายเหตุ: การประเมิน P/E เทียบกับเกณฑ์มาตรฐานทั่วไป ควรพิจารณาร่วมกับ PEG Ratio (เทียบการเติบโต) และปัจจัยเฉพาะอุตสาหกรรม")
            else:
                st.info("ไม่พบข้อมูลพื้นฐาน (Fundamental Data) สำหรับสินทรัพย์นี้ (อาจเป็น Crypto หรือ Commodity)")

        # Tab 3: AI News
        with tabs[2]:
            st.markdown("### 📰 AI Sentiment Analysis (วิเคราะห์อารมณ์ข่าว)")
            if news_list:
                for n in news_list:
                    st.markdown(f"""<div class="news-card {n['class']}"><div style="display:flex; justify-content:space-between; align-items:center;"><span style="font-size:0.9rem; font-weight:bold; padding:4px 10px; border-radius:10px; background:#fff; color:#000;">{n['icon']} {n['sentiment']}</span></div><h4 style="color:#fff; margin:10px 0;">{n['title_th']}</h4><a href="{n['link']}" target="_blank" style="color:#aaa; font-size:0.9rem;">🔗 อ่านข่าวต้นฉบับ</a></div>""", unsafe_allow_html=True)
            else: st.info("ไม่พบข่าวในขณะนี้ หรือ API ถูกจำกัดการเข้าถึง")

        # Tab 4: S/R & Setup
        with tabs[3]:
            st.markdown("### 🛡️ Key Levels (แนวรับ-แนวต้าน)")
            res_list = sorted([l['price'] for l in sr_levels if l['price'] > curr_price])[:3]
            sup_list = sorted([l['price'] for l in sr_levels if l['price'] < curr_price], reverse=True)[:3]
            sr_html = "<div class='sr-container'>"
            for r in reversed(res_list): sr_html += f"<div class='sr-row res-row'><div>RESISTANCE</div><div>{r:,.2f}</div></div>"
            sr_html += f"<div class='sr-row curr-row'><div>CURRENT: {curr_price:,.2f}</div></div>"
            for s in sup_list: sr_html += f"<div class='sr-row sup-row'><div>SUPPORT</div><div>{s:,.2f}</div></div>"
            sr_html += "</div>"
            st.markdown(sr_html, unsafe_allow_html=True)
            st.markdown("### 🎯 Technical Signal")
            if setup:
                st.markdown(f"""<div class="glass-card" style="text-align:center; border:2px solid {setup['color']}"><h1 style="color:{setup['color']}">{setup['signal']}</h1><p style="font-size:1.5rem;">{setup['trend']}</p></div>""", unsafe_allow_html=True)

        # Tab 5: Entry
        with tabs[4]:
            st.markdown("### 💰 Entry Levels")
            t1, t2, t3 = curr_price*0.99, curr_price*0.97, curr_price*0.94
            st.markdown(f"""<div style="background:#111; padding:20px; border-left:5px solid #00E5FF; margin-bottom:10px; font-size:1.2rem;"><b>Probe Buy (20%):</b> {t1:,.2f}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:#111; padding:20px; border-left:5px solid #FFD600; margin-bottom:10px; font-size:1.2rem;"><b>Accumulate (30%):</b> {t2:,.2f}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:#111; padding:20px; border-left:5px solid #FF1744; font-size:1.2rem;"><b>Sniper Zone (50%):</b> {t3:,.2f}</div>""", unsafe_allow_html=True)

        # Tab 6: AI Verdict
        with tabs[5]:
            st.markdown("### 🤖 AI Market Analysis")
            if ai_score >= 70: score_color = "#00E676"
            elif ai_score <= 30: score_color = "#FF1744"
            else: score_color = "#FFD600"
            st.markdown(f"""<div class="ai-card" style="text-align:center; border-color:{score_color};"><div class="ai-score-circle" style="border-color:{score_color}; color:{score_color};">{ai_score}</div><div style="font-size:2rem; font-weight:bold; color:{score_color};">{ai_verdict}</div><p>{ai_text}</p></div>""", unsafe_allow_html=True)
            
        # Tab 7: S/R Dynamics
        with tabs[6]:
            pivots = calculate_pivot_points(df)
            dynamic = calculate_dynamic_levels(df)
            col_static, col_dynamic = st.columns(2)
            with col_static:
                st.markdown("### 🧱 Static Levels (Pivot Points)")
                if pivots:
                    # Fix indentation for HTML string
                    st.markdown(f"""
<div style="display:flex; flex-direction:column; gap:8px;">
<div style="background:#220a0a; border:1px solid #FF1744; padding:15px; border-radius:10px; display:flex; justify-content:space-between;"><span style="color:#FF1744; font-weight:bold;">R2 (ต้านแข็ง)</span> <span style="font-weight:bold;">{pivots['R2']:,.2f}</span></div>
<div style="background:#221111; border:1px solid #FF5252; padding:15px; border-radius:10px; display:flex; justify-content:space-between;"><span style="color:#FF5252; font-weight:bold;">R1 (ต้านแรก)</span> <span style="font-weight:bold;">{pivots['R1']:,.2f}</span></div>
<div style="background:#1a1a1a; border:1px solid #FFD600; padding:15px; border-radius:10px; display:flex; justify-content:space-between; transform:scale(1.02); box-shadow:0 0 10px rgba(255,214,0,0.2);"><span style="color:#FFD600; font-weight:bold;">PIVOT (จุดหมุน)</span> <span style="font-weight:bold;">{pivots['PP']:,.2f}</span></div>
<div style="background:#0a1a11; border:1px solid #69F0AE; padding:15px; border-radius:10px; display:flex; justify-content:space-between;"><span style="color:#69F0AE; font-weight:bold;">S1 (รับแรก)</span> <span style="font-weight:bold;">{pivots['S1']:,.2f}</span></div>
<div style="background:#0a2215; border:1px solid #00E676; padding:15px; border-radius:10px; display:flex; justify-content:space-between;"><span style="color:#00E676; font-weight:bold;">S2 (รับแข็ง)</span> <span style="font-weight:bold;">{pivots['S2']:,.2f}</span></div>
</div>""", unsafe_allow_html=True)
            with col_dynamic:
                st.markdown("### 🌊 Dynamic Levels (Moving Avgs)")
                if dynamic:
                    curr = dynamic['Current']
                    def get_status(price, level):
                        diff = ((price - level) / level) * 100
                        if price > level: return "SUPPORT (รับ)", "#00E676", f"+{diff:.2f}%"
                        else: return "RESIST (ต้าน)", "#FF1744", f"{diff:.2f}%"
                    dyn_items = [("BB Upper", dynamic['BB Upper']), ("EMA 20", dynamic['EMA 20']), ("EMA 50", dynamic['EMA 50']), ("EMA 200", dynamic['EMA 200']), ("BB Lower", dynamic['BB Lower'])]
                    dyn_items.sort(key=lambda x: x[1], reverse=True)
                    html_dyn = "<div style='display:flex; flex-direction:column; gap:10px;'>"
                    for name, val in dyn_items:
                        role, color, pct = get_status(curr, val)
                        # Fix single line f-string
                        html_dyn += f"<div style='background:{color}10; border-left:5px solid {color}; padding:15px; border-radius:5px; display:flex; justify-content:space-between; align-items:center;'><div><div style='font-size:0.8rem; color:#888;'>{name}</div><div style='font-size:1.2rem; font-weight:bold;'>{val:,.2f}</div></div><div style='text-align:right;'><div style='font-size:0.9rem; font-weight:bold; color:{color};'>{role}</div><div style='font-size:0.8rem; color:#ccc;'>Dist: {pct}</div></div></div>"
                    html_dyn += "</div>"
                    st.markdown(html_dyn, unsafe_allow_html=True)

        # Tab 8: Bitkub AI S/R (NEW)
        with tabs[7]:
            st.markdown("### 🇹🇭 Bitkub AI Support & Resistance (บาท)")
            bk_coin_sel = st.radio("เลือกเหรียญ (THB Pair)", ["BTC", "ETH"], horizontal=True)
            
            if bk_data:
                pair = f"THB_{bk_coin_sel}"
                coin_data = bk_data.get(pair, {})
                
                if coin_data:
                    last_thb = coin_data.get('last', 0)
                    high_24 = coin_data.get('high24hr', 0)
                    low_24 = coin_data.get('low24hr', 0)
                    
                    # AI Calculation
                    ai_levels = calculate_bitkub_ai_levels(high_24, low_24, last_thb)
                    
                    # Display Big Price
                    st.markdown(f"""
<div style="text-align:center; padding:20px; background:#111; border-radius:20px; border:2px solid {ai_levels['color']}; margin-bottom:20px;">
<div style="font-size:1.2rem; color:#aaa;">ราคา Bitkub ล่าสุด</div>
<div style="font-size:3.5rem; font-weight:bold; color:#fff;">{last_thb:,.2f} <span style="font-size:1.5rem;">THB</span></div>
<div style="font-size:1.5rem; font-weight:bold; color:{ai_levels['color']};">{ai_levels['status']}</div>
</div>""", unsafe_allow_html=True)
                    
                    c_ai_1, c_ai_2 = st.columns(2)
                    
                    with c_ai_1:
                        st.markdown("#### 🤖 AI Pivot Levels (Intraday)")
                        html_lvls = "<div style='display:flex; flex-direction:column; gap:8px;'>"
                        for lvl in ai_levels['levels']:
                            color = "#00E676" if lvl['type'] == 'sup' else "#FF1744" if lvl['type'] == 'res' else "#FFD600"
                            # Fix indented f-string to prevent code block rendering
                            html_lvls += f"""<div style="display:flex; justify-content:space-between; padding:15px; background:#161616; border-left:5px solid {color}; border-radius:5px;"><span style="font-weight:bold; color:{color};">{lvl['name']}</span><span style="font-weight:bold; font-size:1.1rem;">{lvl['price']:,.2f}</span></div>"""
                        html_lvls += "</div>"
                        st.markdown(html_lvls, unsafe_allow_html=True)
                        
                    with c_ai_2:
                        st.markdown("#### 📐 Fibonacci Golden Zone (24H)")
                        st.info(f"""
                        **กรอบทองคำ (Golden Zone):**
                        \nโซนกลับตัวที่มีนัยยะสำคัญจาก High/Low ใน 24 ชม.
                        \n\n🟢 **Golden Bottom (รับ):** {ai_levels['fib']['bot']:,.2f}
                        \n🔴 **Golden Top (ต้าน):** {ai_levels['fib']['top']:,.2f}
                        """)
                        
                        st.markdown("#### 🧠 AI Insight")
                        st.caption(f"""
                        ระบบวิเคราะห์จากพฤติกรรมราคาใน 24 ชั่วโมงล่าสุดของ Bitkub พบว่าราคาปัจจุบันอยู่ที่ **{last_thb:,.2f}** 
                        ซึ่งอยู่ในโซน **{ai_levels['status']}** เมื่อเทียบกับจุดกึ่งกลางวัน
                        \nแนะนำให้จับตาดูแนวรับ **S1 ({ai_levels['levels'][3]['price']:,.0f})** หากรับอยู่มีโอกาสเด้งไปทดสอบ Pivot
                        """)
                else:
                    st.error("ไม่พบข้อมูลเหรียญนี้ใน Bitkub API")
            else:
                st.warning("กำลังเชื่อมต่อ Bitkub API...")
