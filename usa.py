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
import re

# --- Libraries Setup ---
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# API Config
FINNHUB_KEY = "d4l5ku1r01qt7v18ll40d4l5ku1r01qt7v18ll4g" 

# --- 1. Setup & Design ---
st.set_page_config(
    page_title="Smart Trader AI : Ultra Black",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

if 'symbol' not in st.session_state: st.session_state.symbol = 'BTC-USD'

def set_symbol(sym): st.session_state.symbol = sym

# --- 2. CSS Styling (Ultra Black UI) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
        
        .stApp { background-color: #000000 !important; color: #ffffff; }
        
        /* Input Field */
        div[data-testid="stTextInput"] input { 
            background-color: #111 !important; color: #fff !important; 
            font-weight: bold !important; font-size: 1.2rem !important;
            border: 2px solid #00E5FF !important; border-radius: 10px;
        }

        /* Cards */
        .glass-card {
            background: rgba(20, 20, 20, 0.8); backdrop-filter: blur(10px);
            border-radius: 20px; border: 1px solid #333;
            padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0, 229, 255, 0.1);
        }
        
        /* Stat Box */
        .stat-box { 
            background: #0a0a0a; border-radius: 15px; padding: 15px; 
            text-align: center; border: 1px solid #222; 
        }
        .stat-val { font-size: 1.8rem; font-weight: 800; color: #fff; }
        .stat-lbl { color: #888; font-size: 0.9rem; text-transform: uppercase; }

        /* News Card */
        .news-card { 
            padding: 20px; margin-bottom: 15px; background: #111; 
            border-radius: 12px; border-left: 5px solid #888; 
            transition: all 0.3s ease;
        }
        .news-card:hover { transform: translateX(5px); background: #1a1a1a; }
        .nc-pos { border-left-color: #00E676; }
        .nc-neg { border-left-color: #FF1744; }
        .nc-neu { border-left-color: #FFD600; }

        /* Bitkub Badge */
        .bk-badge {
            background: #111; padding: 12px; border-radius: 12px; 
            border: 1px solid #333; margin-bottom: 8px;
            display: flex; justify-content: space-between; align-items: center;
        }

        /* Custom Tabs */
        button[data-baseweb="tab"] { font-size: 1rem !important; font-weight: 600 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. Functions ---

@st.cache_data(ttl=300)
def get_market_data(symbol, period, interval):
    try: return yf.Ticker(symbol).history(period=period, interval=interval)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try: return yf.Ticker(symbol).info
    except: return None

@st.cache_data(ttl=15)
def get_bitkub_ticker():
    try:
        r = requests.get("https://api.bitkub.com/api/market/ticker", timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

def clean_html(raw_html):
    return re.sub(re.compile('<.*?>'), '', raw_html)

def get_finnhub_news(symbol):
    try:
        to_date = datetime.date.today()
        from_date = to_date - datetime.timedelta(days=2)
        clean_sym = symbol.split("-")[0]
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_sym}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
        data = requests.get(url).json()
        return data[:5] if isinstance(data, list) else []
    except: return []

@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    news_list = []
    translator = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    
    # 1. Finnhub
    fh_news = get_finnhub_news(symbol)
    if fh_news:
        for i in fh_news:
            t, s, l = i.get('headline',''), i.get('summary',''), i.get('url','#')
            sc = TextBlob(t).sentiment.polarity
            icon = "🚀" if sc > 0.05 else "🔻" if sc < -0.05 else "⚖️"
            cls = "nc-pos" if sc > 0.05 else "nc-neg" if sc < -0.05 else "nc-neu"
            
            t_th, s_th = t, s
            if translator:
                try: 
                    t_th = translator.translate(t)
                    if s: s_th = translator.translate(s)
                except: pass
            
            news_list.append({'title': t_th, 'summary': s_th, 'link': l, 'icon': icon, 'class': cls, 'score': sc, 'source': 'Finnhub'})

    # 2. Google News (Backup)
    if len(news_list) < 3:
        try:
            cl_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
            q = urllib.parse.quote(f"site:bloomberg.com {cl_sym} market")
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            
            if not feed.entries:
                q = urllib.parse.quote(f"{cl_sym} finance news")
                feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")

            for i in feed.entries[:5]:
                t = i.title
                s = clean_html(getattr(i, 'summary', '') or getattr(i, 'description', ''))[:300]
                sc = TextBlob(t).sentiment.polarity
                icon = "🚀" if sc > 0.05 else "🔻" if sc < -0.05 else "⚖️"
                cls = "nc-pos" if sc > 0.05 else "nc-neg" if sc < -0.05 else "nc-neu"
                
                t_th, s_th = t, s
                if translator:
                    try: 
                        t_th = translator.translate(t)
                        if s: s_th = translator.translate(s)
                    except: pass
                
                news_list.append({'title': t_th, 'summary': s_th, 'link': i.link, 'icon': icon, 'class': cls, 'score': sc, 'source': 'Google'})
        except: pass
        
    return news_list[:10]

def calculate_technical_setup(df):
    try:
        # Full Series for Plotting
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        
        # Last Values for Logic
        close = df['Close'].iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        
        atr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        
        rsi_val = rsi_series.iloc[-1]

        if close > ema50 and ema50 > ema200: trend, sig, col, sc = "Uptrend (ขาขึ้น)", "BUY", "#00E676", 2
        elif close < ema50 and ema50 < ema200: trend, sig, col, sc = "Downtrend (ขาลง)", "SELL", "#FF1744", -2
        else: trend, sig, col, sc = "Sideways (ออกข้าง)", "WAIT", "#888", 0
        
        return {
            'trend': trend, 'signal': sig, 'color': col, 
            'rsi_series': rsi_series, 'rsi_val': rsi_val,
            'entry': close, 
            'sl': close-(1.5*atr) if sc>=0 else close+(1.5*atr), 
            'tp': close+(2.5*atr) if sc>=0 else close-(2.5*atr)
        }
    except: return None

def calculate_pivot_points(df):
    try:
        p = df.iloc[-2]
        pp = (p['High']+p['Low']+p['Close'])/3
        return {"PP":pp, "R1":(2*pp)-p['Low'], "S1":(2*pp)-p['High'], "R2":pp+(p['High']-p['Low']), "S2":pp-(p['High']-p['Low'])}
    except: return None

def calculate_dynamic_levels(df):
    try:
        sma = df['Close'].rolling(20).mean().iloc[-1]
        std = df['Close'].rolling(20).std().iloc[-1]
        return {
            "EMA 20": df['Close'].ewm(span=20).mean().iloc[-1],
            "EMA 50": df['Close'].ewm(span=50).mean().iloc[-1],
            "EMA 200": df['Close'].ewm(span=200).mean().iloc[-1],
            "BB Upper": sma+(2*std), "BB Lower": sma-(2*std), "Current": df['Close'].iloc[-1]
        }
    except: return None

def generate_dynamic_insight(price, pivots, dynamics):
    e200, e20 = dynamics['EMA 200'], dynamics['EMA 20']
    
    if price > e200:
        msg, col = ("Bullish Strong (แกร่งมาก)", "#00E676") if price > e20 else ("Bullish Retrace (ย่อตัว)", "#00E676")
    else:
        msg, col = ("Bearish Strong (ลงหนัก)", "#FF1744") if price < e20 else ("Bearish Correction (ดีดตัว)", "#FF1744")
    
    # Find nearest level
    all_lvls = {**pivots, **{k:v for k,v in dynamics.items() if k!='Current'}}
    n_name, n_price, min_d = "", 0, float('inf')
    for k,v in all_lvls.items():
        if abs(price-v) < min_d: min_d, n_name, n_price = abs(price-v), k, v
    
    dist_pct = (min_d / price) * 100
    act = f"⚠️ ราคากำลังทดสอบ **{n_name}**" if dist_pct < 0.8 else f"🏃 มีพื้นที่วิ่งไปหา **{n_name}**"
    return msg, col, act

def calculate_bitkub_ai_levels(h, l, c):
    pp = (h+l+c)/3
    rng = h-l
    mid = (h+l)/2
    st, col = ("BULLISH", "#00E676") if c > mid else ("BEARISH", "#FF1744")
    return {
        "levels": [
            {"name":"🚀 R2","price":pp+rng,"type":"res"}, {"name":"🛑 R1","price":(2*pp)-l,"type":"res"},
            {"name":"⚖️ PIVOT","price":pp,"type":"neu"},
            {"name":"🛡️ S1","price":(2*pp)-h,"type":"sup"}, {"name":"💎 S2","price":pp-rng,"type":"sup"}
        ],
        "fib": {"top": l+(rng*0.618), "bot": l+(rng*0.382)}, "status": st, "color": col
    }

def calculate_heikin_ashi(df):
    ha = df.copy()
    ha['Close'] = (df['Open']+df['High']+df['Low']+df['Close'])/4
    ha['Open'] = [ (df['Open'][0]+df['Close'][0])/2 ] + [0]*(len(df)-1)
    for i in range(1, len(df)): ha['Open'].iloc[i] = (ha['Open'].iloc[i-1]+ha['Close'].iloc[i-1])/2
    ha['High'] = ha[['High','Open','Close']].max(axis=1)
    ha['Low'] = ha[['Low','Open','Close']].min(axis=1)
    return ha

def gen_ai_verdict(setup, news):
    score = 50
    text = ""
    
    # Technical
    if setup['trend'] == "Uptrend (ขาขึ้น)": score += 20; text += "📈 กราฟเทคนิคเป็นขาขึ้น "
    elif setup['trend'] == "Downtrend (ขาลง)": score -= 20; text += "📉 กราฟเทคนิคเป็นขาลง "
    
    if setup['rsi_val'] > 70: score -= 5; text += "(RSI ตึงตัว ระวังแรงขาย) "
    elif setup['rsi_val'] < 30: score += 5; text += "(RSI ขายมากเกิน รอเด้ง) "
    
    # News
    n_score = sum([n['score'] for n in news]) if news else 0
    if n_score > 0.5: score += 15; text += "\n📰 ข่าวสารเป็นใจ สนับสนุนราคา"
    elif n_score < -0.5: score -= 15; text += "\n📰 มีข่าวลบกดดันตลาด"
    
    score = max(0, min(100, score))
    verd = "STRONG BUY" if score>=75 else "BUY" if score>=55 else "SELL" if score<=25 else "STRONG SELL" if score<=15 else "HOLD"
    return text, score, verd

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    if c1.button("BTC-USD"): set_symbol("BTC-USD")
    if c2.button("ETH-USD"): set_symbol("ETH-USD")
    c3, c4 = st.columns(2)
    if c3.button("GOLD"): set_symbol("GC=F")
    if c4.button("OIL"): set_symbol("CL=F")
    
    st.markdown("---")
    st.markdown("### 🇹🇭 Bitkub Live")
    bk = get_bitkub_ticker()
    if bk:
        b_p = bk.get('THB_BTC',{}).get('last',0)
        b_c = bk.get('THB_BTC',{}).get('percentChange',0)
        b_col = "#00E676" if b_c >= 0 else "#FF1744"
        
        e_p = bk.get('THB_ETH',{}).get('last',0)
        e_c = bk.get('THB_ETH',{}).get('percentChange',0)
        e_col = "#00E676" if e_c >= 0 else "#FF1744"
        
        st.markdown(f"""
        <div class='bk-badge'><span>BTC</span><span style='color:{b_col};font-weight:bold'>{b_p:,.0f}</span></div>
        <div class='bk-badge'><span>ETH</span><span style='color:{e_col};font-weight:bold'>{e_p:,.0f}</span></div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    chart_type = st.selectbox("Chart Style", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Timeframe", ["1mo","3mo","6mo","1y"], value="6mo")

# --- 5. Main Content ---
st.markdown("<h2 style='color:#00E5FF;'>🔍 Smart Search</h2>", unsafe_allow_html=True)
c1, c2 = st.columns([4,1])
with c1: sym_input = st.text_input("Symbol", st.session_state.symbol, label_visibility="collapsed")
with c2: 
    if st.button("วิเคราะห์ ⚡"): set_symbol(sym_input); st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner("🚀 AI กำลังประมวลผล..."):
        df = get_market_data(symbol, period, "1d")
    
    if not df.empty:
        curr = df['Close'].iloc[-1]
        chg = curr - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        color = "#00E676" if chg >= 0 else "#FF1744"
        
        setup = calculate_technical_setup(df)
        news = get_ai_analyzed_news_thai(symbol)
        info = get_stock_info(symbol)
        ai_txt, ai_sc, ai_vd = gen_ai_verdict(setup, news)
        
        # Determine score color
        if ai_sc >= 75: sc_color = "#00E676"
        elif ai_sc <= 25: sc_color = "#FF1744"
        else: sc_color = "#FFD600"

        # --- Hero Section ---
        st.markdown(f"""
        <div class="glass-card" style="border-top:5px solid {color};text-align:center;">
            <div style="font-size:4rem;font-weight:900;line-height:1;">{symbol}</div>
            <div style="font-size:3.5rem;color:{color};font-weight:bold;">{curr:,.2f}</div>
            <div style="background:{color}20;padding:5px 25px;border-radius:20px;display:inline-block;margin-top:10px;">
                <span style="color:{color};font-weight:bold;font-size:1.2rem;">{chg:+.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- TABS ---
        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R Dynamic", "🇹🇭 Bitkub AI"])

        # 1. Chart
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.05)
            
            # Candle / HA
            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['Open'], high=ha['High'], low=ha['Low'], close=ha['Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # EMA
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF', width=2), name="EMA50"), row=1, col=1)
            
            # RSI
            rsi_plot = setup['rsi_series'] if setup else [50]*len(df)
            fig.add_trace(go.Scatter(x=df.index, y=rsi_plot, line=dict(color='#E040FB', width=2), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            
            fig.update_layout(template='plotly_dark', height=600, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='#333')
            st.plotly_chart(fig, use_container_width=True)

        # 2. Stats
        with tabs[1]:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='stat-box'><div class='stat-lbl'>High</div><div class='stat-val' style='color:#00E676'>{df['High'].max():,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-box'><div class='stat-lbl'>Low</div><div class='stat-val' style='color:#FF1744'>{df['Low'].min():,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-box'><div class='stat-lbl'>Vol</div><div class='stat-val' style='color:#E040FB'>{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>", unsafe_allow_html=True)
            
            if info:
                st.markdown("---")
                st.markdown(f"**🏢 Sector:** {info.get('sector','N/A')} | **Industry:** {info.get('industry','N/A')}")
                
                c1, c2, c3 = st.columns(3)
                pe = info.get('trailingPE')
                eps = info.get('trailingEps')
                peg = info.get('pegRatio')
                
                pe_c = "#00E676" if pe and pe < 15 else "#FFD600" if pe and pe < 30 else "#FF1744"
                c1.markdown(f"<div class='fund-box' style='border-left:4px solid {pe_c}'><div class='fund-title'>P/E Ratio</div><div class='fund-val'>{pe if pe else 'N/A'}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='fund-box' style='border-left:4px solid #fff'><div class='fund-title'>EPS</div><div class='fund-val'>{eps if eps else 'N/A'}</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='fund-box' style='border-left:4px solid #fff'><div class='fund-title'>PEG Ratio</div><div class='fund-val'>{peg if peg else 'N/A'}</div></div>", unsafe_allow_html=True)

        # 3. AI News
        with tabs[2]:
            st.markdown("### 📰 Market Sentiment")
            if news:
                for n in news:
                    st.markdown(f"""
                    <div class="news-card {n['class']}">
                        <div style="display:flex;justify-content:space-between;">
                            <span style="font-size:0.8rem;background:#333;padding:2px 8px;border-radius:5px;">{n['source']}</span>
                            <span>{n['icon']}</span>
                        </div>
                        <h4 style="margin:10px 0;">{n['title']}</h4>
                        <p style="color:#aaa;font-size:0.9rem;">{n['summary']}</p>
                        <div style="text-align:right;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;text-decoration:none;">🔗 อ่านต่อ</a></div>
                    </div>
                    """, unsafe_allow_html=True)
            else: st.info("ไม่พบข่าว หรือ API ถูกจำกัด")

        # 4. Setup & Entry
        with tabs[3]:
            c_tech, c_entry = st.columns(2)
            with c_tech:
                st.markdown("### 🎯 Technical Status")
                if setup:
                    st.markdown(f"""
                    <div style="background:#111;padding:20px;border-radius:15px;border:1px solid {setup['color']};text-align:center;">
                        <h2 style="color:{setup['color']};margin:0;">{setup['signal']}</h2>
                        <p style="font-size:1.2rem;margin-top:5px;">{setup['trend']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with c_entry:
                st.markdown("### 💰 Entry Zones")
                st.markdown(f"""
                <div style="display:flex;flex-direction:column;gap:10px;">
                    <div style="background:#1a1a1a;padding:15px;border-radius:10px;border-left:5px solid #00E5FF;">
                        <span style="color:#aaa;">Probe Buy (20%)</span><br>
                        <span style="font-size:1.2rem;font-weight:bold;">{curr*0.99:,.2f}</span>
                    </div>
                    <div style="background:#1a1a1a;padding:15px;border-radius:10px;border-left:5px solid #FFD600;">
                        <span style="color:#aaa;">Accumulate (30%)</span><br>
                        <span style="font-size:1.2rem;font-weight:bold;">{curr*0.97:,.2f}</span>
                    </div>
                    <div style="background:#1a1a1a;padding:15px;border-radius:10px;border-left:5px solid #FF1744;">
                        <span style="color:#aaa;">Sniper (50%)</span><br>
                        <span style="font-size:1.2rem;font-weight:bold;">{curr*0.94:,.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 5. AI Verdict
        with tabs[4]:
            st.markdown(f"""
            <div class="ai-card" style="border-color:{sc_color};">
                <div class="ai-score-circle" style="border-color:{sc_color};color:{sc_color};">{ai_sc}</div>
                <h2 style="color:{sc_color};">{ai_vd}</h2>
                <p style="font-size:1.1rem;line-height:1.6;">{ai_txt}</p>
            </div>
            """, unsafe_allow_html=True)

        # 6. S/R Dynamics
        with tabs[5]:
            pivots = calculate_pivot_points(df)
            dynamic = calculate_dynamic_levels(df)
            
            if pivots and dynamic:
                t_msg, t_col, a_msg = generate_dynamic_insight(curr, pivots, dynamic)
                
                st.markdown(f"""
                <div style="background:#111; border:1px solid {t_col}; padding:20px; border-radius:15px; margin-bottom:20px;">
                    <h3 style="color:{t_col}; margin-top:0;">🧠 AI Insight: {t_msg}</h3>
                    <p style="font-size:1rem;color:#ccc;">{a_msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🧱 Static Pivots")
                    for k, v in pivots.items():
                        cl = "#FF1744" if "R" in k else "#00E676" if "S" in k else "#FFD600"
                        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:10px;background:#161616;border-left:4px solid {cl};margin-bottom:5px;'><b>{k}</b><span>{v:,.2f}</span></div>", unsafe_allow_html=True)
                
                with c2:
                    st.markdown("#### 🌊 Dynamic Levels")
                    for k, v in dynamic.items():
                        if k == "Current": continue
                        dist = ((curr - v) / v) * 100
                        cl = "#00E676" if curr > v else "#FF1744"
                        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:10px;background:#161616;border-left:4px solid {cl};margin-bottom:5px;'><div>{k}</div><div style='text-align:right;'>{v:,.2f}<br><span style='font-size:0.8rem;color:{cl}'>{dist:+.2f}%</span></div></div>", unsafe_allow_html=True)

        # 7. Bitkub AI
        with tabs[6]:
            st.markdown("### 🇹🇭 Bitkub AI Analysis")
            bk_sel = st.radio("เลือกเหรียญ:", ["BTC", "ETH"], horizontal=True)
            
            if bk_data:
                pair = f"THB_{bk_sel}"
                coin = bk_data.get(pair, {})
                if coin:
                    last_thb = coin.get('last', 0)
                    h24, l24 = coin.get('high24hr', 0), coin.get('low24hr', 0)
                    ai_bk = calculate_bitkub_ai_levels(h24, l24, last_thb)
                    
                    st.markdown(f"""
                    <div style="text-align:center;padding:25px;background:#111;border-radius:20px;border:2px solid {ai_bk['color']};margin-bottom:20px;">
                        <div style="color:#aaa;">Price (THB)</div>
                        <div style="font-size:3rem;font-weight:bold;color:#fff;">{last_thb:,.0f}</div>
                        <div style="font-size:1.5rem;font-weight:bold;color:{ai_bk['color']};">{ai_bk['status']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### 🤖 Intraday Pivots")
                        for l in ai_bk['levels']:
                            cl = "#00E676" if l['type']=='sup' else "#FF1744" if l['type']=='res' else "#FFD600"
                            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:12px;background:#161616;border-left:5px solid {cl};margin-bottom:5px;border-radius:5px;'><span style='font-weight:bold;color:{cl}'>{l['name']}</span><span>{l['price']:,.0f}</span></div>", unsafe_allow_html=True)
                    
                    with c2:
                        st.markdown("#### 📐 Golden Zone (24H)")
                        st.info(f"**Bottom (รับ):** {ai_bk['fib']['bot']:,.0f}\n\n**Top (ต้าน):** {ai_bk['fib']['top']:,.0f}")
                        st.caption("โซนรับต้านจิตวิทยา คำนวณจาก High/Low ใน 24 ชั่วโมงล่าสุดของกระดานไทย")
                else: st.error("ไม่พบข้อมูลเหรียญนี้")
            else: st.warning("กำลังเชื่อมต่อ Bitkub API...")

    else: st.error("❌ ไม่พบข้อมูลหุ้น/เหรียญนี้")
