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
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

FINNHUB_KEY = "d4l5ku1r01qt7v18ll40d4l5ku1r01qt7v18ll4g" 

# --- Setup ---
st.set_page_config(page_title="Smart Trader AI : Ultra Black", layout="wide", page_icon="💎")

if 'symbol' not in st.session_state: st.session_state.symbol = 'BTC-USD'
if 'article_url' not in st.session_state: st.session_state.article_url = None

def set_symbol(sym): st.session_state.symbol = sym

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Kanit',sans-serif;}
.stApp{background-color:#000000!important;color:#ffffff;}
div[data-testid="stTextInput"] input{background-color:#ffffff!important;color:#000000!important;font-weight:700!important;font-size:1.5rem!important;height:60px!important;border:3px solid #00E5FF!important;border-radius:15px!important;}
.glass-card{background:rgba(20,20,20,0.6);backdrop-filter:blur(10px);border-radius:25px;border:1px solid rgba(255,255,255,0.15);padding:35px;margin-bottom:30px;box-shadow:0 0 20px rgba(255,255,255,0.05);}
.sr-container{display:flex;flex-direction:column;gap:10px;margin-bottom:20px;}
.sr-row{display:flex;justify-content:space-between;align-items:center;padding:15px 25px;border-radius:15px;font-size:1.5rem;font-weight:bold;}
.res-row{background:linear-gradient(90deg,rgba(255,23,68,0.1),rgba(0,0,0,0));border-left:8px solid #FF1744;color:#FF1744;}
.sup-row{background:linear-gradient(90deg,rgba(0,230,118,0.1),rgba(0,0,0,0));border-left:8px solid #00E676;color:#00E676;}
.curr-row{background:#222;border:1px solid #555;color:#fff;justify-content:center;font-size:1.8rem;}
.stat-box{background:#0a0a0a;border-radius:20px;padding:25px;text-align:center;border:1px solid #333;margin-bottom:15px;}
.fund-box{background:#111;border:1px solid #444;border-radius:15px;padding:20px;margin-bottom:10px;}
.news-card{padding:25px;margin-bottom:15px;background:#111;border-radius:15px;border-left:6px solid #888;transition:transform 0.2s;}
.news-card:hover{transform:translateY(-2px);background:#161616;}
.nc-pos{border-left-color:#00E676;} .nc-neg{border-left-color:#FF1744;} .nc-neu{border-left-color:#FFD600;}
.ai-card{background:linear-gradient(145deg,#111,#0d0d0d);border:2px solid #00E5FF;border-radius:20px;padding:30px;text-align:center;}
.ai-score-circle{width:100px;height:100px;border-radius:50%;border:5px solid #00E5FF;display:flex;align-items:center;justify-content:center;font-size:2.5rem;font-weight:bold;color:#00E5FF;margin:0 auto 20px auto;}
div.stButton>button{font-size:1.1rem!important;padding:10px 20px!important;border-radius:10px!important;background:#111;border:1px solid #333;color:#fff;width:100%;}
div.stButton>button:hover{background:#00E5FF;color:#000!important;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# --- Functions ---
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
        sym = symbol.split("-")[0]
        url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
        r = requests.get(url)
        data = r.json()
        return data[:5] if isinstance(data, list) and len(data) > 0 else []
    except: return []

@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    news_list = []
    translator = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    
    # Finnhub
    fh_news = get_finnhub_news(symbol)
    if fh_news:
        for i in fh_news:
            title, summary, link = i.get('headline',''), i.get('summary',''), i.get('url','#')
            score = TextBlob(title).sentiment.polarity
            icon = "🚀" if score > 0.05 else "🔻" if score < -0.05 else "⚖️"
            cls = "nc-pos" if score > 0.05 else "nc-neg" if score < -0.05 else "nc-neu"
            title_th, summary_th = title, summary
            if translator:
                try:
                    title_th = translator.translate(title)
                    if summary: summary_th = translator.translate(summary)
                except: pass
            news_list.append({'title_th': title_th, 'summary_th': summary_th, 'link': link, 'icon': icon, 'class': cls, 'score': score, 'source': 'Finnhub'})

    # Google News
    if len(news_list) < 3:
        try:
            cl_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
            q = urllib.parse.quote(f"site:bloomberg.com {cl_sym} market")
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            if not feed.entries:
                q = urllib.parse.quote(f"{cl_sym} finance news")
                feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            
            for i in feed.entries[:5]:
                title = i.title
                summary = clean_html(getattr(i, 'summary', '') or getattr(i, 'description', ''))[:300]
                score = TextBlob(title).sentiment.polarity
                icon = "🚀" if score > 0.05 else "🔻" if score < -0.05 else "⚖️"
                cls = "nc-pos" if score > 0.05 else "nc-neg" if score < -0.05 else "nc-neu"
                title_th, summary_th = title, summary
                if translator:
                    try:
                        title_th = translator.translate(title)
                        if summary: summary_th = translator.translate(summary)
                    except: pass
                news_list.append({'title_th': title_th, 'summary_th': summary_th, 'link': i.link, 'icon': icon, 'class': cls, 'score': score, 'source': 'Google'})
        except: pass
    return news_list[:10]

@st.cache_data(ttl=3600)
def fetch_full_article(url):
    if not HAS_NEWSPAPER: return "Error: Install newspaper3k", ""
    try:
        art = Article(url)
        art.download()
        art.parse()
        title, text = art.title, art.text
        if HAS_TRANSLATOR and text:
            trans = GoogleTranslator(source='auto', target='th')
            try:
                title = trans.translate(title)
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                text = "\n\n".join([trans.translate(c) for c in chunks])
            except: pass
        return title, text
    except: return "Cannot fetch article", "เนื้อหาถูกป้องกันหรือเข้าถึงไม่ได้"

def calculate_technical_setup(df):
    try:
        close = df['Close'].iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        if close > ema50 and ema50 > ema200: trend, sig, col, sc = "Uptrend", "BUY", "#00E676", 2
        elif close < ema50 and ema50 < ema200: trend, sig, col, sc = "Downtrend", "SELL", "#FF1744", -2
        else: trend, sig, col, sc = "Sideways", "WAIT", "#888", 0
        
        return {'trend': trend, 'signal': sig, 'color': col, 'rsi': rsi, 'entry': close, 'sl': close-(1.5*atr) if sc>=0 else close+(1.5*atr), 'tp': close+(2.5*atr) if sc>=0 else close-(2.5*atr)}
    except: return None

def get_levels(df):
    levels = []
    for i in range(5, len(df)-5):
        if df['Low'][i] == df['Low'][i-5:i+6].min(): levels.append(df['Low'][i])
        elif df['High'][i] == df['High'][i-5:i+6].max(): levels.append(df['High'][i])
    return sorted(list(set([l for l in levels if abs(l - df['Close'].iloc[-1])/df['Close'].iloc[-1] < 0.15]))) # Filter near price

def gen_ai_verdict(setup, news):
    score = 50
    text = ""
    if setup['trend'] == "Uptrend": score += 20; text += "📈 เทคนิคขาขึ้นแข็งแกร่ง "
    elif setup['trend'] == "Downtrend": score -= 20; text += "📉 เทคนิคขาลงชัดเจน "
    else: text += "⚖️ ตลาดไซด์เวย์ "
    
    if setup['rsi'] > 70: score -= 5; text += "(RSI Overbought ระวังแรงขาย) "
    elif setup['rsi'] < 30: score += 5; text += "(RSI Oversold รอเด้ง) "
    
    n_score = sum([n['score'] for n in news]) if news else 0
    if n_score > 0.5: score += 15; text += "\n📰 ข่าวสารเป็นบวกสนับสนุน"
    elif n_score < -0.5: score -= 15; text += "\n📰 ข่าวสารเชิงลบกดดัน"
    
    score = max(0, min(100, score))
    verd = "STRONG BUY" if score>=75 else "BUY" if score>=55 else "SELL" if score<=25 else "STRONG SELL" if score<=15 else "HOLD"
    return text, score, verd

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 💎 ULTRA MENU")
    if st.button("BTC-USD"): set_symbol("BTC-USD")
    if st.button("ETH-USD"): set_symbol("ETH-USD")
    if st.button("Gold"): set_symbol("GC=F")
    if st.button("Oil"): set_symbol("CL=F")
    st.markdown("---")
    st.markdown("### 🇹🇭 Bitkub Rates")
    bk = get_bitkub_ticker()
    if bk:
        b_thb = bk.get('THB_BTC',{}).get('last',0)
        e_thb = bk.get('THB_ETH',{}).get('last',0)
        st.markdown(f"**BTC:** {b_thb:,.0f} THB")
        st.markdown(f"**ETH:** {e_thb:,.0f} THB")
    
    period = st.select_slider("Timeframe", ["1mo","3mo","6mo","1y"], value="6mo")

# --- Main ---
st.markdown("<h2 style='color:#00E5FF;'>🔍 Smart Search</h2>", unsafe_allow_html=True)
c1, c2 = st.columns([4,1])
with c1: sym_input = st.text_input("Symbol", st.session_state.symbol, label_visibility="collapsed")
with c2: 
    if st.button("วิเคราะห์ ⚡"): set_symbol(sym_input); st.rerun()

symbol = st.session_state.symbol.upper()
if symbol:
    df = get_market_data(symbol, period, "1d")
    if not df.empty:
        curr = df['Close'].iloc[-1]
        chg = curr - df['Close'].iloc[-2]
        pct = chg / df['Close'].iloc[-2] * 100
        color = "#00E676" if chg >= 0 else "#FF1744"
        
        setup = calculate_technical_setup(df)
        news = get_ai_analyzed_news_thai(symbol)
        levels = get_levels(df)
        info = get_stock_info(symbol)
        ai_txt, ai_sc, ai_vd = gen_ai_verdict(setup, news)

        # Header
        st.markdown(f"""
        <div class="glass-card" style="border-top:5px solid {color};text-align:center;">
            <div style="font-size:4rem;font-weight:900;">{symbol}</div>
            <div style="font-size:3rem;color:{color};">{curr:,.2f}</div>
            <div style="background:{color}20;padding:5px 20px;border-radius:20px;display:inline-block;">
                <span style="color:{color};font-weight:bold;">{chg:+.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "📖 Full Reader", "🎯 Setup", "🤖 Verdict", "🛡️ S/R", "🇹🇭 Bitkub AI"])

        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2962FF'), name="EMA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=setup['rsi'] if setup else [50]*len(df), line=dict(color='#AA00FF'), name="RSI"), row=2, col=1)
            fig.update_layout(template='plotly_dark', height=600, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            c1,c2,c3 = st.columns(3)
            c1.markdown(f"<div class='stat-box'><div>HIGH</div><div style='color:#00E676'>{df['High'].max():,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-box'><div>LOW</div><div style='color:#FF1744'>{df['Low'].min():,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-box'><div>VOL</div><div style='color:#E040FB'>{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>", unsafe_allow_html=True)
            if info:
                pe = info.get('trailingPE', 0)
                peg = info.get('pegRatio', 0)
                st.markdown("---")
                c1, c2 = st.columns(2)
                c1.metric("P/E Ratio", f"{pe:.2f}")
                c2.metric("PEG Ratio", f"{peg:.2f}")

        with tabs[2]: # AI News
            if news:
                for n in news:
                    st.markdown(f"""
                    <div class="news-card {n['class']}">
                        <div style="display:flex;justify-content:space-between;">
                            <span>{n['icon']} {n['source']}</span>
                        </div>
                        <h4>{n['title_th']}</h4>
                        <div style="color:#aaa;font-size:0.9rem;">{n['summary_th']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"📖 อ่านฉบับเต็ม: {n['title_th'][:20]}...", key=n['link']):
                        st.session_state.article_url = n['link']
                        st.success("ส่งไปที่แท็บ Full Reader แล้ว!")
            else: st.info("No News")

        with tabs[3]: # Full Reader
            if st.session_state.article_url:
                with st.spinner("แกะข่าว..."):
                    t_th, b_th = fetch_full_article(st.session_state.article_url)
                    st.markdown(f"### {t_th}")
                    st.write(b_th)
            else: st.warning("เลือกข่าวจากแท็บ AI News ก่อน")

        with tabs[4]: # Setup
            if setup:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;border:2px solid {setup['color']}">
                    <h1 style="color:{setup['color']}">{setup['signal']}</h1>
                    <h3>{setup['trend']}</h3>
                    <p>Entry: {setup['entry']:.2f} | TP: {setup['tp']:.2f} | SL: {setup['sl']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

        with tabs[5]: # Verdict
            st.markdown(f"""
            <div class="ai-card">
                <div class="ai-score-circle">{ai_sc}</div>
                <h2 style="color:{score_color}">{ai_vd}</h2>
                <p>{ai_txt}</p>
            </div>
            """, unsafe_allow_html=True)

        with tabs[6]: # S/R
            res = sorted([l for l in levels if l > curr])[:3]
            sup = sorted([l for l in levels if l < curr], reverse=True)[:3]
            st.write("RESISTANCE (ต้าน):", [f"{r:,.2f}" for r in reversed(res)])
            st.write("SUPPORT (รับ):", [f"{s:,.2f}" for s in sup])

        with tabs[7]: # Bitkub AI
            bk_sel = st.radio("Bitkub Coin", ["BTC","ETH"], horizontal=True)
            if bk:
                pair = f"THB_{bk_sel}"
                d = bk.get(pair, {})
                last, h24, l24 = d.get('last',0), d.get('high24hr',0), d.get('low24hr',0)
                pivot = (h24+l24+last)/3
                st.markdown(f"### Price: {last:,.0f} THB")
                st.write(f"Pivot: {pivot:,.0f}")
                st.write(f"R1: {(2*pivot)-l24:,.0f} | S1: {(2*pivot)-h24:,.0f}")
    else: st.error("ไม่พบข้อมูล")
