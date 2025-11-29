import streamlit as st
import yfinance as yf
import pandas as pd
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
try: from deep_translator import GoogleTranslator; HAS_TRANS=True
except: HAS_TRANS=False
try: nltk.data.find('tokenizers/punkt')
except: nltk.download('punkt')

FINNHUB_KEY = "d4l5ku1r01qt7v18ll40d4l5ku1r01qt7v18ll4g" 

st.set_page_config(page_title="Smart Trader AI : Ultra Black", layout="wide", page_icon="💎")
if 'symbol' not in st.session_state: st.session_state.symbol = 'BTC-USD'
def set_symbol(sym): st.session_state.symbol = sym

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Kanit',sans-serif;}
.stApp{background-color:#000000!important;color:#ffffff;}
div[data-testid="stTextInput"] input{background-color:#ffffff!important;color:#000000!important;font-weight:700!important;font-size:1.5rem!important;height:60px!important;border:3px solid #00E5FF!important;border-radius:15px!important;}
.glass-card{background:rgba(20,20,20,0.6);backdrop-filter:blur(10px);border-radius:25px;border:1px solid rgba(255,255,255,0.15);padding:35px;margin-bottom:30px;box-shadow:0 0 20px rgba(255,255,255,0.05);}
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

def clean_html(raw_html): return re.sub(re.compile('<.*?>'), '', raw_html)

def get_finnhub_news(symbol):
    try:
        to_date = datetime.date.today()
        from_date = to_date - datetime.timedelta(days=2)
        sym = symbol.split("-")[0]
        url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
        return requests.get(url).json()[:5]
    except: return []

@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    news_list = []
    trans = GoogleTranslator(source='auto', target='th') if HAS_TRANS else None
    
    fh_news = get_finnhub_news(symbol)
    if fh_news and isinstance(fh_news, list):
        for i in fh_news:
            t, s, l = i.get('headline',''), i.get('summary',''), i.get('url','#')
            sc = TextBlob(t).sentiment.polarity
            icon = "🚀" if sc>0.05 else "🔻" if sc<-0.05 else "⚖️"
            cls = "nc-pos" if sc>0.05 else "nc-neg" if sc<-0.05 else "nc-neu"
            if trans:
                try: t = trans.translate(t); s = trans.translate(s) if s else ""
                except: pass
            news_list.append({'title_th':t, 'summary_th':s, 'link':l, 'icon':icon, 'class':cls, 'score':sc, 'source':'Finnhub'})

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
                s = clean_html(getattr(i,'summary','') or getattr(i,'description',''))[:300]
                sc = TextBlob(t).sentiment.polarity
                icon = "🚀" if sc>0.05 else "🔻" if sc<-0.05 else "⚖️"
                cls = "nc-pos" if sc>0.05 else "nc-neg" if sc<-0.05 else "nc-neu"
                if trans:
                    try: t = trans.translate(t); s = trans.translate(s) if s else ""
                    except: pass
                news_list.append({'title_th':t, 'summary_th':s, 'link':i.link, 'icon':icon, 'class':cls, 'score':sc, 'source':'Google'})
        except: pass
    return news_list[:10]

def calculate_technical_setup(df):
    try:
        c = df['Close'].iloc[-1]
        e50 = df['Close'].ewm(span=50).mean().iloc[-1]
        e200 = df['Close'].ewm(span=200).mean().iloc[-1]
        atr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        
        delta = df['Close'].diff()
        gain = (delta.where(delta>0,0)).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        rsi_s = 100 - (100/(1+(gain/loss)))
        
        if c > e50 and e50 > e200: t,s,co,sc = "Uptrend", "BUY", "#00E676", 2
        elif c < e50 and e50 < e200: t,s,co,sc = "Downtrend", "SELL", "#FF1744", -2
        else: t,s,co,sc = "Sideways", "WAIT", "#888", 0
        
        return {'trend':t, 'signal':s, 'color':co, 'rsi':rsi_s, 'rsi_val':rsi_s.iloc[-1], 'entry':c, 'sl':c-(1.5*atr) if sc>=0 else c+(1.5*atr), 'tp':c+(2.5*atr) if sc>=0 else c-(2.5*atr)}
    except: return None

def get_levels(df):
    lvls = []
    for i in range(5, len(df)-5):
        if df['Low'][i] == df['Low'][i-5:i+6].min(): lvls.append(df['Low'][i])
        elif df['High'][i] == df['High'][i-5:i+6].max(): lvls.append(df['High'][i])
    return sorted(list(set([l for l in lvls if abs(l - df['Close'].iloc[-1])/df['Close'].iloc[-1] < 0.15])))

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
    msg, col = ("Bullish Strong", "#00E676") if price > e200 and price > e20 else ("Bullish Retrace", "#00E676") if price > e200 else ("Bearish Strong", "#FF1744") if price < e20 else ("Bearish Correction", "#FF1744")
    
    all_lvls = {**pivots, **{k:v for k,v in dynamics.items() if k!='Current'}}
    n_name, n_price, min_d = "", 0, float('inf')
    for k,v in all_lvls.items():
        if abs(price-v) < min_d: min_d, n_name, n_price = abs(price-v), k, v
    
    act = f"⚠️ Testing **{n_name}**" if (min_d/price)*100 < 0.8 else f"🏃 Room to run to **{n_name}**"
    return msg, col, act

def calculate_bitkub_ai_levels(h, l, c):
    pp = (h+l+c)/3; rng = h-l
    st, col = ("BULLISH", "#00E676") if c > (h+l)/2 else ("BEARISH", "#FF1744")
    return {
        "levels": [{"name":"R2","price":pp+rng,"type":"res"}, {"name":"R1","price":(2*pp)-l,"type":"res"}, {"name":"PIVOT","price":pp,"type":"neu"}, {"name":"S1","price":(2*pp)-h,"type":"sup"}, {"name":"S2","price":pp-rng,"type":"sup"}],
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
    if setup['trend'] == "Uptrend": score += 20; text += "📈 เทคนิคขาขึ้น "
    elif setup['trend'] == "Downtrend": score -= 20; text += "📉 เทคนิคขาลง "
    
    if setup['rsi_val'] > 70: score -= 5; text += "(RSI Overbought) "
    elif setup['rsi_val'] < 30: score += 5; text += "(RSI Oversold) "
    
    n_score = sum([n['score'] for n in news]) if news else 0
    if n_score > 0.5: score += 15; text += "\n📰 ข่าวบวก"
    elif n_score < -0.5: score -= 15; text += "\n📰 ข่าวลบ"
    
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
        st.markdown(f"**BTC:** {bk.get('THB_BTC',{}).get('last',0):,.0f} THB")
        st.markdown(f"**ETH:** {bk.get('THB_ETH',{}).get('last',0):,.0f} THB")
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
        
        # FIX: ประกาศตัวแปรสีให้ครบถ้วน
        sc_col = "#00E676" if ai_sc>=70 else "#FF1744" if ai_sc<=30 else "#FFD600"

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

        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R", "🇹🇭 Bitkub"])

        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2962FF'), name="EMA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=setup['rsi'] if setup else [50]*len(df), line=dict(color='#AA00FF'), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            fig.update_layout(template='plotly_dark', height=600, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            c1,c2,c3 = st.columns(3)
            c1.markdown(f"<div class='stat-box'><div>HIGH</div><div style='color:#00E676'>{df['High'].max():,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-box'><div>LOW</div><div style='color:#FF1744'>{df['Low'].min():,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-box'><div>VOL</div><div style='color:#E040FB'>{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>", unsafe_allow_html=True)
            if info:
                pe, peg = info.get('trailingPE', 0), info.get('pegRatio', 0)
                st.markdown("---")
                c1, c2 = st.columns(2)
                c1.metric("P/E", f"{pe:.2f}")
                c2.metric("PEG", f"{peg:.2f}")

        with tabs[2]:
            if news:
                for n in news:
                    st.markdown(f"""
                    <div class="news-card {n['class']}">
                        <div style="display:flex;justify-content:space-between;">
                            <span>{n['icon']} {n['source']}</span>
                        </div>
                        <h4>{n['title_th']}</h4>
                        <div style="color:#aaa;font-size:0.9rem;">{n['summary_th']}</div>
                        <div style="margin-top:10px;text-align:right;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;">อ่านต่อ...</a></div>
                    </div>
                    """, unsafe_allow_html=True)
            else: st.info("No News")

        with tabs[3]:
            if setup:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;border:2px solid {setup['color']}">
                    <h1 style="color:{setup['color']}">{setup['signal']}</h1>
                    <h3>{setup['trend']}</h3>
                    <p>Entry: {setup['entry']:.2f} | TP: {setup['tp']:.2f} | SL: {setup['sl']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Probe:** {curr*0.99:.2f} | **Accumulate:** {curr*0.97:.2f} | **Sniper:** {curr*0.94:.2f}")

        with tabs[4]:
            st.markdown(f"""
            <div class="ai-card">
                <div class="ai-score-circle">{ai_sc}</div>
                <h2 style="color:{sc_col}">{ai_vd}</h2>
                <p>{ai_txt}</p>
            </div>
            """, unsafe_allow_html=True)

        with tabs[5]:
            res = sorted([l for l in levels if l > curr])[:3]
            sup = sorted([l for l in levels if l < curr], reverse=True)[:3]
            st.write("RES:", [f"{r:,.2f}" for r in reversed(res)])
            st.write("SUP:", [f"{s:,.2f}" for s in sup])
            
            pivots = calculate_pivot_points(df)
            dyn = calculate_dynamic_levels(df)
            if pivots and dyn:
                t_msg, t_col, a_msg = generate_dynamic_insight(curr, pivots, dyn)
                st.markdown(f"""<div style="background:#111;border:1px solid {t_col};padding:20px;border-radius:15px;margin-top:20px;"><h3 style="color:{t_col};margin:0;">🧠 AI Insight: {t_msg}</h3><p>{a_msg}</p></div>""", unsafe_allow_html=True)

        with tabs[6]:
            bk_sel = st.radio("Bitkub Coin", ["BTC","ETH"], horizontal=True)
            if bk:
                pair = f"THB_{bk_sel}"
                d = bk.get(pair, {})
                last, h24, l24 = d.get('last',0), d.get('high24hr',0), d.get('low24hr',0)
                ai_bk = calculate_bitkub_ai_levels(h24, l24, last)
                st.markdown(f"""<div style="text-align:center;padding:20px;background:#111;border-radius:20px;border:2px solid {ai_bk['color']};margin-bottom:20px;"><div style="font-size:3rem;font-weight:bold;color:#fff;">{last:,.0f}</div><div style="font-size:1.5rem;font-weight:bold;color:{ai_bk['color']};">{ai_bk['status']}</div></div>""", unsafe_allow_html=True)
                for l in ai_bk['levels']:
                    cl = "#00E676" if l['type']=='sup' else "#FF1744" if l['type']=='res' else "#FFD600"
                    st.markdown(f"<div style='display:flex;justify-content:space-between;padding:10px;border-left:5px solid {cl};background:#161616;margin-bottom:5px;'><b>{l['name']}</b><span>{l['price']:,.0f}</span></div>", unsafe_allow_html=True)
                st.info(f"Golden Top: {ai_bk['fib']['top']:,.0f} | Bot: {ai_bk['fib']['bot']:,.0f}")
    else: st.error("ไม่พบข้อมูล")
