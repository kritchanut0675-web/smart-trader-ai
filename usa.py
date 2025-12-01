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

# --- 1. Setup & Config ---
st.set_page_config(
    page_title="Smart Trader AI : Ultra Black",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- 2. CSS Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
        .stApp { background-color: #050505 !important; color: #e0e0e0; }
        
        /* Input & Select */
        div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div > div { 
            background-color: #111 !important; color: #fff !important; 
            border: 2px solid #00E5FF !important; border-radius: 10px;
        }
        
        /* Cards */
        .glass-card {
            background: linear-gradient(145deg, #1a1a1a, #0d0d0d);
            border: 1px solid #333; border-radius: 20px;
            padding: 25px; margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        
        /* Metrics */
        .metric-box {
            background: #111; border-radius: 15px; padding: 20px;
            border-left: 4px solid #333; transition: transform 0.2s;
        }
        .metric-box:hover { transform: translateY(-5px); border-left-color: #00E5FF; }
        .metric-val { font-size: 1.8rem; font-weight: 800; color: #fff; margin-top: 5px; }
        .metric-lbl { font-size: 0.9rem; color: #888; text-transform: uppercase; }

        /* AI Insight */
        .ai-insight-box {
            background: linear-gradient(135deg, #111, #0a0a0a);
            border: 1px solid #333; border-radius: 15px; padding: 25px;
            position: relative; overflow: hidden; margin-bottom: 20px;
        }
        
        /* Verdict Ring */
        .verdict-ring {
            width: 140px; height: 140px; border-radius: 50%;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            font-size: 3rem; font-weight: 900; margin: 0 auto 20px;
            border: 8px solid #333; background: #000;
            box-shadow: 0 0 40px rgba(0,0,0,0.5);
        }
        
        /* News */
        .news-card { padding: 20px; margin-bottom: 15px; background: #111; border-radius: 15px; border-left: 5px solid #888; }
        .nc-pos { border-left-color: #00E676; } .nc-neg { border-left-color: #FF1744; } .nc-neu { border-left-color: #FFD600; }
        
        /* S/R Cards */
        .sr-card { padding: 12px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #333; background: #161616; }
        .sr-res { border-left: 5px solid #FF1744; } .sr-sup { border-left: 5px solid #00E676; } .sr-piv { border-left: 5px solid #FFD600; }
        
        /* Guru */
        .guru-card { background: #111; padding: 15px; border-radius: 12px; border: 1px solid #333; margin-bottom: 10px; }
        
        /* Buttons */
        div.stButton > button { width: 100%; background: linear-gradient(45deg, #00E5FF, #2979FF); border: none; color: #000; font-weight: bold; padding: 12px; border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. Functions (Core Logic) ---

@st.cache_data(ttl=300)
def get_market_data(symbol, period, interval):
    try: return yf.Ticker(symbol).history(period=period, interval=interval)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try: return yf.Ticker(symbol).info
    except: return None

@st.cache_data(ttl=15)
def get_bitkub_ticker(sym=None):
    try:
        r = requests.get("https://api.bitkub.com/api/market/ticker", timeout=5)
        data = r.json()
        if sym: return data.get(f"THB_{sym}")
        return data
    except: return None

def get_finnhub_news(symbol):
    try:
        t = datetime.date.today(); f = t - datetime.timedelta(days=2)
        s = symbol.split("-")[0]
        url = f"https://finnhub.io/api/v1/company-news?symbol={s}&from={f}&to={t}&token={FINNHUB_KEY}"
        return requests.get(url).json()[:5]
    except: return []

@st.cache_data(ttl=3600)
def get_news_analysis(symbol):
    lst = []
    tr = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    
    # Finnhub
    fh = get_finnhub_news(symbol)
    if fh:
        for i in fh:
            t, s, l = i.get('headline',''), i.get('summary',''), i.get('url','#')
            sc = TextBlob(t).sentiment.polarity
            if sc > 0.05: lbl, ic, cl = "ข่าวดี", "🚀", "nc-pos"
            elif sc < -0.05: lbl, ic, cl = "ข่าวร้าย", "🔻", "nc-neg"
            else: lbl, ic, cl = "ทั่วไป", "⚖️", "nc-neu"
            if tr: 
                try: t=tr.translate(t); s=tr.translate(s) if s else ""
                except: pass
            lst.append({'title':t, 'summary':s, 'link':l, 'icon':ic, 'class':cl, 'label':lbl, 'score':sc, 'source':'Finnhub'})
            
    # Google News Backup
    if len(lst) < 3:
        try:
            cl = symbol.replace("-THB","").replace("-USD","").replace("=F","")
            q = urllib.parse.quote(f"site:bloomberg.com {cl} market")
            fd = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            for i in fd.entries[:5]:
                t = i.title
                s = re.sub(re.compile('<.*?>'), '', getattr(i,'summary','') or getattr(i,'description',''))[:300]
                sc = TextBlob(t).sentiment.polarity
                if sc > 0.05: lbl, ic, cl = "ข่าวดี", "🚀", "nc-pos"
                elif sc < -0.05: lbl, ic, cl = "ข่าวร้าย", "🔻", "nc-neg"
                else: lbl, ic, cl = "ทั่วไป", "⚖️", "nc-neu"
                if tr:
                    try: t=tr.translate(t); s=tr.translate(s) if s else ""
                    except: pass
                lst.append({'title':t, 'summary':s, 'link':i.link, 'icon':ic, 'class':cl, 'label':lbl, 'score':sc, 'source':'Google'})
        except: pass
    return lst[:10]

def calc_technical(df):
    try:
        c = df['Close'].iloc[-1]
        e50 = df['Close'].ewm(span=50).mean().iloc[-1]
        e200 = df['Close'].ewm(span=200).mean().iloc[-1]
        atr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        delta = df['Close'].diff()
        rsi_s = 100 - (100/(1 + (delta.where(delta>0,0).rolling(14).mean() / (-delta.where(delta<0,0)).rolling(14).mean())))
        rsi = rsi_s.iloc[-1]

        if c > e50 and e50 > e200: t, s, cl, sc = "UPTREND", "BUY", "#00E676", 2
        elif c < e50 and e50 < e200: t, s, cl, sc = "DOWNTREND", "SELL", "#FF1744", -2
        else: t, s, cl, sc = "SIDEWAYS", "WAIT", "#FFD600", 0
        
        return {'trend':t, 'signal':s, 'color':cl, 'rsi_s':rsi_s, 'rsi':rsi, 'entry':c, 'sl':c-(1.5*atr) if sc>=0 else c+(1.5*atr), 'tp':c+(2.5*atr) if sc>=0 else c-(2.5*atr)}
    except: return None

def calc_pivots(df):
    p = df.iloc[-2]; pp = (p['High']+p['Low']+p['Close'])/3
    return {"PP":pp, "R1":(2*pp)-p['Low'], "S1":(2*pp)-p['High'], "R2":pp+(p['High']-p['Low']), "S2":pp-(p['High']-p['Low'])}

def calc_dynamic(df):
    sma = df['Close'].rolling(20).mean().iloc[-1]; std = df['Close'].rolling(20).std().iloc[-1]
    return {"EMA20":df['Close'].ewm(span=20).mean().iloc[-1], "EMA50":df['Close'].ewm(span=50).mean().iloc[-1], "EMA200":df['Close'].ewm(span=200).mean().iloc[-1], "BBUp":sma+(2*std), "BBLow":sma-(2*std), "Cur":df['Close'].iloc[-1]}

def analyze_sr_hybrid(price, piv, dyn, guru):
    lvls = {**piv, **{k:v for k,v in dyn.items() if k!='Cur'}}
    n_l, min_d, n_p = "", float('inf'), 0
    for k,v in lvls.items():
        if abs(price-v) < min_d: min_d, n_l, n_p = abs(price-v), k, v
    
    at_lvl = (min_d/price)*100 < 1.0
    f_score = guru['val_score'] if guru else 5
    
    msg, col, icon = "", "#888", "🔍"
    if at_lvl:
        if price > n_p: # Sup
            if f_score>=7: msg, col, icon = f"💎 รับ {n_l} + พื้นฐานดี", "#00E676", "🚀"
            elif f_score<=4: msg, col, icon = f"⚠️ ระวังหลุด {n_l}", "#FF1744", "🩸"
            else: msg, col, icon = f"🛡️ เด้งสั้น {n_l}", "#00E5FF", "🛡️"
        else: # Res
            if f_score>=7: msg, col, icon = f"📈 ลุ้นเบรค {n_l}", "#FFD600", "👀"
            else: msg, col, icon = f"🧱 ชนต้าน {n_l}", "#FF1744", "💰"
    else:
        msg, col, icon = (f"🏃 วิ่งหา {n_l}", "#00E676", "🌊") if f_score>=5 else (f"⏳ รอย่อ", "#888", "💤")
    return msg, col, icon

# --- STOCK GURU LOGIC (Stocks) ---
def analyze_stock_guru(info, setup, symbol):
    pe = info.get('trailingPE'); peg = info.get('pegRatio'); pb = info.get('priceToBook')
    roe = info.get('returnOnEquity'); pm = info.get('profitMargins')
    
    vs, rq, rv = 0, [], []
    if roe and roe>0.15: rq.append("✅ ROE สูง"); vs+=1
    if pm and pm>0.1: rq.append("✅ Margin ดี"); vs+=1
    
    if pe:
        if pe<15: vs+=3; rv.append("✅ P/E ต่ำ")
        elif pe>40: rv.append("⚠️ P/E สูง")
        else: vs+=1
    else: vs+=1
    
    if peg and peg<1: vs+=3; rv.append("✅ PEG คุ้ม")
    vs = min(10, vs+2) # Base score

    if vs>=8: vd, cl = "💎 Hidden Gem", "#00E676"
    elif vs>=5: vd, cl = "⚖️ Fair Value", "#FFD600"
    else: vd, cl = "⚠️ High Risk", "#FF1744"
    
    art = f"**บทวิเคราะห์ {symbol}:** พื้นฐาน {vd} ({vs}/10)\nกราฟ {setup['trend']} " + ("แนะนำสะสม" if vs>=7 else "ระวัง")
    return {"verdict":vd, "color":cl, "val_score":vs, "article":art, "rq":rq, "rv":rv}

# --- CRYPTO GURU LOGIC (New) ---
def analyze_crypto_guru(setup, symbol):
    # Crypto has no P/E, use Momentum & Volatility instead
    vs, rq, rv = 5, [], [] # Start neutral
    
    if setup['trend'] == "UPTREND": vs+=3; rq.append("✅ เทรนด์ขาขึ้นแข็งแกร่ง")
    elif setup['trend'] == "DOWNTREND": vs-=3; rq.append("❌ เทรนด์ขาลงกดดัน")
    
    if setup['rsi'] > 70: vs-=1; rv.append("⚠️ RSI Overbought (ระวังเทขาย)")
    elif setup['rsi'] < 30: vs+=2; rv.append("✅ RSI Oversold (ของถูก)")
    else: rv.append("⚖️ RSI เป็นกลาง")
    
    vs = min(10, max(0, vs))
    if vs>=8: vd, cl = "🚀 Moon Shot", "#00E676"
    elif vs>=5: vd, cl = "⚖️ Hold/Swing", "#FFD600"
    else: vd, cl = "🩸 Correction", "#FF1744"
    
    art = f"**บทวิเคราะห์ Crypto {symbol}:**\nเน้นเทคนิคและโมเมนตัม: {vd} ({vs}/10)\nกลยุทธ์: {setup['signal']} ตามเทรนด์"
    return {"verdict":vd, "color":cl, "val_score":vs, "article":art, "rq":rq, "rv":rv}

def gen_verdict(setup, news):
    sc = 50
    t_t = "ขาขึ้น" if setup['trend']=="UPTREND" else "ขาลง"
    n_sc = sum([n['score'] for n in news]) if news else 0
    n_t = "ข่าวบวก" if n_sc>0.3 else "ข่าวลบ" if n_sc<-0.3 else "ทรงตัว"
    
    if "UP" in setup['trend']: sc+=20
    elif "DOWN" in setup['trend']: sc-=20
    if n_sc > 0.3: sc+=15
    elif n_sc < -0.3: sc-=15
    
    sc = max(0, min(100, sc))
    vd = "BUY" if sc>=60 else "SELL" if sc<=40 else "HOLD"
    return t_t, n_t, sc, vd

def calc_bk_static(price):
    step = 50000 if price>2000000 else 10000 if price>100000 else 1000
    b = (price//step)*step
    return {"R2":b+(step*2), "R1":b+step, "S1":b, "S2":b-step}

# --- 4. Layout & Navigation ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    
    # MODE SWITCHER
    mode = st.radio("Select Market", ["🌏 Global Stocks", "🇹🇭 Bitkub Crypto"])
    
    st.markdown("---")
    st.markdown("### 🇹🇭 Bitkub Rate")
    bk_all = get_bitkub_ticker()
    if bk_all:
        b_p = bk_all.get('THB_BTC',{}).get('last',0)
        e_p = bk_all.get('THB_ETH',{}).get('last',0)
        st.markdown(f"**BTC:** <span style='color:#00E676'>{b_p:,.0f}</span>", unsafe_allow_html=True)
        st.markdown(f"**ETH:** <span style='color:#00E676'>{e_p:,.0f}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    chart_type = st.selectbox("Chart", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Period", ["1mo","3mo","6mo","1y"], value="6mo")

# --- 5. Main Content ---
st.markdown(f"<h2 style='color:#00E5FF;'>🔍 Analyze: {mode}</h2>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 1])

with c1:
    if mode == "🌏 Global Stocks":
        sym_input = st.text_input("Stock Symbol (e.g. AAPL, PTT.BK)", "BTC-USD", label_visibility="collapsed")
        is_crypto_mode = False
    else:
        # Bitkub Dropdown
        bk_coins = [k.replace("THB_","") for k in bk_all.keys()] if bk_all else ["BTC", "ETH", "KUB"]
        sel_coin = st.selectbox("Select Coin", bk_coins, label_visibility="collapsed")
        sym_input = f"{sel_coin}-THB" # Use yfinance for chart, Bitkub for price
        is_crypto_mode = True

with c2: 
    if st.button("วิเคราะห์ ⚡", use_container_width=True): 
        st.session_state.symbol = sym_input
        st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner("🚀 AI Analyzing..."):
        # Fetch Data
        df = get_market_data(symbol, period, "1d")
        
    if not df.empty:
        curr = df['Close'].iloc[-1]
        chg = curr - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        color = "#00E676" if chg >= 0 else "#FF1744"
        
        # Calculate Common Metrics
        setup = calc_technical(df)
        news = get_ai_analyzed_news_thai(symbol)
        t_txt, n_txt, ai_sc, ai_vd = gen_verdict(setup, news)
        
        sc_col, sc_glow = ("#00E676","0,230,118") if ai_sc>=70 else ("#FF1744","255,23,68") if ai_sc<=30 else ("#FFD600","255,214,0")

        # Hero Section
        st.markdown(f"""
        <div class="glass-card" style="border-top:5px solid {color};text-align:center;">
            <div style="font-size:3.5rem;font-weight:900;line-height:1;">{symbol}</div>
            <div style="font-size:3rem;color:{color};font-weight:bold;">{curr:,.2f}</div>
            <div style="background:rgba({sc_glow}, 0.2);padding:5px 20px;border-radius:20px;display:inline-block;margin-top:10px;">
                <span style="color:{color};font-weight:bold;font-size:1.1rem;">{chg:+.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Define Tabs based on Mode
        if not is_crypto_mode:
            tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R Hybrid", "🧠 AI Guru", "🧮 Calc"])
        else:
            tabs = st.tabs(["📈 Chart", "📊 Bitkub Data", "📰 AI News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R Hybrid", "🧠 Crypto Guru", "🧮 Calc"])

        # 1. Chart (Common)
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
            if chart_type == "Heikin Ashi":
                ha = df.copy(); ha['Close']=(df['Open']+df['High']+df['Low']+df['Close'])/4
                fig.add_trace(go.Candlestick(x=df.index, open=ha['Open'], high=ha['High'], low=ha['Low'], close=ha['Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF'), name="EMA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=setup['rsi_s'], line=dict(color='#E040FB'), name="RSI"), row=2, col=1)
            fig.update_layout(template='plotly_dark', height=550, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        # 2. Stats (Split Logic)
        with tabs[1]:
            if not is_crypto_mode:
                # Stock Stats
                info = get_stock_info(symbol)
                c1,c2,c3=st.columns(3)
                c1.markdown(f"<div class='metric-box'><div class='metric-lbl'>High</div><div class='metric-val' style='color:#00E676'>{df['High'].max():,.2f}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-box'><div class='metric-lbl'>Low</div><div class='metric-val' style='color:#FF1744'>{df['Low'].min():,.2f}</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-box'><div class='metric-lbl'>P/E</div><div class='metric-val'>{info.get('trailingPE','N/A')}</div></div>", unsafe_allow_html=True)
            else:
                # Bitkub Stats
                bk_coin = f"THB_{symbol.split('-')[0]}"
                d = bk_all.get(bk_coin, {})
                if d:
                    c1,c2=st.columns(2)
                    c1.markdown(f"<div class='metric-box'><div class='metric-lbl'>24H High</div><div class='metric-val' style='color:#00E676'>{d.get('high24hr',0):,.0f}</div></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='metric-box'><div class='metric-lbl'>24H Low</div><div class='metric-val' style='color:#FF1744'>{d.get('low24hr',0):,.0f}</div></div>", unsafe_allow_html=True)
                else: st.warning("Bitkub Data Unavailable")

        # 3. AI News (Common)
        with tabs[2]:
            if news:
                for n in news: st.markdown(f"""<div class="news-card {n['class']}"><div style="display:flex;justify-content:space-between;"><div>{n['icon']} <b>{n['label']}</b></div><span style="font-size:0.8rem;background:#333;padding:2px 8px;border-radius:5px;">{n['source']}</span></div><h4 style="margin:10px 0;color:#e0e0e0;">{n['title']}</h4><div style="text-align:right;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;">อ่านต่อ</a></div></div>""", unsafe_allow_html=True)
            else: st.info("No News")

        # 4. Setup (Common)
        with tabs[3]:
             st.markdown(f"""<div class='ai-insight-box' style='border-left:5px solid {setup['color']};margin-bottom:20px;'><h2 style='margin:0;color:{setup['color']};'>{setup['signal']}</h2><p style='font-size:1.2rem;color:#ccc;'>{setup['trend']}</p></div>""", unsafe_allow_html=True)
             c1,c2,c3=st.columns(3)
             c1.markdown(f"<div class='metric-box' style='border-left-color:#00E5FF'><div class='metric-lbl'>Buy Zone</div><div class='metric-val'>{curr*0.99:,.2f}</div></div>", unsafe_allow_html=True)
             c2.markdown(f"<div class='metric-box' style='border-left-color:#00E676'><div class='metric-lbl'>Target</div><div class='metric-val'>{setup['tp']:,.2f}</div></div>", unsafe_allow_html=True)
             c3.markdown(f"<div class='metric-box' style='border-left-color:#FF1744'><div class='metric-lbl'>Stop</div><div class='metric-val'>{setup['sl']:,.2f}</div></div>", unsafe_allow_html=True)

        # 5. Verdict (Common)
        with tabs[4]:
            c1,c2 = st.columns([1,1.5])
            with c1: st.markdown(f"""<div class="verdict-ring" style="border-color:{sc_col};color:{sc_col};box-shadow:0 0 30px rgba({sc_glow},0.5);">{ai_sc}</div><div style="text-align:center;font-size:2rem;font-weight:900;color:{sc_col};">{ai_vd}</div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="sr-card" style="border-left:4px solid {sc_col};"><h4 style="margin:0;">📈 Tech</h4><p>{t_txt}</p></div><div class="sr-card" style="border-left:4px solid #00E5FF;"><h4 style="margin:0;">📰 News</h4><p>{n_txt}</p></div>""", unsafe_allow_html=True)

        # 6. S/R Hybrid (Adaptive)
        with tabs[5]:
            pivots = calc_pivots(df); dynamic = calc_dynamic(df)
            
            # Select Strategy based on Mode
            if not is_crypto_mode and info:
                guru = analyze_stock_guru(info, setup, symbol)
            else:
                guru = analyze_crypto_guru(setup, symbol) # New Crypto Logic
                
            msg_s, col_s, icon_s = analyze_sr_hybrid(curr, pivots, dynamic, guru)
            st.markdown(f"""<div class='ai-insight-box' style='border:2px solid {col_s};margin-bottom:25px;'><div style="display:flex;align-items:center;gap:15px;"><span style="font-size:2.5rem;">{icon_s}</span><div><h2 style="margin:0;color:{col_s};">{msg_s}</h2><p style="color:#ddd;margin:5px 0;">AI Strategy: <b style="color:{guru['color']}">{guru['verdict']}</b></p></div></div></div>""", unsafe_allow_html=True)
            
            c1,c2=st.columns(2)
            with c1:
                st.markdown("#### 🧱 Static")
                for k,v in pivots.items(): st.markdown(f"<div class='sr-card'><b>{k}</b><span>{v:,.2f}</span></div>", unsafe_allow_html=True)
            with c2:
                st.markdown("#### 🌊 Dynamic")
                for k,v in dynamic.items():
                    if k!="Cur": st.markdown(f"<div class='sr-card'><span>{k}</span><span>{v:,.2f}</span></div>", unsafe_allow_html=True)

        # 7. Guru (Adaptive)
        with tabs[6]:
            st.markdown(f"""<div class='ai-insight-box' style='border:2px solid {guru['color']};text-align:center;margin-bottom:20px;'><h1 style='color:{guru['color']};font-size:3rem;margin:0;'>{guru['verdict']}</h1><p>Score: {guru['val_score']}/10</p></div><div class='ai-article'>{guru['article']}</div>""", unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: 
                for r in guru['rq']: st.markdown(f"<div class='guru-card' style='border-left:4px solid #00E676'>{r}</div>", unsafe_allow_html=True)
            with c2: 
                for r in guru['rv']: st.markdown(f"<div class='guru-card' style='border-left:4px solid #FF1744'>{r}</div>", unsafe_allow_html=True)

        # 8. Calculator
        with tabs[-1]: # Last tab always calc
            st.markdown("### 🧮 Position Size")
            c1,c2=st.columns(2)
            with c1: 
                bal = st.number_input("Balance", 100000.0, step=1000.0)
                risk = st.number_input("Risk %", 1.0, step=0.1)
            with c2:
                ent = st.number_input("Entry", value=setup['entry'])
                sl = st.number_input("Stop", value=setup['sl'])
            if st.button("Calculate", use_container_width=True):
                if ent!=sl:
                    qty = (bal*(risk/100))/abs(ent-sl); cost = qty*ent
                    c1,c2,c3=st.columns(3)
                    c1.metric("Qty", f"{qty:,.2f}")
                    c2.metric("Cost", f"{cost:,.2f}")
                    c3.metric("Risk", f"{bal*(risk/100):,.2f}")
    else: st.error("No Data")
