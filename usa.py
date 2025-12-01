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

# --- 1. Config & Libraries ---
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

FINNHUB_KEY = "d4l5ku1r01qt7v18ll40d4l5ku1r01qt7v18ll4g" 

st.set_page_config(page_title="Smart Trader AI : Ultra Black", layout="wide", page_icon="💎")

# --- 2. CSS Styling (Ultra Modern) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
        .stApp { background-color: #000000 !important; color: #e0e0e0; }
        
        /* Inputs */
        div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div > div { 
            background-color: #111 !important; color: #fff !important; 
            border: 2px solid #00E5FF !important; border-radius: 12px;
        }
        
        /* Cards */
        .glass-card {
            background: linear-gradient(145deg, #1a1a1a, #0d0d0d);
            border: 1px solid #333; border-radius: 20px;
            padding: 25px; margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        }
        
        /* Metrics */
        .metric-box {
            background: #111; border-radius: 15px; padding: 20px;
            border-left: 4px solid #333; margin-bottom: 10px; transition: transform 0.2s;
        }
        .metric-box:hover { transform: translateY(-5px); border-left-color: #00E5FF; }
        .metric-lbl { font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .metric-val { font-size: 1.8rem; font-weight: 800; color: #fff; margin-top: 5px; }

        /* AI Insight */
        .ai-insight-box {
            background: linear-gradient(135deg, #161616, #0a0a0a);
            border: 1px solid #333; border-radius: 15px; padding: 25px;
            position: relative; overflow: hidden; margin-bottom: 20px;
            border-left: 5px solid #00E5FF;
        }
        
        /* S/R Rows */
        .sr-card { 
            padding: 15px; border-radius: 10px; margin-bottom: 8px; 
            display: flex; justify-content: space-between; align-items: center; 
            border: 1px solid #222; background: #111; 
        }
        
        /* Verdict */
        .verdict-ring { 
            width: 150px; height: 150px; border-radius: 50%; 
            display: flex; flex-direction: column; align-items: center; justify-content: center; 
            font-size: 3.5rem; font-weight: 900; margin: 0 auto 20px; 
            border: 8px solid #333; background: #000; 
            box-shadow: 0 0 30px rgba(0, 229, 255, 0.2);
        }
        
        /* Buttons */
        div.stButton > button { 
            width: 100%; background: linear-gradient(45deg, #00E5FF, #2979FF); 
            border: none; color: #000; font-weight: 800; padding: 15px; border-radius: 12px; 
            font-size: 1.2rem; transition: transform 0.1s;
        }
        div.stButton > button:hover { transform: scale(1.02); }
        
        /* Tabs */
        button[data-baseweb="tab"] {
            font-size: 1rem; font-weight: bold; border-radius: 8px;
            background: #111; margin: 0 2px; border: 1px solid #333;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: #00E5FF; color: #000; border-color: #00E5FF;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. Core Functions ---

@st.cache_data(ttl=15)
def get_bitkub_ticker():
    try:
        r = requests.get("https://api.bitkub.com/api/market/ticker", timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

@st.cache_data(ttl=300)
def get_market_data(symbol, period, interval):
    try: return yf.Ticker(symbol).history(period=period, interval=interval)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try: return yf.Ticker(symbol).info
    except: return None

def get_finnhub_news(symbol):
    try:
        t = datetime.date.today(); f = t - datetime.timedelta(days=2)
        s = symbol.split("-")[0]
        url = f"https://finnhub.io/api/v1/company-news?symbol={s}&from={f}&to={t}&token={FINNHUB_KEY}"
        return requests.get(url).json()[:5]
    except: return []

@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    lst = []
    tr = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    
    fh = get_finnhub_news(symbol)
    if fh:
        for i in fh:
            t, s, l = i.get('headline',''), i.get('summary',''), i.get('url','#')
            sc = TextBlob(t).sentiment.polarity
            lbl, ic, cl = ("ข่าวดี","🚀","nc-pos") if sc>0.05 else ("ข่าวร้าย","🔻","nc-neg") if sc<-0.05 else ("ทั่วไป","⚖️","nc-neu")
            if tr: 
                try: t=tr.translate(t); s=tr.translate(s) if s else ""
                except: pass
            lst.append({'title':t, 'summary':s, 'link':l, 'icon':ic, 'class':cl, 'label':lbl, 'score':sc, 'source':'Finnhub'})
    
    if len(lst)<3:
        try:
            cl = symbol.replace("-THB","").replace("-USD","").replace("=F","")
            q = urllib.parse.quote(f"site:bloomberg.com {cl} market")
            fd = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            if not fd.entries: fd = feedparser.parse(f"https://news.google.com/rss/search?q={urllib.parse.quote(f'{cl} finance')}&hl=en-US&gl=US&ceid=US:en")
            for i in fd.entries[:5]:
                t, s = i.title, re.sub(r'<.*?>','', getattr(i,'summary','') or getattr(i,'description',''))[:300]
                sc = TextBlob(t).sentiment.polarity
                lbl, ic, cl = ("ข่าวดี","🚀","nc-pos") if sc>0.05 else ("ข่าวร้าย","🔻","nc-neg") if sc<-0.05 else ("ทั่วไป","⚖️","nc-neu")
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
        rs = (delta.where(delta>0,0)).rolling(14).mean() / (-delta.where(delta<0,0)).rolling(14).mean()
        rsi_s = 100 - (100/(1+rs))
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

def calculate_heikin_ashi(df):
    ha = df.copy()
    ha['Close'] = (df['Open']+df['High']+df['Low']+df['Close'])/4
    ha['Open'] = [ (df['Open'][0]+df['Close'][0])/2 ] + [0]*(len(df)-1)
    for i in range(1, len(df)): ha['Open'].iloc[i] = (ha['Open'].iloc[i-1]+ha['Close'].iloc[i-1])/2
    ha['High'] = ha[['High','Open','Close']].max(axis=1)
    ha['Low'] = ha[['Low','Open','Close']].min(axis=1)
    return ha

# --- 4. AI GURU LOGIC (Enhanced) ---

# For Stocks (Fundamental + Technical)
def analyze_stock_guru(info, setup, symbol):
    pe = info.get('trailingPE'); peg = info.get('pegRatio'); pb = info.get('priceToBook')
    roe = info.get('returnOnEquity'); pm = info.get('profitMargins')
    
    score = 0; good = []; bad = []
    
    if roe and roe > 0.15: score+=2; good.append("✅ ROE สูง บริหารเก่ง")
    if pm and pm > 0.1: score+=1; good.append("✅ กำไรต่อยอดขายดี")
    if pe:
        if pe < 15: score+=3; good.append("✅ P/E ถูก")
        elif pe > 40: score-=2; bad.append("⚠️ P/E แพง")
    else: score-=1
    
    if setup['trend'] == "UPTREND": score+=3; good.append("✅ กราฟขาขึ้น")
    elif setup['trend'] == "DOWNTREND": score-=3; bad.append("❌ กราฟขาลง")
    
    score = min(10, max(0, score))
    
    if score >= 7: 
        verd, col, act = "GEM / STRONG BUY", "#00E676", "พื้นฐานดีและกราฟสวย เหมาะแก่การสะสม"
    elif score >= 4:
        verd, col, act = "WATCH / HOLD", "#FFD600", "พื้นฐานปานกลาง หรือราคายังไม่ไป รอจังหวะ"
    else:
        verd, col, act = "AVOID / SELL", "#FF1744", "พื้นฐานแย่หรือกราฟเสียทรง ควรหลีกเลี่ยง"
        
    art = f"**บทวิเคราะห์หุ้น {symbol}:**\n\n" + "\n".join(good) + "\n" + "\n".join(bad) + f"\n\n**สรุป:** {act}"
    return {"verdict": verd, "color": col, "score": score, "article": art, "strategy": act}

# For Crypto (Technical + Momentum + Volatility)
def analyze_crypto_guru(setup, symbol, bk_data=None):
    score = 5
    good = []; bad = []
    
    # Trend
    if setup['trend'] == "UPTREND": score+=3; good.append("✅ เทรนด์หลักขาขึ้น (EMA Bullish)")
    elif setup['trend'] == "DOWNTREND": score-=3; bad.append("❌ เทรนด์หลักขาลง (EMA Bearish)")
    
    # RSI
    if setup['rsi'] > 70: score-=1; bad.append("⚠️ RSI Overbought (ระวังแรงเทขาย)")
    elif setup['rsi'] < 30: score+=2; good.append("✅ RSI Oversold (จุดเด้งระยะสั้น)")
    
    # Bitkub Data
    if bk_data:
        last = bk_data.get('last', 0)
        high24 = bk_data.get('high24hr', 0)
        change = bk_data.get('percentChange', 0)
        
        if change > 5: score+=1; good.append("🔥 โมเมนตัมรายวันแรง (+5%)")
        elif change < -5: score-=1; bad.append("🩸 แรงขายรายวันหนัก (-5%)")
        
        if last > high24 * 0.95: score+=1; good.append("🚀 ใกล้ Breakout High เดิม")
    
    score = min(10, max(0, score))
    
    if score >= 8: 
        verd, col, act = "🚀 MOON SHOT", "#00E676", "โมเมนตัมกระทิงดุ ซื้อตามน้ำ (Follow Buy) หรือย่อรับ"
    elif score >= 5:
        verd, col, act = "⚖️ SWING TRADE", "#FFD600", "ไซด์เวย์รอเลือกทาง ซื้อแนวรับ-ขายแนวต้าน"
    else:
        verd, col, act = "🩸 DUMP / WAIT", "#FF1744", "แรงขายหนัก อย่าเพิ่งรับมีด รอสร้างฐานใหม่"
        
    art = f"**บทวิเคราะห์ Crypto {symbol}:**\n\n" + "\n".join(good) + "\n" + "\n".join(bad) + f"\n\n**กลยุทธ์ AI:** {act}"
    return {"verdict": verd, "color": col, "score": score, "article": art, "strategy": act}

# --- 5. UI Logic ---

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    mode = st.radio("Select Market", ["🌏 Global Stocks", "🇹🇭 Bitkub Crypto"])
    st.markdown("---")
    
    bk_all = get_bitkub_ticker()
    if bk_all:
        st.markdown("### 🇹🇭 Bitkub Live Rate")
        b_p = bk_all.get('THB_BTC',{}).get('last',0)
        e_p = bk_all.get('THB_ETH',{}).get('last',0)
        k_p = bk_all.get('THB_KUB',{}).get('last',0)
        st.markdown(f"**BTC:** <span style='color:#00E676'>{b_p:,.0f}</span>", unsafe_allow_html=True)
        st.markdown(f"**ETH:** <span style='color:#00E676'>{e_p:,.0f}</span>", unsafe_allow_html=True)
        st.markdown(f"**KUB:** <span style='color:#FFD600'>{k_p:,.2f}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    chart_type = st.selectbox("Chart", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Timeframe", ["1mo","3mo","6mo","1y"], value="6mo")

# Main Input
st.markdown(f"<h2 style='color:#00E5FF;'>🔍 Analyze: {mode}</h2>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 1])

with c1:
    if mode == "🌏 Global Stocks":
        sym_in = st.text_input("Symbol (e.g. AAPL, PTT.BK)", "TSLA", label_visibility="collapsed")
        is_crypto = False
    else:
        bk_coins = [k.replace("THB_","") for k in bk_all.keys()] if bk_all else ["BTC", "ETH", "KUB", "DOGE"]
        sel_coin = st.selectbox("Select Coin", bk_coins, label_visibility="collapsed")
        sym_in = f"{sel_coin}-THB" # yfinance format for chart
        is_crypto = True

with c2:
    if st.button("วิเคราะห์ ⚡", use_container_width=True):
        st.session_state.symbol = sym_in
        st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner("🚀 AI กำลังประมวลผลข้อมูล..."):
        # Fetch Data
        if is_crypto:
            # Data from Bitkub for Price/Stats
            bk_pair = f"THB_{symbol.split('-')[0]}"
            bk_d = bk_all.get(bk_pair, {}) if bk_all else {}
            
            # Data from yfinance for Chart/Indicators
            df = get_market_data(symbol, period, "1d")
            
            curr = bk_d.get('last', df['Close'].iloc[-1] if not df.empty else 0)
            prev = bk_d.get('prevClose', df['Close'].iloc[-2] if not df.empty else curr)
            
            # No fundamental info for crypto
            info = None
        else:
            # Data from yfinance for Stocks
            df = get_market_data(symbol, period, "1d")
            info = get_stock_info(symbol)
            curr = df['Close'].iloc[-1] if not df.empty else 0
            prev = df['Close'].iloc[-2] if not df.empty else curr
            bk_d = {}

    if not df.empty and curr > 0:
        chg = curr - prev
        pct = (chg/prev)*100 if prev else 0
        col = "#00E676" if chg>=0 else "#FF1744"
        
        # Calculations
        setup = calc_technical(df)
        news = get_ai_analyzed_news_thai(symbol)
        pivots = calc_pivots(df)
        dynamic = calc_dynamic(df)
        
        # AI Guru Selection
        if is_crypto:
            guru = analyze_crypto_guru(setup, symbol, bk_d)
        else:
            guru = analyze_stock_guru(info if info else {}, setup, symbol)

        # UI Layout
        st.markdown(f"""
        <div class="glass-card" style="border-top:5px solid {col};text-align:center;">
            <div style="font-size:3.5rem;font-weight:900;line-height:1;">{symbol.replace('-THB','')}</div>
            <div style="font-size:3.5rem;color:{col};font-weight:bold;">{curr:,.2f}</div>
            <div style="background:rgba(255,255,255,0.1);padding:5px 20px;border-radius:20px;display:inline-block;margin-top:10px;">
                <span style="color:{col};font-weight:bold;font-size:1.2rem;">{chg:+.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Define Tabs
        if is_crypto:
            tabs = st.tabs(["📈 Chart", "🧠 Crypto Guru", "📊 Bitkub Data", "🛡️ S/R & Strategy", "📰 News", "🧮 Calc"])
        else:
            tabs = st.tabs(["📈 Chart", "🧠 AI Guru", "📊 Stats", "🛡️ S/R & Strategy", "📰 News", "🧮 Calc"])

        # 1. Chart
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7,0.3], shared_xaxes=True)
            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['Open'], high=ha['High'], low=ha['Low'], close=ha['Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF'), name="EMA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=setup['rsi_s'], line=dict(color='#E040FB'), name="RSI"), row=2, col=1)
            fig.update_layout(template='plotly_dark', height=500, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        # 2. AI Guru (Adaptive)
        with tabs[1]:
            st.markdown(f"""
            <div class='ai-insight-box' style='border:2px solid {guru['color']};text-align:center;'>
                <div class='verdict-ring' style='border-color:{guru['color']};color:{guru['color']};'>{guru['score']}</div>
                <h1 style='color:{guru['color']};margin:0;'>{guru['verdict']}</h1>
                <h3 style='color:#fff;margin-top:10px;'>{guru['strategy']}</h3>
            </div>
            <div class='ai-article'>{guru['article']}</div>
            """, unsafe_allow_html=True)

        # 3. Stats / Bitkub Data
        with tabs[2]:
            if is_crypto and bk_d:
                c1,c2 = st.columns(2)
                c1.markdown(f"<div class='metric-box'><div class='metric-lbl'>24H High</div><div class='metric-val' style='color:#00E676'>{bk_d.get('high24hr',0):,.2f}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-box'><div class='metric-lbl'>24H Low</div><div class='metric-val' style='color:#FF1744'>{bk_d.get('low24hr',0):,.2f}</div></div>", unsafe_allow_html=True)
                st.info(f"Volume: {bk_d.get('baseVolume',0):,.2f} {symbol.split('-')[0]}")
            elif info:
                c1,c2,c3 = st.columns(3)
                c1.markdown(f"<div class='metric-box'><div class='metric-lbl'>P/E</div><div class='metric-val'>{info.get('trailingPE','N/A')}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-box'><div class='metric-lbl'>PEG</div><div class='metric-val'>{info.get('pegRatio','N/A')}</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-box'><div class='metric-lbl'>ROE</div><div class='metric-val'>{info.get('returnOnEquity','N/A')}</div></div>", unsafe_allow_html=True)
                with st.expander("🏢 Business Profile"): st.write(info.get('longBusinessSummary',''))
            else: st.warning("No Data Available")

        # 4. S/R & Strategy
        with tabs[3]:
            st.markdown("### 🛡️ Support & Resistance Map")
            
            # Calculate Static Levels (Different logic for Crypto/Stocks)
            if is_crypto:
                # Crypto uses Round Numbers & 24H Range
                step = 10**(len(str(int(curr)))-2) # Auto step size
                base = (curr // step) * step
                res = base + step; sup = base
            else:
                # Stocks use Pivots
                res = pivots['R1']; sup = pivots['S1']

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class='sr-card' style='border-left:5px solid #FF1744;'><div>🧱 RESISTANCE</div><div style='color:#FF1744;font-weight:bold;font-size:1.2rem;'>{res:,.2f}</div></div>""", unsafe_allow_html=True)
                if pivots: st.markdown(f"<div class='sr-card'><span>R2</span><span>{pivots['R2']:,.2f}</span></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class='sr-card' style='border-left:5px solid #00E676;'><div>🛡️ SUPPORT</div><div style='color:#00E676;font-weight:bold;font-size:1.2rem;'>{sup:,.2f}</div></div>""", unsafe_allow_html=True)
                if pivots: st.markdown(f"<div class='sr-card'><span>S2</span><span>{pivots['S2']:,.2f}</span></div>", unsafe_allow_html=True)
            
            st.markdown("### 🌊 Dynamic Trends")
            for k, v in dynamic.items():
                if k != 'Current':
                    cl = "#00E676" if curr > v else "#FF1744"
                    st.markdown(f"<div class='sr-card' style='border-left:5px solid {cl};'><span>{k}</span><span>{v:,.2f}</span></div>", unsafe_allow_html=True)

            if is_crypto:
                st.markdown("### 📐 Golden Zone (24H)")
                h24 = bk_d.get('high24hr', curr*1.05)
                l24 = bk_d.get('low24hr', curr*0.95)
                rng = h24 - l24
                g_top = l24 + (rng * 0.618)
                g_bot = l24 + (rng * 0.382)
                st.info(f"Golden Pocket: {g_bot:,.2f} - {g_top:,.2f}")

        # 5. News
        with tabs[4]:
            if news:
                for n in news: st.markdown(f"""<div class="news-card {n['class']}"><div style="display:flex;justify-content:space-between;"><div>{n['icon']} <b>{n['label']}</b></div><span style="font-size:0.8rem;background:#333;padding:2px 8px;border-radius:5px;">{n['source']}</span></div><h4 style="margin:10px 0;color:#e0e0e0;">{n['title']}</h4><div style="text-align:right;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;">อ่านต่อ</a></div></div>""", unsafe_allow_html=True)
            else: st.info("No News")

        # 6. Calculator
        with tabs[5]:
            c1,c2=st.columns(2)
            with c1: 
                bal = st.number_input("Balance", 100000.0)
                risk = st.number_input("Risk %", 1.0)
            with c2:
                ent = st.number_input("Entry", setup['entry'])
                sl = st.number_input("Stop", setup['sl'])
            if st.button("Calculate"):
                q = (bal*(risk/100))/abs(ent-sl)
                c = q*ent
                st.success(f"Buy: {q:,.4f} units | Cost: {c:,.2f}")

    else: st.error("ไม่พบข้อมูล หรือ Symbol ผิด")
