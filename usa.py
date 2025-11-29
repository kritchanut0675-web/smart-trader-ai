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

FINNHUB_KEY = "d4l5ku1r01qt7v18ll40d4l5ku1r01qt7v18ll4g" 

# --- Setup ---
st.set_page_config(page_title="Smart Trader AI", layout="wide", page_icon="💎")
if 'symbol' not in st.session_state: st.session_state.symbol = 'BTC-USD'
def set_symbol(sym): st.session_state.symbol = sym

# --- CSS (Minified for Safety) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Kanit',sans-serif;}
.stApp{background-color:#050505!important;color:#e0e0e0;}
div[data-testid="stTextInput"] input{background-color:#111!important;color:#fff!important;border:1px solid #333!important;border-radius:10px;}
.glass-card{background:linear-gradient(145deg,#1a1a1a,#0d0d0d);border:1px solid #333;border-radius:20px;padding:25px;margin-bottom:20px;box-shadow:0 8px 32px rgba(0,0,0,0.5);}
.metric-box{background:#111;border-radius:15px;padding:20px;border-left:4px solid #333;transition:transform 0.2s;margin-bottom:10px;}
.metric-box:hover{transform:translateY(-5px);border-left-color:#00E5FF;}
.metric-val{font-size:1.8rem;font-weight:800;color:#fff;margin-top:5px;}
.sr-card{padding:15px 20px;border-radius:12px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;border:1px solid rgba(255,255,255,0.05);backdrop-filter:blur(5px);}
.sr-res{background:linear-gradient(90deg,rgba(255,23,68,0.2),rgba(0,0,0,0));border-left:5px solid #FF1744;}
.sr-sup{background:linear-gradient(90deg,rgba(0,230,118,0.2),rgba(0,0,0,0));border-left:5px solid #00E676;}
.verdict-ring{width:140px;height:140px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:3rem;font-weight:900;margin:0 auto 20px auto;border:8px solid #333;background:#000;}
.ai-insight-box{background:linear-gradient(135deg,#111,#0a0a0a);border:1px solid #333;border-radius:15px;padding:25px;position:relative;overflow:hidden;}
.news-card{padding:20px;margin-bottom:15px;background:#111;border-radius:15px;border-left:5px solid #888;transition:transform 0.2s;}
.news-card:hover{transform:translateX(5px);background:#161616;}
.nc-pos{border-left-color:#00E676;} .nc-neg{border-left-color:#FF1744;} .nc-neu{border-left-color:#FFD600;}
div.stButton>button{width:100%;justify-content:center;font-size:1.1rem!important;padding:12px!important;border-radius:12px!important;background:linear-gradient(45deg,#00E5FF,#2979FF);border:none!important;color:#000!important;font-weight:800!important;}
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

def get_sector_pe(sector):
    bench = {'Technology':25,'Financial':15,'Healthcare':22,'Energy':12}
    return bench.get(sector, 20)

@st.cache_data(ttl=15)
def get_bitkub_ticker():
    try:
        r = requests.get("https://api.bitkub.com/api/market/ticker", timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

def get_finnhub_news(symbol):
    try:
        td = datetime.date.today()
        fd = td - datetime.timedelta(days=2)
        sym = symbol.split("-")[0]
        url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={fd}&to={td}&token={FINNHUB_KEY}"
        data = requests.get(url).json()
        return data[:5] if isinstance(data, list) else []
    except: return []

@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    news_list = []
    trans = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    
    fh = get_finnhub_news(symbol)
    if fh:
        for i in fh:
            t, s, l = i.get('headline',''), i.get('summary',''), i.get('url','#')
            sc = TextBlob(t).sentiment.polarity
            if sc > 0.05: lbl, ic, cls = "ข่าวดี", "🚀", "nc-pos"
            elif sc < -0.05: lbl, ic, cls = "ข่าวร้าย", "🔻", "nc-neg"
            else: lbl, ic, cls = "ทั่วไป", "⚖️", "nc-neu"
            
            if trans:
                try: t = trans.translate(t); s = trans.translate(s) if s else ""
                except: pass
            news_list.append({'title':t, 'summary':s, 'link':l, 'icon':ic, 'class':cls, 'label':lbl, 'score':sc, 'source':'Finnhub'})

    if len(news_list) < 3:
        try:
            cl = symbol.replace("-THB","").replace("-USD","").replace("=F","")
            q = urllib.parse.quote(f"site:bloomberg.com {cl} market")
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            for i in feed.entries[:5]:
                t = i.title
                s = re.sub(re.compile('<.*?>'), '', getattr(i,'summary','') or getattr(i,'description',''))[:300]
                sc = TextBlob(t).sentiment.polarity
                if sc > 0.05: lbl, ic, cls = "ข่าวดี", "🚀", "nc-pos"
                elif sc < -0.05: lbl, ic, cls = "ข่าวร้าย", "🔻", "nc-neg"
                else: lbl, ic, cls = "ทั่วไป", "⚖️", "nc-neu"
                if trans:
                    try: t = trans.translate(t); s = trans.translate(s) if s else ""
                    except: pass
                news_list.append({'title':t, 'summary':s, 'link':i.link, 'icon':ic, 'class':cls, 'label':lbl, 'score':sc, 'source':'Google'})
        except: pass
    return news_list[:10]

def calculate_technical_setup(df):
    try:
        c = df['Close'].iloc[-1]
        e50 = df['Close'].ewm(span=50).mean().iloc[-1]
        e200 = df['Close'].ewm(span=200).mean().iloc[-1]
        atr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        
        delta = df['Close'].diff()
        rs = (delta.where(delta>0,0)).rolling(14).mean() / (-delta.where(delta<0,0)).rolling(14).mean()
        rsi_s = 100 - (100/(1+rs))
        rsi_val = rsi_s.iloc[-1]

        if c > e50 and e50 > e200: t,s,co,sc = "UPTREND", "BUY", "#00E676", 2
        elif c < e50 and e50 < e200: t,s,co,sc = "DOWNTREND", "SELL", "#FF1744", -2
        else: t,s,co,sc = "SIDEWAYS", "WAIT", "#FFD600", 0
        
        return {'trend':t, 'signal':s, 'color':co, 'rsi_series':rsi_s, 'rsi_val':rsi_val, 'entry':c, 'sl':c-(1.5*atr) if sc>=0 else c+(1.5*atr), 'tp':c+(2.5*atr) if sc>=0 else c-(2.5*atr)}
    except: return None

def calculate_pivots(df):
    p = df.iloc[-2]
    pp = (p['High']+p['Low']+p['Close'])/3
    return {"PP":pp, "R1":(2*pp)-p['Low'], "S1":(2*pp)-p['High'], "R2":pp+(p['High']-p['Low']), "S2":pp-(p['High']-p['Low'])}

def calculate_dynamic(df):
    sma = df['Close'].rolling(20).mean().iloc[-1]
    std = df['Close'].rolling(20).std().iloc[-1]
    return {"EMA 20":df['Close'].ewm(span=20).mean().iloc[-1], "EMA 50":df['Close'].ewm(span=50).mean().iloc[-1], "EMA 200":df['Close'].ewm(span=200).mean().iloc[-1], "BB Upper":sma+(2*std), "BB Lower":sma-(2*std), "Current":df['Close'].iloc[-1]}

def calculate_bitkub_ai(h, l, c):
    pp = (h+l+c)/3
    rng = h-l
    mid = (h+l)/2
    st, col = ("BULLISH", "#00E676") if c > mid else ("BEARISH", "#FF1744")
    insight = f"ราคายืนเหนือ Pivot ({pp:,.0f}) ได้" if c > pp else f"ราคาหลุดต่ำกว่า Pivot ({pp:,.0f})"
    return {
        "levels": [{"name":"🚀 R2","price":pp+rng,"type":"res"}, {"name":"🛑 R1","price":(2*pp)-l,"type":"res"}, {"name":"⚖️ PIVOT","price":pp,"type":"neu"}, {"name":"🛡️ S1","price":(2*pp)-h,"type":"sup"}, {"name":"💎 S2","price":pp-rng,"type":"sup"}],
        "fib": {"top": l+(rng*0.618), "bot": l+(rng*0.382)}, "status": st, "color": col, "insight": insight
    }

def gen_verdict(setup, news):
    score = 50
    t_txt, n_txt = "", ""
    if setup['trend']=="UPTREND": score+=20; t_txt="กราฟขาขึ้นชัดเจน"
    elif setup['trend']=="DOWNTREND": score-=20; t_txt="กราฟขาลงชัดเจน"
    else: t_txt="กราฟออกข้าง"
    
    if setup['rsi_val']>70: score-=5; t_txt+=" (Overbought)"
    elif setup['rsi_val']<30: score+=5; t_txt+=" (Oversold)"
    
    n_sc = sum([n['score'] for n in news]) if news else 0
    if n_sc>0.3: score+=15; n_txt="ข่าวบวกหนุนราคา"
    elif n_sc<-0.3: score-=15; n_txt="ข่าวลบกดดัน"
    else: n_txt="ข่าวทรงตัว"
    
    score = max(0, min(100, score))
    verd = "STRONG BUY" if score>=80 else "BUY" if score>=60 else "SELL" if score<=40 else "STRONG SELL" if score<=20 else "HOLD"
    return t_txt, n_txt, score, verd

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    if c1.button("BTC"): set_symbol("BTC-USD")
    if c2.button("ETH"): set_symbol("ETH-USD")
    st.markdown("---")
    bk_data = get_bitkub_ticker()
    if bk_data:
        b=bk_data.get('THB_BTC',{})
        e=bk_data.get('THB_ETH',{})
        st.markdown(f"**BTC:** <span style='color:#00E676'>{b.get('last',0):,.0f}</span>", unsafe_allow_html=True)
        st.markdown(f"**ETH:** <span style='color:#00E676'>{e.get('last',0):,.0f}</span>", unsafe_allow_html=True)
    st.markdown("---")
    chart_type = st.selectbox("Style", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Period", ["1mo","3mo","6mo","1y"], value="6mo")

# --- Main ---
st.markdown("<h2 style='color:#00E5FF;'>🔍 Smart Search</h2>", unsafe_allow_html=True)
c1,c2 = st.columns([3,1])
with c1: sym_input = st.text_input("Symbol", st.session_state.symbol, label_visibility="collapsed")
with c2: 
    if st.button("วิเคราะห์ ⚡", use_container_width=True): set_symbol(sym_input); st.rerun()

symbol = st.session_state.symbol.upper()
if symbol:
    with st.spinner("🚀 Analyzing..."):
        df = get_market_data(symbol, period, "1d")
    
    if not df.empty:
        curr = df['Close'].iloc[-1]
        chg = curr - df['Close'].iloc[-2]
        pct = (chg/df['Close'].iloc[-2])*100
        color = "#00E676" if chg >= 0 else "#FF1744"
        
        setup = calculate_technical_setup(df)
        news = get_ai_analyzed_news_thai(symbol)
        info = get_stock_info(symbol)
        t_txt, n_txt, ai_sc, ai_vd = gen_verdict(setup, news)
        
        sc_col = "#00E676" if ai_sc>=70 else "#FF1744" if ai_sc<=30 else "#FFD600"
        sc_glow = "0,230,118" if ai_sc>=70 else "255,23,68" if ai_sc<=30 else "255,214,0"

        st.markdown(f"""
        <div class="glass-card" style="border-top:5px solid {color};text-align:center;">
            <div style="font-size:3.5rem;font-weight:900;">{symbol}</div>
            <div style="font-size:3rem;color:{color};font-weight:bold;">{curr:,.2f}</div>
            <div style="background:rgba({sc_glow},0.2);padding:5px 20px;border-radius:20px;display:inline-block;">
                <span style="color:{color};font-weight:bold;">{chg:+.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R", "🇹🇭 Bitkub AI"])

        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7,0.3], shared_xaxes=True)
            if chart_type == "Heikin Ashi":
                ha = df.copy(); ha['Close']=(df['Open']+df['High']+df['Low']+df['Close'])/4
                fig.add_trace(go.Candlestick(x=df.index, open=ha['Open'], high=ha['High'], low=ha['Low'], close=ha['Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF'), name="EMA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=setup['rsi_series'], line=dict(color='#E040FB'), name="RSI"), row=2, col=1)
            fig.update_layout(template='plotly_dark', height=550, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            c1,c2,c3 = st.columns(3)
            c1.markdown(f"<div class='metric-box'><div class='metric-label'>High</div><div class='metric-val' style='color:#00E676'>{df['High'].max():,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><div class='metric-label'>Low</div><div class='metric-val' style='color:#FF1744'>{df['Low'].min():,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box'><div class='metric-label'>Vol</div><div class='metric-val' style='color:#E040FB'>{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>", unsafe_allow_html=True)
            if info:
                st.markdown("---")
                sm = info.get('longBusinessSummary','No Data.')
                if HAS_TRANSLATOR: 
                    try: sm = GoogleTranslator(source='auto', target='th').translate(sm[:2000])
                    except: pass
                with st.expander(f"🏢 เกี่ยวกับ {symbol}"): st.write(sm)
                
                pe = info.get('trailingPE')
                sec = info.get('sector','Unknown')
                avg_pe = get_sector_pe(sec)
                
                c1,c2 = st.columns(2)
                c1.markdown(f"<div class='metric-box'><div class='metric-label'>P/E Ratio</div><div class='metric-val'>{pe if pe else 'N/A'}</div></div>", unsafe_allow_html=True)
                if pe:
                    diff = ((pe-avg_pe)/avg_pe)*100
                    stt = "แพงกว่ากลุ่ม" if diff>0 else "ถูกกว่ากลุ่ม"
                    cl = "#FF1744" if diff>0 else "#00E676"
                    c2.markdown(f"<div class='metric-box' style='border-left-color:{cl}'><div class='metric-label'>เทียบกลุ่ม ({avg_pe})</div><div class='metric-val' style='color:{cl}'>{stt} ({abs(diff):.1f}%)</div></div>", unsafe_allow_html=True)

        with tabs[2]:
            if news:
                for n in news:
                    st.markdown(f"""<div class="news-card {n['class']}"><div style="display:flex;justify-content:space-between;margin-bottom:5px;"><div style="display:flex;gap:10px;"><span>{n['icon']}</span><span style="font-weight:bold;color:#fff;">{n['label']}</span></div><span style="font-size:0.8rem;background:#333;padding:2px 8px;border-radius:5px;">{n['source']}</span></div><h4 style="margin:10px 0;color:#e0e0e0;">{n['title']}</h4><p style="color:#aaa;font-size:0.9rem;">{n['summary']}</p><div style="text-align:right;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;">อ่านต่อ</a></div></div>""", unsafe_allow_html=True)
            else: st.info("ไม่พบข่าว")

        with tabs[3]:
            st.markdown(f"""<div class='ai-insight-box' style='border-left:5px solid {setup['color']};margin-bottom:20px;'><h2 style='margin:0;color:{setup['color']};'>{setup['signal']}</h2><p style='font-size:1.2rem;color:#ccc;'>{setup['trend']}</p><div style='margin-top:15px;display:flex;gap:10px;'><span style='background:#111;padding:5px 15px;border-radius:10px;border:1px solid #333;'>RSI: {setup['rsi_val']:.1f}</span><span style='background:#111;padding:5px 15px;border-radius:10px;border:1px solid #333;'>Entry: {setup['entry']:,.2f}</span></div></div>""", unsafe_allow_html=True)
            c1,c2,c3=st.columns(3)
            c1.markdown(f"<div class='metric-box' style='border-left-color:#00E5FF'><div class='metric-label'>Buy Zone</div><div class='metric-val'>{curr*0.99:,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box' style='border-left-color:#00E676'><div class='metric-label'>Target</div><div class='metric-val'>{setup['tp']:,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box' style='border-left-color:#FF1744'><div class='metric-label'>Stop Loss</div><div class='metric-val'>{setup['sl']:,.2f}</div></div>", unsafe_allow_html=True)

        with tabs[4]:
            c1,c2 = st.columns([1,1.5])
            with c1: st.markdown(f"""<div class="verdict-ring" style="border-color:{sc_col};color:{sc_col};box-shadow:0 0 30px rgba({sc_glow},0.5);">{ai_sc}</div><div style="text-align:center;font-size:2rem;font-weight:900;color:{sc_col};">{ai_vd}</div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="metric-box" style="border-left-color:{sc_col};"><h4 style="margin:0;">📈 Technical</h4><p>{t_txt}</p></div><div class="metric-box" style="border-left-color:{'#00E676' if 'บวก' in n_txt else '#FF1744'};"><h4 style="margin:0;">📰 News</h4><p>{n_txt}</p></div>""", unsafe_allow_html=True)

        with tabs[5]:
            pivs = calculate_pivot_points(df)
            dyn = calculate_dynamic_levels(df)
            c1,c2=st.columns(2)
            with c1:
                st.markdown("#### 🧱 Static Pivots")
                for k,v in pivs.items():
                    cl = "#FF1744" if "R" in k else "#00E676" if "S" in k else "#FFD600"
                    st.markdown(f"<div class='sr-card' style='border-left:5px solid {cl};'><b>{k}</b><span>{v:,.2f}</span></div>", unsafe_allow_html=True)
            with c2:
                st.markdown("#### 🌊 Dynamic")
                for k,v in dyn.items():
                    if k!="Current":
                        cl="#00E676" if curr>v else "#FF1744"
                        st.markdown(f"<div class='sr-card' style='border-left:5px solid {cl};'><span>{k}</span><div style='text-align:right;'>{v:,.2f}</div></div>", unsafe_allow_html=True)

        with tabs[6]:
            bk_sel = st.radio("Select Coin", ["BTC","ETH"], horizontal=True)
            if bk_data:
                pair = f"THB_{bk_sel}"
                d = bk_data.get(pair,{})
                if d:
                    ai_bk = calculate_bitkub_ai(d.get('high24hr',0), d.get('low24hr',0), d.get('last',0))
                    st.markdown(f"""<div class='ai-insight-box' style='text-align:center;border:2px solid {ai_bk['color']};'><div style='font-size:3rem;font-weight:900;color:#fff;'>{d.get('last',0):,.0f} THB</div><div style='font-size:1.5rem;font-weight:bold;color:{ai_bk['color']};'>{ai_bk['status']}</div><p style='margin-top:10px;color:#ccc;'>🧠 {ai_bk['insight']}</p></div>""", unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        for l in ai_bk['levels']:
                            cl = "#00E676" if l['type']=='sup' else "#FF1744" if l['type']=='res' else "#FFD600"
                            st.markdown(f"<div class='sr-card' style='border-left:5px solid {cl};'><b>{l['name']}</b><span>{l['price']:,.0f}</span></div>", unsafe_allow_html=True)
                    with c2:
                        st.info(f"**Golden Bot:** {ai_bk['fib']['bot']:,.0f}\n\n**Golden Top:** {ai_bk['fib']['top']:,.0f}")
                        with st.expander("ℹ️ Golden Zone?"):
                            st.write("โซนราคา Fibonacci (61.8% - 38.2%) ของ 24 ชม. ล่าสุด เป็นจุดวัดใจสำคัญ ถ้าหลุดคือลง ถ้าเบรกคือขึ้น")
                else: st.error("No Data")
            else: st.warning("Connecting...")
    else: st.error("ไม่พบข้อมูล")
