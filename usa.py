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

# --- 1. Libraries & Config ---
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

FINNHUB_KEY = "d4l5ku1r01qt7v18ll40d4l5ku1r01qt7v18ll4g" 

st.set_page_config(page_title="Smart Trader AI : Ultra Black", layout="wide", page_icon="💎")
if 'symbol' not in st.session_state: st.session_state.symbol = 'BTC-USD'
def set_symbol(sym): st.session_state.symbol = sym

# --- 2. CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
html,body{font-family:'Kanit',sans-serif;} .stApp{background-color:#050505;color:#e0e0e0;}
div[data-testid="stTextInput"] input{background-color:#111;color:#fff;border:2px solid #00E5FF;border-radius:10px;}
.glass-card{background:linear-gradient(145deg,#1a1a1a,#0d0d0d);border:1px solid #333;border-radius:20px;padding:25px;margin-bottom:20px;box-shadow:0 8px 32px rgba(0,0,0,0.5);}
.metric-box{background:#111;border-radius:15px;padding:20px;border-left:4px solid #333;margin-bottom:10px;}
.metric-label{font-size:0.9rem;color:#888;text-transform:uppercase;}
.metric-val{font-size:1.8rem;font-weight:800;color:#fff;margin-top:5px;}
.ai-insight-box{background:linear-gradient(135deg,#111,#0a0a0a);border:1px solid #333;border-radius:15px;padding:25px;overflow:hidden;margin-bottom:20px;}
.news-card{padding:20px;margin-bottom:15px;background:#111;border-radius:15px;border-left:5px solid #888;transition:0.2s;}
.news-card:hover{transform:translateX(5px);background:#161616;}
.nc-pos{border-left-color:#00E676;} .nc-neg{border-left-color:#FF1744;} .nc-neu{border-left-color:#FFD600;}
.sr-card{padding:15px;border-radius:12px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;border:1px solid rgba(255,255,255,0.1);}
.sr-res{background:linear-gradient(90deg,rgba(255,23,68,0.2),rgba(0,0,0,0));border-left:5px solid #FF1744;}
.sr-sup{background:linear-gradient(90deg,rgba(0,230,118,0.2),rgba(0,0,0,0));border-left:5px solid #00E676;}
.sr-piv{background:linear-gradient(90deg,rgba(255,214,0,0.2),rgba(0,0,0,0));border-left:5px solid #FFD600;}
.static-card{background:#161616;padding:15px;border-radius:10px;border:1px solid #333;margin-bottom:8px;display:flex;justify-content:space-between;}
.verdict-ring{width:140px;height:140px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:3rem;font-weight:900;margin:0 auto 20px;border:8px solid #333;background:#000;}
.ai-article{background:rgba(255,255,255,0.05);padding:25px;border-radius:15px;border-left:4px solid #00E5FF;margin-top:20px;line-height:1.8;}
.guru-card{background:#111;padding:15px;border-radius:12px;border:1px solid #333;margin-bottom:10px;}
button[data-baseweb="tab"]{font-size:1rem;font-weight:600;border-radius:8px;margin:0 4px;background:#111;border:1px solid #333;}
button[data-baseweb="tab"][aria-selected="true"]{background:#00E5FF;color:#000;border-color:#00E5FF;}
div.stButton>button{width:100%;font-size:1.1rem;padding:12px;border-radius:12px;background:linear-gradient(45deg,#00E5FF,#2979FF);border:none;color:#000;font-weight:800;}
</style>
""", unsafe_allow_html=True)

# --- 3. Functions (ALL DEFINITIONS HERE) ---

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

def get_finnhub_news(symbol):
    try:
        td = datetime.date.today()
        fd = td - datetime.timedelta(days=2)
        sym = symbol.split("-")[0]
        url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={fd}&to={td}&token={FINNHUB_KEY}"
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

def calculate_technical_setup(df):
    try:
        c = df['Close'].iloc[-1]
        e50 = df['Close'].ewm(span=50).mean().iloc[-1]
        e200 = df['Close'].ewm(span=200).mean().iloc[-1]
        atr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        delta = df['Close'].diff()
        rs = (delta.where(delta>0,0)).rolling(14).mean() / (-delta.where(delta<0,0)).rolling(14).mean()
        rsi_s = 100 - (100/(1+rs))
        rsi = rsi_s.iloc[-1]
        
        if c > e50 and e50 > e200: t,s,co,sc = "UPTREND (ขาขึ้น)", "BUY", "#00E676", 2
        elif c < e50 and e50 < e200: t,s,co,sc = "DOWNTREND (ขาลง)", "SELL", "#FF1744", -2
        else: t,s,co,sc = "SIDEWAYS (ออกข้าง)", "WAIT", "#FFD600", 0
        
        return {'trend':t, 'signal':s, 'color':co, 'rsi_series':rsi_s, 'rsi_val':rsi, 'entry':c, 'sl':c-(1.5*atr) if sc>=0 else c+(1.5*atr), 'tp':c+(2.5*atr) if sc>=0 else c-(2.5*atr)}
    except: return None

def calculate_pivot_points(df):
    try:
        p = df.iloc[-2]; pp = (p['High']+p['Low']+p['Close'])/3
        return {"PP":pp, "R1":(2*pp)-p['Low'], "S1":(2*pp)-p['High'], "R2":pp+(p['High']-p['Low']), "S2":pp-(p['High']-p['Low'])}
    except: return None

def calculate_dynamic_levels(df):
    try:
        sma = df['Close'].rolling(20).mean().iloc[-1]; std = df['Close'].rolling(20).std().iloc[-1]
        return {"EMA 20":df['Close'].ewm(span=20).mean().iloc[-1], "EMA 50":df['Close'].ewm(span=50).mean().iloc[-1], "EMA 200":df['Close'].ewm(span=200).mean().iloc[-1], "BB Upper":sma+(2*std), "BB Lower":sma-(2*std), "Current":df['Close'].iloc[-1]}
    except: return None

def calculate_static_round_numbers(price):
    step = 50000 if price > 2000000 else 10000 if price > 100000 else 1000
    base = (price // step) * step
    return {"Res 2":base+(step*2), "Res 1":base+step, "Sup 1":base, "Sup 2":base-step}

def calculate_bitkub_ai_levels(h, l, c):
    pp=(h+l+c)/3; rng=h-l; mid=(h+l)/2
    st, cl = ("BULLISH", "#00E676") if c > mid else ("BEARISH", "#FF1744")
    return {
        "levels": [{"name":"🚀 R2","price":pp+rng,"type":"res"}, {"name":"🛑 R1","price":(2*pp)-l,"type":"res"}, {"name":"⚖️ PV","price":pp,"type":"neu"}, {"name":"🛡️ S1","price":(2*pp)-h,"type":"sup"}, {"name":"💎 S2","price":pp-rng,"type":"sup"}],
        "fib": {"top": l+(rng*0.618), "bot": l+(rng*0.382)}, "status": st, "color": cl, "insight": f"เหนือ Pivot {pp:,.0f}" if c>pp else f"ต่ำกว่า Pivot {pp:,.0f}"
    }

def calculate_heikin_ashi(df):
    ha = df.copy()
    ha['Close'] = (df['Open']+df['High']+df['Low']+df['Close'])/4
    ha['Open'] = [ (df['Open'][0]+df['Close'][0])/2 ] + [0]*(len(df)-1)
    for i in range(1, len(df)): ha['Open'].iloc[i] = (ha['Open'].iloc[i-1]+ha['Close'].iloc[i-1])/2
    ha['High'] = ha[['High','Open','Close']].max(axis=1)
    ha['Low'] = ha[['Low','Open','Close']].min(axis=1)
    return ha

def get_sector_pe_benchmark(sector):
    bench = {'Technology': 25, 'Financial Services': 15, 'Healthcare': 22, 'Energy': 12}
    return bench.get(sector, 20)

# --- ANALYSIS LOGIC ---
def analyze_stock_guru(info, setup, symbol):
    pe = info.get('trailingPE'); peg = info.get('pegRatio'); pb = info.get('priceToBook')
    roe = info.get('returnOnEquity'); pm = info.get('profitMargins')
    rev = info.get('revenueGrowth'); sec = info.get('sector', 'General')
    
    vs, qs, rv, rq = 0, 0, [], []
    
    # Quality
    if roe and roe>0.15: qs+=1; rq.append("✅ ROE สูง (>15%)")
    elif roe and roe<0: rq.append("❌ ROE ติดลบ")
    if pm and pm>0.1: qs+=1; rq.append("✅ Margin ดี (>10%)")
    if rev and rev>0: qs+=1; rq.append("✅ รายได้โต")
    else: rq.append("⚠️ รายได้หดตัว")
    
    # Valuation
    if pe:
        if pe<15: vs+=3; rv.append("✅ P/E ต่ำ")
        elif pe<25: vs+=2; rv.append("⚖️ P/E เหมาะสม")
        else: vs+=1; rv.append("⚠️ P/E สูง")
    else: vs+=1
    
    if peg:
        if peg<1: vs+=3; rv.append("✅ PEG ต่ำ (คุ้ม)")
        elif peg<2: vs+=2
        else: rv.append("❌ PEG สูง")
    
    if pb and pb<3: vs+=2
    vs = min(10, vs+qs) # Combine Quality into Valuation Score for simplicity here

    if vs>=8: vd, cl = "💎 Hidden Gem", "#00E676"
    elif vs>=5: vd, cl = "⚖️ Fair Value", "#FFD600"
    else: vd, cl = "⚠️ High Risk", "#FF1744"
    
    # Article
    art = f"**บทวิเคราะห์ {symbol} ({sec})**\n\n"
    art += f"1. **ความคุ้มค่า:** P/E {pe:.2f} " + ("ถูกกว่าตลาด" if pe and pe<20 else "สูงกว่าตลาด") + "\n\n"
    art += f"2. **คุณภาพ:** ROE {roe*100 if roe else 0:.1f}% " + ("บริหารดีเยี่ยม" if roe and roe>0.15 else "พอใช้") + "\n\n"
    art += f"3. **เทคนิค:** {setup['trend']} " + ("แนะนำสะสม" if vs>=7 else "ระวังแรงขาย")
    
    return {"verdict":vd, "color":cl, "val_score":vs, "article":art, "reasons_q":rq, "reasons_v":rv}

def analyze_smart_sr_strategy(price, pivots, dynamics, guru_data):
    lvls = {**pivots, **{k:v for k,v in dynamics.items() if k!='Current'}}
    n_lvl, min_d, n_price = "", float('inf'), 0
    for k,v in lvls.items():
        if abs(price-v) < min_d: min_d, n_lvl, n_price = abs(price-v), k, v
            
    at_lvl = (min_d/price)*100 < 1.0
    f_score = guru_data['val_score'] if guru_data else 5
    
    if at_lvl:
        if price > n_price: # Support
            if f_score>=7: msg, cl, ic = f"💎 รับ {n_lvl} + พื้นฐานดี", "#00E676", "🚀"
            else: msg, cl, ic = f"🛡️ เด้งสั้น {n_lvl}", "#00E5FF", "🛡️"
        else: # Resist
            if f_score>=7: msg, cl, ic = f"📈 ลุ้นเบรค {n_lvl}", "#FFD600", "👀"
            else: msg, cl, ic = f"🧱 ชนต้าน {n_lvl} ขาย", "#FF1744", "💰"
    else:
        msg, cl, ic = (f"🏃 วิ่งหา {n_lvl}", "#00E676", "🌊") if f_score>=5 else (f"⏳ รอย่อ", "#888", "💤")
            
    return msg, cl, ic, n_lvl, n_price

def generate_dynamic_insight(price, pivots, dynamics):
    e200 = dynamics['EMA 200']
    msg, col, icon = ("Bullish Strong", "#00E676", "🐂") if price > e200 else ("Bearish Strong", "#FF1744", "🐻")
    
    all_lvls = {**pivots, **{k:v for k,v in dynamics.items() if k!='Current'}}
    n_name, n_price, min_d = "", 0, float('inf')
    for k,v in all_lvls.items():
        if abs(price-v) < min_d: min_d, n_name, n_price = abs(price-v), k, v
    
    act = f"⚠️ ทดสอบ {n_name}" if (min_d/price)*100 < 0.8 else f"🏃 วิ่งหา {n_name}"
    return msg, col, icon, act

def analyze_bitkub_static_guru(last, static_levels):
    r1, s1 = static_levels['Res 1'], static_levels['Sup 1']
    if last >= r1: return "🚀 BREAKOUT", "#00E676", f"ทะลุ {r1:,.0f}", "Follow Trend"
    elif last <= s1: return "🩸 BREAKDOWN", "#FF1744", f"หลุด {s1:,.0f}", "Wait & See"
    else: return "⚖️ RANGE", "#FFD600", f"กรอบ {s1:,.0f}-{r1:,.0f}", "Swing"

def gen_ai_verdict(setup, news):
    sc, t_t, n_t = 50, "", ""
    if "UP" in setup['trend']: sc+=20; t_t="ขาขึ้น"
    elif "DOWN" in setup['trend']: sc-=20; t_t="ขาลง"
    else: t_t="ออกข้าง"
    
    n_sc = sum([n['score'] for n in news]) if news else 0
    if n_sc>0.3: sc+=15; n_t="ข่าวบวก"
    elif n_sc<-0.3: sc-=15; n_t="ข่าวลบ"
    else: n_t="ข่าวทรงตัว"
    
    sc = max(0, min(100, sc))
    vd = "BUY" if sc>=60 else "SELL" if sc<=40 else "HOLD"
    return t_t, n_t, sc, vd

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("BTC"): set_symbol("BTC-USD")
    if c2.button("ETH"): set_symbol("ETH-USD")
    st.markdown("---")
    bk_data = get_bitkub_ticker()
    if bk_data:
        b = bk_data.get('THB_BTC',{})
        e = bk_data.get('THB_ETH',{})
        st.markdown(f"**BTC:** <span style='color:#00E676'>{b.get('last',0):,.0f}</span>", unsafe_allow_html=True)
        st.markdown(f"**ETH:** <span style='color:#00E676'>{e.get('last',0):,.0f}</span>", unsafe_allow_html=True)
    st.markdown("---")
    chart_type = st.selectbox("Style", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Period", ["1mo","3mo","6mo","1y"], value="6mo")

# --- 5. Main ---
st.markdown("<h2 style='color:#00E5FF;'>🔍 Smart Search</h2>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 1]) 
with c1: sym_input = st.text_input("Symbol", st.session_state.symbol, label_visibility="collapsed")
with c2: 
    if st.button("วิเคราะห์ ⚡", use_container_width=True): set_symbol(sym_input); st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner("🚀 AI Analyzing..."):
        df = get_market_data(symbol, period, "1d")
    
    if not df.empty:
        curr = df['Close'].iloc[-1]
        chg = curr - df['Close'].iloc[-2]
        pct = (chg / df['Close'].iloc[-2]) * 100
        color = "#00E676" if chg >= 0 else "#FF1744"
        
        setup = calculate_technical_setup(df)
        news = get_ai_analyzed_news_thai(symbol)
        info = get_stock_info(symbol)
        t_txt, n_txt, ai_sc, ai_vd = gen_ai_verdict(setup, news)
        sc_col = "#00E676" if ai_sc>=70 else "#FF1744" if ai_sc<=30 else "#FFD600"
        sc_glow = "0,230,118" if ai_sc>=70 else "255,23,68" if ai_sc<=30 else "255,214,0"

        st.markdown(f"""<div class="glass-card" style="border-top:5px solid {color};text-align:center;"><div style="font-size:3.5rem;font-weight:900;">{symbol}</div><div style="font-size:3rem;color:{color};font-weight:bold;">{curr:,.2f}</div><div style="background:rgba({sc_glow},0.2);padding:5px 20px;border-radius:20px;display:inline-block;"><span style="color:{color};font-weight:bold;">{chg:+.2f} ({pct:+.2f}%)</span></div></div>""", unsafe_allow_html=True)

        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R Hybrid", "🧠 AI Guru", "🇹🇭 Bitkub AI", "🧮 Calc"])

        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['Open'], high=ha['High'], low=ha['HA_Low'], close=ha['Close'], name="HA"), row=1, col=1)
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
                sm = info.get('longBusinessSummary','')
                if HAS_TRANSLATOR: 
                    try: sm = GoogleTranslator(source='auto', target='th').translate(sm[:2000])
                    except: pass
                with st.expander(f"🏢 เกี่ยวกับ {symbol}"): st.write(sm)
                pe = info.get('trailingPE')
                sec = info.get('sector','-')
                c1,c2=st.columns(2)
                c1.markdown(f"<div class='metric-box'><div class='metric-label'>P/E Ratio</div><div class='metric-val'>{pe if pe else 'N/A'}</div></div>", unsafe_allow_html=True)
                if pe:
                    avg = get_sector_pe_benchmark(sec)
                    dfp = ((pe-avg)/avg)*100
                    stt,cl = ("แพงกว่า","red") if dfp>0 else ("ถูกกว่า","green")
                    c2.markdown(f"<div class='metric-box'><div class='metric-label'>Sector ({avg})</div><div class='metric-val' style='color:{cl}'>{stt} ({abs(dfp):.1f}%)</div></div>", unsafe_allow_html=True)

        with tabs[2]:
            if news:
                for n in news: st.markdown(f"""<div class="news-card {n['class']}"><div style="display:flex;justify-content:space-between;"><div>{n['icon']} <b>{n['label']}</b></div><span style="font-size:0.8rem;background:#333;padding:2px 8px;border-radius:5px;">{n['source']}</span></div><h4 style="margin:10px 0;color:#e0e0e0;">{n['title']}</h4><p style="color:#aaa;font-size:0.9rem;">{n['summary']}</p><div style="text-align:right;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;">อ่านต่อ</a></div></div>""", unsafe_allow_html=True)
            else: st.info("ไม่พบข่าว")

        with tabs[3]:
            st.markdown(f"""<div class='ai-insight-box' style='border-left:5px solid {setup['color']};margin-bottom:20px;'><h2 style='margin:0;color:{setup['color']};'>{setup['signal']}</h2><p style='font-size:1.2rem;color:#ccc;'>{setup['trend']}</p></div>""", unsafe_allow_html=True)
            c1,c2,c3=st.columns(3)
            c1.markdown(f"<div class='metric-box' style='border-left-color:#00E5FF'><div class='metric-label'>Buy Zone</div><div class='metric-val'>{curr*0.99:,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box' style='border-left-color:#00E676'><div class='metric-label'>Target</div><div class='metric-val'>{setup['tp']:,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box' style='border-left-color:#FF1744'><div class='metric-label'>Stop</div><div class='metric-val'>{setup['sl']:,.2f}</div></div>", unsafe_allow_html=True)

        with tabs[4]:
            c1,c2 = st.columns([1,1.5])
            with c1: st.markdown(f"""<div class="verdict-ring" style="border-color:{sc_col};color:{sc_col};box-shadow:0 0 30px rgba({sc_glow},0.5);">{ai_sc}</div><div style="text-align:center;font-size:2rem;font-weight:900;color:{sc_col};">{ai_vd}</div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="metric-box" style="border-left-color:{sc_col};"><h4 style="margin:0;">📈 Tech</h4><p>{t_txt}</p></div><div class="metric-box" style="border-left-color:{'#00E676' if 'บวก' in n_txt else '#FF1744'};"><h4 style="margin:0;">📰 News</h4><p>{n_txt}</p></div>""", unsafe_allow_html=True)

        with tabs[5]: # S/R Hybrid
            pivots = calculate_pivot_points(df)
            dynamic = calculate_dynamic_levels(df)
            if info and pivots and dynamic:
                guru = analyze_stock_guru(info, setup, symbol)
                msg_s, col_s, icon_s, lvl_s, pr_s = analyze_smart_sr_strategy(curr, pivots, dynamic, guru)
                st.markdown(f"""<div class='ai-insight-box' style='border:2px solid {col_s};margin-bottom:25px;'><div style="display:flex;align-items:center;gap:15px;"><span style="font-size:2.5rem;">{icon_s}</span><div><h2 style="margin:0;color:{col_s};">{msg_s}</h2><p style="color:#ddd;margin:5px 0;">Strategy: <b style="color:{guru['color']}">{guru['verdict']}</b></p></div></div></div>""", unsafe_allow_html=True)
            elif pivots and dynamic:
                msg, col, icon, act = generate_dynamic_insight(curr, pivots, dynamic)
                st.markdown(f"""<div class='ai-insight-box' style='border-color:{col};'><h3 style='margin:0;color:{col};'>{msg}</h3><p>{act}</p></div>""", unsafe_allow_html=True)

            c1,c2=st.columns(2)
            with c1:
                st.markdown("#### 🧱 Static S/R")
                for k,v in pivots.items(): 
                    cl="sr-res" if "R" in k else "sr-sup" if "S" in k else "sr-piv"
                    st.markdown(f"<div class='sr-card {cl}'><b>{k}</b><span>{v:,.2f}</span></div>", unsafe_allow_html=True)
            with c2:
                st.markdown("#### 🌊 Dynamic S/R")
                for k,v in dynamic.items():
                    if k!="Current": 
                        cl = "#00E676" if curr>v else "#FF1744"
                        st.markdown(f"<div class='sr-card' style='border-left:4px solid {cl};'><span>{k}</span><span>{v:,.2f}</span></div>", unsafe_allow_html=True)

        with tabs[6]:
            if info:
                guru = analyze_stock_guru(info, setup, symbol)
                st.markdown(f"""<div class='ai-insight-box' style='border:2px solid {guru['color']};text-align:center;margin-bottom:20px;'><h1 style='color:{guru['color']};font-size:3rem;margin:0;'>{guru['verdict']}</h1><div style="margin:20px 0;background:#333;border-radius:10px;height:10px;width:100%;"><div style="width:{guru['val_score']*10}%;background:{guru['color']};height:100%;border-radius:10px;"></div></div><p>Score: {guru['val_score']}/10</p></div><div class='ai-article'>{guru['article']}</div>""", unsafe_allow_html=True)
                c1,c2=st.columns(2)
                with c1:
                    for r in guru['reasons_q']: st.markdown(f"<div class='guru-card' style='border-left:4px solid #00E676'>{r}</div>", unsafe_allow_html=True)
                with c2:
                    for r in guru['reasons_v']: st.markdown(f"<div class='guru-card' style='border-left:4px solid #00E676'>{r}</div>", unsafe_allow_html=True)
            else: st.info("No Data")

        with tabs[7]:
            bk_sel = st.radio("Coin", ["BTC","ETH"], horizontal=True)
            if bk_data:
                pair = f"THB_{bk_sel}"; d = bk_data.get(pair,{})
                if d:
                    last, h24, l24 = d.get('last',0), d.get('high24hr',0), d.get('low24hr',0)
                    ai_bk = calculate_bitkub_ai_levels(h24, l24, last)
                    static = calculate_static_round_numbers(last)
                    bk_vd, bk_cl, bk_dc, bk_st = analyze_bitkub_static_guru(last, static)
                    st.markdown(f"""<div class='ai-insight-box' style='text-align:center;border:2px solid {ai_bk['color']};'><div style='font-size:3rem;font-weight:900;color:#fff;'>{d.get('last',0):,.0f} THB</div><div style='font-size:1.5rem;font-weight:bold;color:{ai_bk['color']};'>{ai_bk['status']}</div></div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class='ai-insight-box' style='border-color:{bk_cl};margin-top:15px;'><h3 style='margin:0;color:{bk_cl};'>{bk_vd}</h3><p>{bk_dc}</p><div style='background:rgba(255,255,255,0.05);padding:10px;border-radius:5px;margin-top:10px;'><b style='color:#00E5FF;'>Strategy:</b> {bk_st}</div></div>""", unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        for l in ai_bk['levels']:
                            cl = "#00E676" if l['type']=='sup' else "#FF1744" if l['type']=='res' else "#FFD600"
                            st.markdown(f"<div class='sr-card' style='border-left:5px solid {cl};'><b>{l['name']}</b><span>{l['price']:,.0f}</span></div>", unsafe_allow_html=True)
                    with c2:
                        st.info(f"Bot: {ai_bk['fib']['bot']:,.0f} | Top: {ai_bk['fib']['top']:,.0f}")
                else: st.error("No Data")
            else: st.warning("Connecting...")

        with tabs[8]:
            c1,c2=st.columns(2)
            with c1: 
                bal = st.number_input("Balance", value=100000.0, step=1000.0)
                risk = st.number_input("Risk %", value=1.0, step=0.1)
            with c2:
                ent = st.number_input("Entry", value=setup['entry'] if setup else curr)
                sl = st.number_input("Stop", value=setup['sl'] if setup else curr*0.95)
            if st.button("Calculate", use_container_width=True):
                if ent>0 and sl>0 and ent!=sl:
                    rps = abs(ent-sl); amt = bal*(risk/100); qty = amt/rps; cost = qty*ent
                    c1,c2,c3=st.columns(3)
                    c1.metric("Qty", f"{qty:,.2f}")
                    c2.metric("Cost", f"{cost:,.2f}")
                    c3.metric("Risk", f"{amt:,.2f}")
    else: st.error("No Data")
