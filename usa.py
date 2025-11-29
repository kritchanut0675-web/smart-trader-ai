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

# --- Library Setup ---
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except: HAS_TRANSLATOR = False

try: nltk.data.find('tokenizers/punkt')
except: nltk.download('punkt')

FINNHUB_KEY = "d4l5ku1r01qt7v18ll40d4l5ku1r01qt7v18ll4g" 

st.set_page_config(page_title="Smart Trader AI", layout="wide", page_icon="💎")
if 'symbol' not in st.session_state: st.session_state.symbol = 'BTC-USD'
def set_symbol(sym): st.session_state.symbol = sym

# --- CSS Minified ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Kanit',sans-serif;}
.stApp{background-color:#050505!important;color:#e0e0e0;}
div[data-testid="stTextInput"] input{background-color:#111!important;color:#fff!important;border:2px solid #00E5FF!important;border-radius:10px;}
.glass-card{background:linear-gradient(145deg,#1a1a1a,#0d0d0d);border:1px solid #333;border-radius:20px;padding:25px;margin-bottom:20px;box-shadow:0 8px 32px rgba(0,0,0,0.5);}
.metric-box{background:#111;border-radius:15px;padding:20px;border-left:4px solid #333;transition:0.2s;}
.metric-box:hover{transform:translateY(-5px);border-left-color:#00E5FF;}
.metric-val{font-size:1.8rem;font-weight:800;color:#fff;margin-top:5px;}
.sr-card{padding:15px;border-radius:12px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;border:1px solid #333;}
.sr-res{background:linear-gradient(90deg,rgba(255,23,68,0.2),#000);border-left:5px solid #FF1744;}
.sr-sup{background:linear-gradient(90deg,rgba(0,230,118,0.2),#000);border-left:5px solid #00E676;}
.sr-piv{background:linear-gradient(90deg,rgba(255,214,0,0.2),#000);border-left:5px solid #FFD600;}
.verdict-ring{width:140px;height:140px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:3rem;font-weight:900;margin:0 auto 20px;border:8px solid #333;background:#000;}
.ai-insight-box{background:linear-gradient(135deg,#111,#0a0a0a);border:1px solid #333;border-radius:15px;padding:25px;overflow:hidden;}
.news-card{padding:20px;margin-bottom:15px;background:#111;border-radius:15px;border-left:5px solid #888;transition:0.2s;}
.news-card:hover{transform:translateX(5px);background:#161616;}
.nc-pos{border-left-color:#00E676;} .nc-neg{border-left-color:#FF1744;} .nc-neu{border-left-color:#FFD600;}
.guru-card{background:#111;padding:15px;border-radius:12px;border:1px solid #333;margin-bottom:10px;}
.ai-article{background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;border-left:4px solid #00E5FF;margin-top:20px;line-height:1.8;}
div.stButton>button{width:100%;font-size:1.1rem!important;padding:12px!important;border-radius:12px!important;background:linear-gradient(45deg,#00E5FF,#2979FF);border:none!important;color:#000!important;font-weight:800!important;}
button[data-baseweb="tab"]{font-size:1rem!important;font-weight:600!important;border-radius:8px!important;background:#111!important;border:1px solid #333!important;margin:0 2px!important;}
button[data-baseweb="tab"][aria-selected="true"]{background:#00E5FF!important;color:#000!important;}
</style>
""", unsafe_allow_html=True)

# --- Functions ---
@st.cache_data(ttl=300)
def get_data(sym, p, i): 
    try: return yf.Ticker(sym).history(period=p, interval=i)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_info(sym): 
    try: return yf.Ticker(sym).info
    except: return None

@st.cache_data(ttl=15)
def get_bitkub():
    try: return requests.get("https://api.bitkub.com/api/market/ticker", timeout=5).json()
    except: return None

def get_finnhub(sym):
    try:
        t = datetime.date.today()
        f = t - datetime.timedelta(days=2)
        s = sym.split("-")[0]
        return requests.get(f"https://finnhub.io/api/v1/company-news?symbol={s}&from={f}&to={t}&token={FINNHUB_KEY}").json()[:5]
    except: return []

@st.cache_data(ttl=3600)
def get_news(sym):
    lst = []
    tr = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    
    fh = get_finnhub(sym)
    if fh:
        for i in fh:
            sc = TextBlob(i.get('headline','')).sentiment.polarity
            lbl, ic, cl = ("ข่าวดี","🚀","nc-pos") if sc>0.05 else ("ข่าวร้าย","🔻","nc-neg") if sc<-0.05 else ("ทั่วไป","⚖️","nc-neu")
            t, s = i.get('headline',''), i.get('summary','')
            if tr:
                try: t=tr.translate(t); s=tr.translate(s) if s else ""
                except: pass
            lst.append({'title':t, 'summary':s, 'link':i.get('url','#'), 'icon':ic, 'class':cl, 'label':lbl, 'score':sc, 'source':'Finnhub'})

    if len(lst)<3:
        try:
            q = urllib.parse.quote(f"{sym.split('-')[0]} finance news")
            fd = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            for i in fd.entries[:5]:
                sc = TextBlob(i.title).sentiment.polarity
                lbl, ic, cl = ("ข่าวดี","🚀","nc-pos") if sc>0.05 else ("ข่าวร้าย","🔻","nc-neg") if sc<-0.05 else ("ทั่วไป","⚖️","nc-neu")
                t, s = i.title, re.sub(r'<.*?>','', getattr(i,'summary','') or getattr(i,'description',''))[:300]
                if tr:
                    try: t=tr.translate(t); s=tr.translate(s) if s else ""
                    except: pass
                lst.append({'title':t, 'summary':s, 'link':i.link, 'icon':ic, 'class':cl, 'label':lbl, 'score':sc, 'source':'Google'})
        except: pass
    return lst[:10]

def tech_setup(df):
    c = df['Close'].iloc[-1]
    e50 = df['Close'].ewm(span=50).mean().iloc[-1]
    e200 = df['Close'].ewm(span=200).mean().iloc[-1]
    atr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
    
    delta = df['Close'].diff()
    rs = (delta.where(delta>0,0)).rolling(14).mean() / (-delta.where(delta<0,0)).rolling(14).mean()
    rsi_s = 100 - (100/(1+rs))
    
    if c > e50 and e50 > e200: t,s,co,sc = "UPTREND", "BUY", "#00E676", 2
    elif c < e50 and e50 < e200: t,s,co,sc = "DOWNTREND", "SELL", "#FF1744", -2
    else: t,s,co,sc = "SIDEWAYS", "WAIT", "#FFD600", 0
    
    return {'trend':t, 'signal':s, 'color':co, 'rsi_s':rsi_s, 'rsi':rsi_s.iloc[-1], 'entry':c, 'sl':c-(1.5*atr) if sc>=0 else c+(1.5*atr), 'tp':c+(2.5*atr) if sc>=0 else c-(2.5*atr)}

def calc_pivots(df):
    p = df.iloc[-2]
    pp = (p['High']+p['Low']+p['Close'])/3
    return {"PP":pp, "R1":(2*pp)-p['Low'], "S1":(2*pp)-p['High'], "R2":pp+(p['High']-p['Low']), "S2":pp-(p['High']-p['Low'])}

def calc_dynamic(df):
    sma = df['Close'].rolling(20).mean().iloc[-1]
    std = df['Close'].rolling(20).std().iloc[-1]
    return {"EMA20":df['Close'].ewm(span=20).mean().iloc[-1], "EMA50":df['Close'].ewm(span=50).mean().iloc[-1], "EMA200":df['Close'].ewm(span=200).mean().iloc[-1], "BBUp":sma+(2*std), "BBLow":sma-(2*std), "Cur":df['Close'].iloc[-1]}

def gen_insight(p, piv, dyn):
    e200 = dyn['EMA200']
    msg, col, ic = ("Bullish Strong", "#00E676", "🐂") if p > e200 else ("Bearish Strong", "#FF1744", "🐻")
    all_lvls = {**piv, **{k:v for k,v in dyn.items() if k!='Cur'}}
    n_n, n_p, min_d = "", 0, float('inf')
    for k,v in all_lvls.items():
        if abs(p-v) < min_d: min_d, n_n, n_p = abs(p-v), k, v
    act = f"⚠️ Testing **{n_n}**" if (min_d/p)*100 < 0.8 else f"🏃 Room to run to **{n_n}**"
    return msg, col, ic, act

def analyze_guru(info, setup, sym):
    qs, vs, rq, rv = 0, 0, [], []
    pe, peg, pb = info.get('trailingPE'), info.get('pegRatio'), info.get('priceToBook')
    roe, pm, rev = info.get('returnOnEquity',0), info.get('profitMargins',0), info.get('revenueGrowth',0)
    
    if roe and roe>0.15: qs+=1; rq.append("✅ ROE สูง (>15%)")
    elif roe and roe<0: rq.append("❌ ROE ติดลบ")
    if pm and pm>0.1: qs+=1; rq.append("✅ Margin ดี")
    if rev and rev>0: qs+=1; rq.append("✅ รายได้โต")
    else: rq.append("⚠️ รายได้หดตัว")
    
    if pe:
        if pe<15: vs+=3; rv.append("✅ P/E ถูก")
        elif pe<25: vs+=2; rv.append("⚖️ P/E กลางๆ")
        else: vs+=1; rv.append("⚠️ P/E แพง")
    else: vs+=1
    
    if peg:
        if peg<1: vs+=3; rv.append("✅ PEG ต่ำ (คุ้ม)")
        elif peg<2: vs+=2; rv.append("⚖️ PEG ปกติ")
        else: rv.append("❌ PEG สูง")
    
    if pb and pb<3: vs+=2
    if roe and roe>0.15: vs+=2
    vs = min(10, vs)
    
    intro = f"หุ้น **{sym}** ในกลุ่ม **{info.get('sector','-')}**:\n\n"
    val_txt = "ราคาถูกมาก (Undervalued) " if vs>=8 else "ราคาเหมาะสม (Fair) " if vs>=5 else "ราคาแพง (Overvalued) "
    qual_txt = "คุณภาพบริษัทดีเยี่ยม " if qs>=2 else "คุณภาพบริษัทปานกลาง "
    tech_txt = f"กราฟเป็น **{setup['trend']}** "
    strat = "แนะนำ **'ทยอยสะสม'**" if setup['trend']=="UPTREND" and vs>=7 else "แนะนำ **'เก็งกำไรสั้นๆ'**"
    
    if vs>=8: vd, cl = "💎 Hidden Gem", "#00E676"
    elif vs>=5: vd, cl = "⚖️ Fair Value", "#FFD600"
    else: vd, cl = "⚠️ High Risk", "#FF1744"
    
    return {"verdict":vd, "color":cl, "score":vs, "article":intro+val_txt+qual_txt+tech_txt+strat, "rq":rq, "rv":rv}

def gen_verdict(setup, news):
    sc, t_t, n_t = 50, "", ""
    if setup['trend']=="UPTREND": sc+=20; t_t="กราฟขาขึ้นชัดเจน"
    elif setup['trend']=="DOWNTREND": sc-=20; t_t="กราฟขาลงชัดเจน"
    else: t_t="กราฟออกข้าง"
    
    n_sc = sum([n['score'] for n in news]) if news else 0
    if n_sc>0.3: sc+=15; n_t="ข่าวบวกหนุน"
    elif n_sc<-0.3: sc-=15; n_t="ข่าวลบกดดัน"
    else: n_t="ข่าวทรงตัว"
    
    sc = max(0, min(100, sc))
    vd = "STRONG BUY" if sc>=80 else "BUY" if sc>=60 else "SELL" if sc<=40 else "STRONG SELL" if sc<=20 else "HOLD"
    return t_t, n_t, sc, vd

def calc_bk_ai(h, l, c):
    pp=(h+l+c)/3; rng=h-l; mid=(h+l)/2
    st, cl = ("BULLISH", "#00E676") if c > mid else ("BEARISH", "#FF1744")
    ins = f"ราคายืนเหนือ Pivot ({pp:,.0f}) ได้" if c > pp else f"ราคาหลุด Pivot ({pp:,.0f}) ระวังลงต่อ"
    return {
        "levels": [{"name":"🚀 R2","p":pp+rng,"t":"res"}, {"name":"🛑 R1","p":(2*pp)-l,"t":"res"}, {"name":"⚖️ PV","p":pp,"t":"neu"}, {"name":"🛡️ S1","p":(2*pp)-h,"t":"sup"}, {"name":"💎 S2","p":pp-rng,"t":"sup"}],
        "fib": {"top": l+(rng*0.618), "bot": l+(rng*0.382)}, "status": st, "color": cl, "insight": ins
    }

# --- UI ---
st.markdown("<h2 style='color:#00E5FF;'>🔍 Smart Search</h2>", unsafe_allow_html=True)
c1,c2 = st.columns([3,1])
with c1: sym_in = st.text_input("Symbol", st.session_state.symbol, label_visibility="collapsed")
with c2: 
    if st.button("วิเคราะห์ ⚡", use_container_width=True): set_symbol(sym_in); st.rerun()

with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    if c1.button("BTC"): set_symbol("BTC-USD")
    if c2.button("ETH"): set_symbol("ETH-USD")
    bk = get_bitkub_ticker()
    if bk:
        b = bk.get('THB_BTC',{}).get('last',0)
        e = bk.get('THB_ETH',{}).get('last',0)
        st.markdown("---")
        st.markdown(f"**BTC:** <span style='color:#00E676'>{b:,.0f}</span>", unsafe_allow_html=True)
        st.markdown(f"**ETH:** <span style='color:#00E676'>{e:,.0f}</span>", unsafe_allow_html=True)
    st.markdown("---")
    c_type = st.selectbox("Chart", ["Candle", "Heikin"])
    tf = st.select_slider("TF", ["1mo","3mo","6mo","1y"], value="6mo")

sym = st.session_state.symbol.upper()
if sym:
    with st.spinner("Analyzing..."):
        df = get_data(sym, tf, "1d")
    if not df.empty:
        curr = df['Close'].iloc[-1]
        chg = curr - df['Close'].iloc[-2]
        pct = (chg/df['Close'].iloc[-2])*100
        col = "#00E676" if chg>=0 else "#FF1744"
        
        setup = tech_setup(df)
        news = get_ai_analyzed_news_thai(sym)
        info = get_stock_info(sym)
        t_txt, n_txt, ai_sc, ai_vd = gen_verdict(setup, news)
        
        sc_cl, sc_gl = ("#00E676","0,230,118") if ai_sc>=70 else ("#FF1744","255,23,68") if ai_sc<=30 else ("#FFD600","255,214,0")
        
        st.markdown(f"""<div class="glass-card" style="border-top:5px solid {col};text-align:center;"><div style="font-size:3.5rem;font-weight:900;line-height:1;">{sym}</div><div style="font-size:3rem;color:{col};font-weight:bold;">{curr:,.2f}</div><div style="background:rgba({sc_gl},0.2);padding:5px 20px;border-radius:20px;display:inline-block;margin-top:10px;"><span style="color:{col};font-weight:bold;font-size:1.1rem;">{chg:+.2f} ({pct:+.2f}%)</span></div></div>""", unsafe_allow_html=True)

        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R", "🧠 Guru", "🇹🇭 Bitkub"])

        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7,0.3], shared_xaxes=True)
            if c_type == "Heikin":
                ha = df.copy(); ha['Close']=(df['Open']+df['High']+df['Low']+df['Close'])/4
                fig.add_trace(go.Candlestick(x=df.index, open=ha['Open'], high=ha['High'], low=ha['Low'], close=ha['Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF'), name="EMA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=setup['rsi_s'], line=dict(color='#E040FB'), name="RSI"), row=2, col=1)
            fig.update_layout(template='plotly_dark', height=500, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            c1,c2,c3 = st.columns(3)
            c1.markdown(f"<div class='metric-box'><div class='metric-label'>High</div><div class='metric-val' style='color:#00E676'>{df['High'].max():,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><div class='metric-label'>Low</div><div class='metric-val' style='color:#FF1744'>{df['Low'].min():,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box'><div class='metric-label'>Vol</div><div class='metric-val' style='color:#E040FB'>{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>", unsafe_allow_html=True)
            if info:
                st.markdown("---")
                if HAS_TRANSLATOR:
                    try: sm = GoogleTranslator(source='auto', target='th').translate(info.get('longBusinessSummary','')[:2000])
                    except: sm = info.get('longBusinessSummary','')
                else: sm = info.get('longBusinessSummary','')
                with st.expander("🏢 Business Summary"): st.write(sm)
                c1,c2 = st.columns(2)
                pe = info.get('trailingPE')
                c1.markdown(f"<div class='metric-box'><div class='metric-label'>P/E</div><div class='metric-val'>{pe if pe else 'N/A'}</div></div>", unsafe_allow_html=True)
                if pe:
                    avg = 20 
                    df_pe = ((pe-avg)/avg)*100
                    stt, cl = ("แพงกว่ากลุ่ม","#FF1744") if df_pe>0 else ("ถูกกว่ากลุ่ม","#00E676")
                    c2.markdown(f"<div class='metric-box' style='border-left-color:{cl}'><div class='metric-label'>Sector ({avg})</div><div class='metric-val' style='color:{cl};font-size:1.4rem'>{stt} ({abs(df_pe):.1f}%)</div></div>", unsafe_allow_html=True)

        with tabs[2]:
            if news:
                for n in news: st.markdown(f"""<div class="news-card {n['class']}"><div style="display:flex;justify-content:space-between;"><div>{n['icon']} <b>{n['label']}</b></div><span style="font-size:0.8rem;background:#333;padding:2px 8px;border-radius:5px;">{n['source']}</span></div><h4 style="margin:10px 0;color:#e0e0e0;">{n['title']}</h4><p style="color:#aaa;font-size:0.9rem;">{n['summary']}</p><div style="text-align:right;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;">อ่านต่อ</a></div></div>""", unsafe_allow_html=True)
            else: st.info("No News")

        with tabs[3]:
            st.markdown(f"""<div class='ai-insight-box' style='border-left:5px solid {setup['color']};margin-bottom:20px;'><h2 style='margin:0;color:{setup['color']};'>{setup['signal']}</h2><p style='font-size:1.2rem;color:#ccc;'>{setup['trend']}</p><div style='margin-top:15px;display:flex;gap:10px;'><span style='background:#111;padding:5px 15px;border-radius:10px;border:1px solid #333;'>RSI: {setup['rsi']:.1f}</span><span style='background:#111;padding:5px 15px;border-radius:10px;border:1px solid #333;'>Entry: {setup['entry']:,.2f}</span></div></div>""", unsafe_allow_html=True)
            c1,c2,c3=st.columns(3)
            c1.markdown(f"<div class='metric-box' style='border-left-color:#00E5FF'><div class='metric-label'>Buy Zone</div><div class='metric-val'>{curr*0.99:,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box' style='border-left-color:#00E676'><div class='metric-label'>Target</div><div class='metric-val'>{setup['tp']:,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box' style='border-left-color:#FF1744'><div class='metric-label'>Stop</div><div class='metric-val'>{setup['sl']:,.2f}</div></div>", unsafe_allow_html=True)

        with tabs[4]:
            c1,c2 = st.columns([1,1.5])
            with c1: st.markdown(f"""<div class="verdict-ring" style="border-color:{sc_cl};color:{sc_cl};box-shadow:0 0 30px rgba({sc_gl},0.5);">{ai_sc}</div><div style="text-align:center;font-size:2rem;font-weight:900;color:{sc_cl};">{ai_vd}</div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="metric-box" style="border-left-color:{sc_cl};"><h4 style="margin:0;">📈 Tech</h4><p>{t_txt}</p></div><div class="metric-box" style="border-left-color:{'#00E676' if 'บวก' in n_txt else '#FF1744'};"><h4 style="margin:0;">📰 News</h4><p>{n_txt}</p></div>""", unsafe_allow_html=True)

        with tabs[5]:
            piv = calc_pivots(df); dyn = calc_dynamic(df)
            if piv and dyn:
                msg, cl, ic, act = gen_insight(curr, piv, dyn)
                st.markdown(f"""<div class='ai-insight-box' style='border-color:{cl};box-shadow:0 0 15px {cl}40;margin-bottom:25px;'><div class='ai-insight-icon'>{ic}</div><h3 style='margin:0;color:{cl};'>{msg}</h3><p style='font-size:1.1rem;color:#ccc;'>{act}</p></div>""", unsafe_allow_html=True)
                c1,c2=st.columns(2)
                with c1:
                    st.markdown("#### 🧱 Pivots")
                    for k,v in piv.items():
                        cl = "#FF1744" if "R" in k else "#00E676" if "S" in k else "#FFD600"
                        bg = "linear-gradient(90deg,rgba(255,23,68,0.2),rgba(0,0,0,0))" if "R" in k else "linear-gradient(90deg,rgba(0,230,118,0.2),rgba(0,0,0,0))" if "S" in k else "linear-gradient(90deg,rgba(255,214,0,0.2),rgba(0,0,0,0))"
                        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:12px;background:{bg};border-left:5px solid {cl};margin-bottom:8px;border-radius:5px;'><b>{k}</b><span>{v:,.2f}</span></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("#### 🌊 Dynamic")
                    for k,v in dyn.items():
                        if k!="Cur":
                            cl="#00E676" if curr>v else "#FF1744"
                            bg = "linear-gradient(90deg,rgba(0,230,118,0.2),rgba(0,0,0,0))" if curr>v else "linear-gradient(90deg,rgba(255,23,68,0.2),rgba(0,0,0,0))"
                            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:12px;background:{bg};border-left:5px solid {cl};margin-bottom:8px;border-radius:5px;'><span>{k}</span><div style='text-align:right;'>{v:,.2f}</div></div>", unsafe_allow_html=True)

        with tabs[6]:
            if info:
                guru = analyze_guru(info, setup, sym)
                st.markdown(f"""<div class='ai-insight-box' style='border:2px solid {guru['color']};text-align:center;margin-bottom:20px;'><h1 style='color:{guru['color']};font-size:3rem;margin:0;'>{guru['verdict']}</h1><div style="margin:20px 0;background:#333;border-radius:10px;height:10px;width:100%;"><div style="width:{guru['score']*10}%;background:{guru['color']};height:100%;border-radius:10px;"></div></div><p style='font-size:1.1rem;color:#ccc;'>Score: {guru['score']}/10</p></div><div class='ai-article'><h4 style='margin-top:0;color:#fff;'>📝 AI Analyst Report</h4>{guru['article']}</div>""", unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1: 
                    st.markdown("#### 🏢 Quality")
                    for r in guru['rq']: st.markdown(f"<div class='guru-card' style='border-left:4px solid {'#00E676' if '✅' in r else '#FF1744'}'>{r}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("#### ⚖️ Valuation")
                    for r in guru['rv']: st.markdown(f"<div class='guru-card' style='border-left:4px solid {'#00E676' if '✅' in r else '#FF1744'}'>{r}</div>", unsafe_allow_html=True)
            else: st.info("No Fundamental Data")

        with tabs[7]:
            bk_sel = st.radio("THB Pair", ["BTC","ETH"], horizontal=True)
            if bk_data:
                pair = f"THB_{bk_sel}"; d = bk_data.get(pair,{})
                if d:
                    ai_bk = calc_bk_ai(d.get('high24hr',0), d.get('low24hr',0), d.get('last',0))
                    st.markdown(f"""<div class='ai-insight-box' style='text-align:center;border:2px solid {ai_bk['color']};'><div style='font-size:3rem;font-weight:900;color:#fff;'>{d.get('last',0):,.0f} THB</div><div style='font-size:1.5rem;font-weight:bold;color:{ai_bk['color']};'>{ai_bk['status']}</div><p style='margin-top:10px;color:#ccc;'>🧠 {ai_bk['insight']}</p></div>""", unsafe_allow_html=True)
                    c1,c2 = st.columns(2)
                    with c1:
                        for l in ai_bk['levels']:
                            cl = "#00E676" if l['t']=='sup' else "#FF1744" if l['t']=='res' else "#FFD600"
                            st.markdown(f"<div class='sr-card' style='border-left:5px solid {cl};'><b>{l['name']}</b><span>{l['p']:,.0f}</span></div>", unsafe_allow_html=True)
                    with c2:
                        st.info(f"**Golden Bot:** {ai_bk['fib']['bot']:,.0f}\n\n**Golden Top:** {ai_bk['fib']['top']:,.0f}")
                        with st.expander("ℹ️ Golden Zone?"): st.write("โซนราคา Fibonacci (61.8% - 38.2%) ของ 24 ชม. ล่าสุด จุดวัดใจสำคัญ")
                else: st.error("No Data")
            else: st.warning("Connecting...")
