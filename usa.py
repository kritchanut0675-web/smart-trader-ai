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

# --- 2. CSS Styling (Ultra Modern UI) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
        
        .stApp { background-color: #050505 !important; color: #e0e0e0; }
        
        /* Input Field */
        div[data-testid="stTextInput"] input { 
            background-color: #111 !important; color: #fff !important; 
            font-weight: bold !important; font-size: 1.2rem !important;
            border: 2px solid #00E5FF !important; border-radius: 10px;
        }

        /* Cards */
        .glass-card {
            background: linear-gradient(145deg, #1a1a1a, #0d0d0d);
            border: 1px solid #333; border-radius: 20px;
            padding: 25px; margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        
        /* Stat Metric Box */
        .metric-box {
            background: #111; border-radius: 15px; padding: 20px;
            border-left: 4px solid #333; position: relative; overflow: hidden;
            transition: transform 0.2s;
        }
        .metric-box:hover { transform: translateY(-5px); border-left-color: #00E5FF; }
        .metric-label { font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .metric-val { font-size: 1.8rem; font-weight: 800; color: #fff; margin-top: 5px; }
        
        /* S/R Dynamic Cards */
        .sr-card {
            padding: 15px 20px; border-radius: 12px; margin-bottom: 10px;
            display: flex; justify-content: space-between; align-items: center;
            border: 1px solid rgba(255,255,255,0.05); backdrop-filter: blur(5px);
        }
        .sr-res { background: linear-gradient(90deg, rgba(255, 23, 68, 0.2), rgba(0,0,0,0)); border-left: 5px solid #FF1744; }
        .sr-sup { background: linear-gradient(90deg, rgba(0, 230, 118, 0.2), rgba(0,0,0,0)); border-left: 5px solid #00E676; }
        .sr-piv { background: linear-gradient(90deg, rgba(255, 214, 0, 0.2), rgba(0,0,0,0)); border-left: 5px solid #FFD600; }
        
        /* AI Verdict Ring */
        .verdict-ring {
            width: 140px; height: 140px; border-radius: 50%;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            font-size: 3rem; font-weight: 900; margin: 0 auto 20px auto;
            border: 8px solid #333; background: #000;
            box-shadow: 0 0 40px rgba(0,0,0,0.5);
        }
        
        /* AI Insight Box */
        .ai-insight-box {
            background: linear-gradient(135deg, #111, #0a0a0a);
            border: 1px solid #333; border-radius: 15px; padding: 25px;
            position: relative; overflow: hidden;
        }
        .ai-insight-icon { font-size: 2rem; margin-bottom: 10px; }
        
        /* NEWS CARD */
        .news-card { 
            padding: 20px; margin-bottom: 15px; background: #111; 
            border-radius: 15px; border-left: 5px solid #888; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .news-card:hover { transform: translateX(5px); background: #161616; }
        .nc-pos { border-left-color: #00E676; }
        .nc-neg { border-left-color: #FF1744; }
        .nc-neu { border-left-color: #FFD600; }
        
        /* GURU CARD */
        .guru-card {
            background: #111; padding: 15px; border-radius: 12px; 
            border: 1px solid #333; margin-bottom: 10px; font-size: 0.95rem;
        }
        
        /* AI Article */
        .ai-article {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px; border-radius: 15px;
            border-left: 4px solid #00E5FF;
            font-size: 1rem; line-height: 1.8; color: #ddd;
            margin-top: 20px;
        }

        /* Custom Tabs */
        button[data-baseweb="tab"] { 
            font-size: 1rem !important; font-weight: 600 !important; 
            border-radius: 8px !important; margin: 0 4px !important;
            background: #111 !important; border: 1px solid #333 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: #00E5FF !important; color: #000 !important; border-color: #00E5FF !important;
        }
        
        /* Centered Button */
        div.stButton > button {
            width: 100%; justify-content: center; font-size: 1.1rem !important; 
            padding: 12px !important; border-radius: 12px !important;
            background: linear-gradient(45deg, #00E5FF, #2979FF); 
            border: none !important; color: #000 !important; font-weight: 800 !important;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
        }
        div.stButton > button:hover {
            transform: scale(1.02); box-shadow: 0 0 25px rgba(0, 229, 255, 0.6);
        }
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

# --- AI Guru Analysis Logic (Corrected) ---
def analyze_stock_guru(info, setup, symbol):
    # 1. Get Data
    pe = info.get('trailingPE')
    peg = info.get('pegRatio')
    pb = info.get('priceToBook')
    roe = info.get('returnOnEquity', 0)
    profit_margin = info.get('profitMargins', 0)
    rev_growth = info.get('revenueGrowth', 0)
    sector = info.get('sector', 'General')
    
    # 2. Valuation Score (Max 10)
    val_score = 0
    reasons_q = []
    reasons_v = []

    if roe and roe > 0.15: 
        reasons_q.append("✅ ROE สูง (>15%) บริหารทุนเก่ง")
    elif roe and roe < 0:
        reasons_q.append("❌ ROE ติดลบ ขาดทุน")
        
    if profit_margin and profit_margin > 0.10: 
        reasons_q.append("✅ อัตรากำไรดี (>10%)")
    
    if rev_growth and rev_growth > 0: 
        reasons_q.append("✅ รายได้เติบโต")
    else:
        reasons_q.append("⚠️ รายได้ไม่โต หรือหดตัว")
    
    # PE Logic
    if pe:
        if pe < 15: 
            val_score += 3 
            reasons_v.append("✅ P/E ต่ำ (ถูก)")
        elif pe < 25: 
            val_score += 2 
            reasons_v.append("⚖️ P/E เหมาะสม")
        elif pe < 40: 
            val_score += 1 
            reasons_v.append("⚠️ P/E เริ่มสูง")
    else: val_score += 1
    
    # PEG Logic (Growth)
    if peg:
        if peg < 1.0: 
            val_score += 3 
            reasons_v.append("✅ PEG < 1 (โตคุ้มราคา)")
        elif peg < 2.0: 
            val_score += 2 
            reasons_v.append("⚖️ PEG ปกติ")
        else: 
            val_score += 0
            reasons_v.append("❌ PEG สูง (โตไม่ทันราคา)")
    
    # PB Logic
    if pb and pb < 3: val_score += 2
    
    # ROE Bonus
    if roe and roe > 0.15: val_score += 2

    val_score = min(10, val_score)

    # 3. Verdict Text Generation
    intro = f"จากการวิเคราะห์หุ้น **{symbol}** ในกลุ่มอุตสาหกรรม **{sector}** ด้วยระบบ AI Guru พบข้อมูลที่น่าสนใจดังนี้:\n\n"
    
    if pe:
        if pe < 15: val_text = f"ในมุมมองความคุ้มค่า (Valuation) หุ้นตัวนี้ถือว่า **'ราคาถูก (Undervalued)'** เมื่อเทียบกับกำไรที่ทำได้ โดยมีค่า P/E อยู่ที่ {pe:.2f} ซึ่งต่ำกว่าเกณฑ์มาตรฐาน "
        elif pe > 40: val_text = f"ในมุมมองความคุ้มค่า ราคาหุ้นปัจจุบันค่อนข้าง **'แพง (Overvalued)'** มีค่า P/E สูงถึง {pe:.2f} สะท้อนความคาดหวังของนักลงทุนที่สูงมาก "
        else: val_text = f"ราคาหุ้นปัจจุบันถือว่า **'สมเหตุสมผล (Fair Price)'** มีค่า P/E อยู่ที่ {pe:.2f} เป็นระดับที่ยอมรับได้ "
    else: val_text = "ไม่สามารถประเมินค่า P/E ได้ชัดเจน (อาจขาดทุนหรือเป็นกองทุน) "

    if peg:
        if peg < 1: val_text += f"และเมื่อเทียบกับการเติบโต (Growth) ถือว่าคุ้มค่ามาก (PEG {peg:.2f}) แปลว่ากำไรบริษัทโตเร็วกว่าราคาหุ้น "
        elif peg > 2: val_text += f"แต่เมื่อดูการเติบโตเทียบราคา (PEG {peg:.2f}) ถือว่าราคาขึ้นไปรอกำไรในอนาคตแล้ว (Growth Priced In) "
    
    qual_text = ""
    if roe and roe > 0.15: qual_text = f"\n\nด้านคุณภาพบริษัท (Quality) จัดว่ายอดเยี่ยม มี ROE สูงถึง {roe*100:.1f}% แสดงถึงความสามารถในการบริหารเงินทุนของผู้บริหารที่เก่งกาจ "
    elif profit_margin and profit_margin < 0.05: qual_text = f"\n\nด้านคุณภาพอาจต้องระวังเรื่องอัตรากำไรที่ค่อนข้างบาง ({profit_margin*100:.1f}%) ซึ่งอาจเปราะบางต่อเศรษฐกิจ "

    tech_text = f"\n\n**คำแนะนำเชิงกลยุทธ์:** เมื่อประกอบกับกราฟเทคนิคที่เป็น **{setup['trend']}** "
    if setup['trend'] == "UPTREND (ขาขึ้น)":
        if val_score >= 7: 
            tech_text += "และพื้นฐานที่แข็งแกร่ง **'แนะนำให้ทยอยสะสม (Buy)'** ได้ทันที เพราะทั้งพื้นฐานและเทคนิคสนับสนุนกัน ราคาเป้าหมายยังมี Upside"
        else:
            tech_text += "แม้เทคนิคจะดูดี แต่พื้นฐานเริ่มตึงตัว **'แนะนำให้เก็งกำไรระยะสั้น (Trading)'** และวางจุด Stop Loss อย่างเคร่งครัด ไม่ควรถือยาว"
    elif setup['trend'] == "DOWNTREND (ขาลง)":
        if val_score >= 8:
            tech_text += "แม้พื้นฐานจะดีและราคาถูกมาก แต่กราฟยังเป็นขาลง **'แนะนำให้ Wait & See'** รอให้กราฟสร้างฐานหรือยืนเหนือเส้น EMA ก่อนค่อยเข้าซื้อ จะได้ของดีในราคาที่ปลอดภัยกว่า"
        else:
            tech_text += "ประกอบกับพื้นฐานที่อ่อนแอ/แพง **'แนะนำให้หลีกเลี่ยง (Avoid)'** ไปก่อน จนกว่าจะมีสัญญาณการกลับตัวที่ชัดเจน"
    else:
        tech_text += "ควรรอให้ราคาเลือกทางที่ชัดเจนก่อนเข้าลงทุน (Wait for Breakout)"

    full_article = intro + val_text + qual_text + tech_text

    # 4. Determine Status
    if val_score >= 8: status, color = "💎 Hidden Gem (ของดีราคาถูก)", "#00E676"
    elif val_score >= 5: status, color = "⚖️ Fair Value (เหมาะสม)", "#FFD600"
    else: status, color = "⚠️ High Risk / Expensive", "#FF1744"

    return {
        "verdict": status,
        "color": color,
        "val_score": val_score,
        "article": full_article,
        "reasons_q": reasons_q,
        "reasons_v": reasons_v
    }

def get_sector_pe_benchmark(sector):
    benchmarks = {
        'Technology': 25, 'Financial Services': 15, 'Healthcare': 22,
        'Consumer Cyclical': 20, 'Industrials': 20, 'Energy': 12,
        'Utilities': 18, 'Real Estate': 35, 'Basic Materials': 15,
        'Communication Services': 20
    }
    return benchmarks.get(sector, 20) 

@st.cache_data(ttl=15)
def get_bitkub_ticker():
    try:
        r = requests.get("https://api.bitkub.com/api/market/ticker", timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

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
            
            if sc > 0.05: lbl, icon, cls = "ข่าวดี (Positive)", "🚀", "nc-pos"
            elif sc < -0.05: lbl, icon, cls = "ข่าวร้าย (Negative)", "🔻", "nc-neg"
            else: lbl, icon, cls = "ทั่วไป (Neutral)", "⚖️", "nc-neu"
            
            t_th, s_th = t, s
            if translator:
                try: t_th = translator.translate(t); s_th = translator.translate(s) if s else ""
                except: pass
            
            news_list.append({'title': t_th, 'summary': s_th, 'link': l, 'icon': icon, 'class': cls, 'label': lbl, 'score': sc, 'source': 'Finnhub'})

    # 2. Google News
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
                s = re.sub(re.compile('<.*?>'), '', getattr(i, 'summary', '') or getattr(i, 'description', ''))[:300]
                sc = TextBlob(t).sentiment.polarity
                
                if sc > 0.05: lbl, icon, cls = "ข่าวดี (Positive)", "🚀", "nc-pos"
                elif sc < -0.05: lbl, icon, cls = "ข่าวร้าย (Negative)", "🔻", "nc-neg"
                else: lbl, icon, cls = "ทั่วไป (Neutral)", "⚖️", "nc-neu"
                
                t_th, s_th = t, s
                if translator:
                    try: t_th = translator.translate(t); s_th = translator.translate(s) if s else ""
                    except: pass
                
                news_list.append({'title': t_th, 'summary': s_th, 'link': i.link, 'icon': icon, 'class': cls, 'label': lbl, 'score': sc, 'source': 'Google'})
        except: pass
    return news_list[:10]

def calculate_technical_setup(df):
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        
        close = df['Close'].iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        atr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        rsi_val = rsi_series.iloc[-1]

        if close > ema50 and ema50 > ema200: trend, sig, col, sc = "UPTREND (ขาขึ้น)", "BUY", "#00E676", 2
        elif close < ema50 and ema50 < ema200: trend, sig, col, sc = "DOWNTREND (ขาลง)", "SELL", "#FF1744", -2
        else: trend, sig, col, sc = "SIDEWAYS (ออกข้าง)", "WAIT", "#FFD600", 0
        
        return {'trend': trend, 'signal': sig, 'color': col, 'rsi_series': rsi_series, 'rsi_val': rsi_val, 'entry': close, 'sl': close-(1.5*atr) if sc>=0 else close+(1.5*atr), 'tp': close+(2.5*atr) if sc>=0 else close-(2.5*atr)}
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
        msg, col, icon = "Bullish Strong (แกร่งมาก)", "#00E676", "🐂" if price > e20 else ("Bullish Retrace (ย่อตัว)", "#00E676", "📉")
    else:
        msg, col, icon = "Bearish Strong (ลงหนัก)", "#FF1744", "🐻" if price < e20 else ("Bearish Correction (ดีดตัว)", "#FF1744", "📈")
    
    all_lvls = {**pivots, **{k:v for k,v in dynamics.items() if k!='Current'}}
    n_name, n_price, min_d = "", 0, float('inf')
    for k,v in all_lvls.items():
        if abs(price-v) < min_d: min_d, n_name, n_price = abs(price-v), k, v
    
    dist_pct = (min_d / price) * 100
    act = f"⚠️ ราคากำลังทดสอบแนว **{n_name}** ({n_price:,.2f}) ระยะห่างเพียง {dist_pct:.2f}%" if dist_pct < 0.8 else f"มีพื้นที่วิ่ง (Room to run) ไปหา **{n_name}** ({n_price:,.2f})"
    return msg, col, icon, act

def calculate_bitkub_ai_levels(h, l, c):
    pp = (h+l+c)/3
    rng = h-l
    mid = (h+l)/2
    st, col = ("BULLISH (กระทิง)", "#00E676") if c > mid else ("BEARISH (หมี)", "#FF1744")
    
    if c > pp: insight = f"ราคายืนเหนือ Pivot ({pp:,.0f}) ได้ ลุ้นทดสอบต้านถัดไปที่ R1"
    else: insight = f"ราคาหลุดต่ำกว่า Pivot ({pp:,.0f}) ระวังลงไปทดสอบ S1"
    
    return {
        "levels": [
            {"name":"🚀 R2","price":pp+rng,"type":"res"}, {"name":"🛑 R1","price":(2*pp)-l,"type":"res"},
            {"name":"⚖️ PIVOT","price":pp,"type":"neu"},
            {"name":"🛡️ S1","price":(2*pp)-h,"type":"sup"}, {"name":"💎 S2","price":pp-rng,"type":"sup"}
        ],
        "fib": {"top": l+(rng*0.618), "bot": l+(rng*0.382)}, 
        "status": st, "color": col, "insight": insight
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
    t_txt, n_txt = "", ""
    
    if setup['trend'] == "UPTREND (ขาขึ้น)": score += 20; t_txt = "กราฟเป็นขาขึ้นชัดเจน ยืนเหนือ EMA"
    elif setup['trend'] == "DOWNTREND (ขาลง)": score -= 20; t_txt = "กราฟเป็นขาลง หลุดแนวรับสำคัญ"
    else: t_txt = "กราฟออกข้าง รอเลือกทาง"
    
    if setup['rsi_val'] > 70: score -= 5; t_txt += " (Overbought ระวังย่อ)"
    elif setup['rsi_val'] < 30: score += 5; t_txt += " (Oversold ลุ้นเด้ง)"
    
    n_score = sum([n['score'] for n in news]) if news else 0
    if n_score > 0.3: score += 15; n_txt = "ข่าวสารเชิงบวก สนับสนุนราคา"
    elif n_score < -0.3: score -= 15; n_txt = "ข่าวสารเชิงลบ กดดันตลาด"
    else: n_txt = "ข่าวสารทรงตัว ไม่มีประเด็นใหญ่"
    
    score = max(0, min(100, score))
    verd = "STRONG BUY" if score>=80 else "BUY" if score>=60 else "SELL" if score<=40 else "STRONG SELL" if score<=20 else "HOLD"
    return t_txt, n_txt, score, verd

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    if c1.button("BTC"): set_symbol("BTC-USD")
    if c2.button("ETH"): set_symbol("ETH-USD")
    
    st.markdown("---")
    st.markdown("### 🇹🇭 Bitkub Rate")
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
    if st.button("วิเคราะห์ ⚡", use_container_width=True): 
        set_symbol(sym_input); st.rerun()

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
        
        if ai_sc >= 70: sc_col, sc_glow = "#00E676", "0, 230, 118"
        elif ai_sc <= 30: sc_col, sc_glow = "#FF1744", "255, 23, 68"
        else: sc_col, sc_glow = "#FFD600", "255, 214, 0"

        # Hero
        st.markdown(f"""
        <div class="glass-card" style="border-top:5px solid {color};text-align:center;">
            <div style="font-size:3.5rem;font-weight:900;line-height:1;margin-bottom:10px;">{symbol}</div>
            <div style="font-size:3rem;color:{color};font-weight:bold;">{curr:,.2f}</div>
            <div style="background:rgba({sc_glow}, 0.2);padding:5px 20px;border-radius:20px;display:inline-block;margin-top:10px;">
                <span style="color:{color};font-weight:bold;font-size:1.1rem;">{chg:+.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R Dynamic", "🧠 AI Guru", "🇹🇭 Bitkub AI"])

        # 1. Chart
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.05)
            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['Open'], high=ha['High'], low=ha['HA_Low'], close=ha['Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF', width=2), name="EMA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=setup['rsi_series'], line=dict(color='#E040FB', width=2), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            fig.update_layout(template='plotly_dark', height=550, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # 2. Stats
        with tabs[1]:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-box'><div class='metric-label'>High</div><div class='metric-val' style='color:#00E676'>{df['High'].max():,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><div class='metric-label'>Low</div><div class='metric-val' style='color:#FF1744'>{df['Low'].min():,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box'><div class='metric-label'>Vol</div><div class='metric-val' style='color:#E040FB'>{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>", unsafe_allow_html=True)
            
            if info:
                st.markdown("---")
                summary = info.get('longBusinessSummary', 'No description available.')
                if HAS_TRANSLATOR:
                    try: 
                        translator = GoogleTranslator(source='auto', target='th')
                        summary = translator.translate(summary[:2000])
                    except: pass
                
                with st.expander(f"🏢 เกี่ยวกับ {symbol} (คลิกเพื่ออ่าน)"):
                    st.write(summary)
                
                sector = info.get('sector', 'Unknown')
                pe = info.get('trailingPE')
                
                st.markdown(f"**Sector:** {sector}")
                
                c1, c2 = st.columns(2)
                with c1:
                    c1.markdown(f"<div class='metric-box'><div class='metric-label'>P/E Ratio</div><div class='metric-val'>{pe if pe else 'N/A'}</div></div>", unsafe_allow_html=True)
                
                with c2:
                    if pe:
                        avg_pe = get_sector_pe_benchmark(sector)
                        diff = ((pe - avg_pe) / avg_pe) * 100
                        status = "แพงกว่ากลุ่ม" if diff > 0 else "ถูกกว่ากลุ่ม"
                        color = "#FF1744" if diff > 0 else "#00E676"
                        st.markdown(f"<div class='metric-box' style='border-left-color:{color}'><div class='metric-label'>เทียบกลุ่ม ({avg_pe})</div><div class='metric-val' style='color:{color}; font-size:1.4rem'>{status} ({abs(diff):.1f}%)</div></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='metric-box'><div class='metric-label'>เทียบกลุ่ม</div><div class='metric-val' style='font-size:1.4rem'>N/A</div></div>", unsafe_allow_html=True)

        # 3. AI News
        with tabs[2]:
            st.markdown("### 📰 Market Sentiment")
            if news:
                for n in news:
                    st.markdown(f"""
                    <div class="news-card {n['class']}">
                        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                            <div style="display:flex;align-items:center;gap:10px;">
                                <span style="font-size:1rem;">{n['icon']}</span>
                                <span style="font-weight:bold;color:#fff;">{n['label']}</span>
                            </div>
                            <span style="font-size:0.8rem;background:#333;padding:2px 8px;border-radius:5px;">{n['source']}</span>
                        </div>
                        <h4 style="margin:10px 0;color:#e0e0e0;">{n['title']}</h4>
                        <p style="color:#aaa;font-size:0.9rem;line-height:1.5;">{n['summary']}</p>
                        <div style="text-align:right;margin-top:10px;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;text-decoration:none;">🔗 อ่านต่อ</a></div>
                    </div>
                    """, unsafe_allow_html=True)
            else: st.info("ไม่พบข่าว หรือ API ถูกจำกัด")

        # 4. Setup
        with tabs[3]:
            if setup:
                st.markdown(f"""
                <div class='ai-insight-box' style='border-left: 5px solid {setup['color']}; margin-bottom:20px;'>
                    <h2 style='margin:0; color:{setup['color']};'>{setup['signal']}</h2>
                    <p style='font-size:1.2rem; color:#ccc; margin-top:5px;'>{setup['trend']}</p>
                    <div style='margin-top:15px; display:flex; gap:10px;'>
                        <span style='background:#111; padding:5px 15px; border-radius:10px; border:1px solid #333;'>RSI: {setup['rsi_val']:.1f}</span>
                        <span style='background:#111; padding:5px 15px; border-radius:10px; border:1px solid #333;'>Entry: {setup['entry']:,.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-box' style='border-left-color:#00E5FF'><div class='metric-label'>Buy Zone</div><div class='metric-val'>{curr*0.99:,.2f}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-box' style='border-left-color:#00E676'><div class='metric-label'>Target (TP)</div><div class='metric-val'>{setup['tp']:,.2f}</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-box' style='border-left-color:#FF1744'><div class='metric-label'>Stop Loss</div><div class='metric-val'>{setup['sl']:,.2f}</div></div>", unsafe_allow_html=True)

        # 5. Verdict
        with tabs[4]:
            col_v1, col_v2 = st.columns([1, 1.5])
            with col_v1:
                st.markdown(f"""
                <div class="verdict-ring" style="border-color:{sc_col}; color:{sc_col}; box-shadow:0 0 30px rgba({sc_glow}, 0.5);">
                    {ai_sc}
                </div>
                <div style="text-align:center; font-size:2rem; font-weight:900; color:{sc_col}; text-transform:uppercase; letter-spacing:2px;">
                    {ai_vd}
                </div>
                """, unsafe_allow_html=True)
            with col_v2:
                st.markdown("### 🔍 AI Analysis Breakdown")
                st.markdown(f"""
                <div class="factor-card" style="border-left-color:{sc_col};">
                    <h4 style="margin:0;color:#fff;">📈 Technical Insight</h4>
                    <p style="margin-top:5px;color:#ccc;">{t_txt}</p>
                </div>
                <div class="factor-card" style="border-left-color:{'#00E676' if 'บวก' in n_txt else '#FF1744'};">
                    <h4 style="margin:0;color:#fff;">📰 News Sentiment</h4>
                    <p style="margin-top:5px;color:#ccc;">{n_txt}</p>
                </div>
                """, unsafe_allow_html=True)

        # 6. S/R Dynamic
        with tabs[5]:
            pivots = calculate_pivot_points(df)
            dynamic = calculate_dynamic_levels(df)
            
            if pivots and dynamic:
                msg, col, icon, act = generate_dynamic_insight(curr, pivots, dynamic)
                st.markdown(f"""
                <div class='ai-insight-box' style='border-color:{col}; box-shadow:0 0 15px {col}40; margin-bottom:25px;'>
                    <div class='ai-insight-icon'>{icon}</div>
                    <h3 style='margin:0; color:{col};'>{msg}</h3>
                    <p style='font-size:1.1rem; color:#ccc; margin-top:5px;'>{act}</p>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🧱 Pivots")
                    for k, v in pivots.items():
                        cls = "sr-res" if "R" in k else "sr-sup" if "S" in k else "sr-piv"
                        st.markdown(f"<div class='sr-card {cls}'><b>{k}</b><span>{v:,.2f}</span></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("#### 🌊 Dynamic")
                    for k, v in dynamic.items():
                        if k!="Current":
                            dist = ((curr-v)/v)*100
                            cl = "#00E676" if curr > v else "#FF1744"
                            st.markdown(f"<div class='sr-card' style='border-left:4px solid {cl}; background:rgba({255 if cl=='#FF1744' else 0}, {230 if cl=='#00E676' else 23}, {118 if cl=='#00E676' else 68}, 0.1);'><span>{k}</span><div style='text-align:right;'>{v:,.2f}<br><small style='color:{cl}'>{dist:+.2f}%</small></div></div>", unsafe_allow_html=True)

        # 7. AI Guru (New)
        with tabs[6]:
            st.markdown("### 🧠 AI Guru: Fundamental & Valuation")
            if info:
                guru = analyze_stock_guru(info, setup, symbol)
                st.markdown(f"""
                <div class='ai-insight-box' style='border:2px solid {guru['color']}; text-align:center; margin-bottom:20px;'>
                    <h1 style='color:{guru['color']}; font-size:3rem; margin:0;'>{guru['verdict']}</h1>
                    
                    <!-- NEW: Valuation Score Bar -->
                    <div style="margin:20px 0; background:#333; border-radius:10px; height:10px; width:100%;">
                        <div style="width:{guru['val_score']*10}%; background:{guru['color']}; height:100%; border-radius:10px;"></div>
                    </div>
                    <p style='font-size:1.1rem; color:#ccc;'>Valuation Score: {guru['val_score']}/10</p>
                </div>
                
                <div class='ai-article'>
                    <h4 style='margin-top:0; color:#fff;'>📝 บทวิเคราะห์โดย AI (AI Analyst Report)</h4>
                    {guru['article']}
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🏢 Quality Score (พื้นฐาน)")
                    for r in guru['reasons_q']:
                        st.markdown(f"<div class='guru-card' style='border-left:4px solid {'#00E676' if '✅' in r else '#FF1744'};'>{r}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("#### ⚖️ Valuation Score (ความคุ้มค่า)")
                    for r in guru['reasons_v']:
                        st.markdown(f"<div class='guru-card' style='border-left:4px solid {'#00E676' if '✅' in r else '#FF1744'};'>{r}</div>", unsafe_allow_html=True)
            else:
                st.info("⚠️ ไม่สามารถวิเคราะห์ได้ (ไม่มีข้อมูลพื้นฐาน/งบการเงิน) สำหรับสินทรัพย์นี้")

        # 8. Bitkub AI
        with tabs[7]:
            bk_sel = st.radio("เลือกเหรียญ (THB)", ["BTC", "ETH"], horizontal=True)
            if bk_data:
                pair = f"THB_{bk_sel}"
                d = bk_data.get(pair, {})
                if d:
                    last, h24, l24 = d.get('last',0), d.get('high24hr',0), d.get('low24hr',0)
                    ai_bk = calculate_bitkub_ai_levels(h24, l24, last)
                    
                    st.markdown(f"""
                    <div class='ai-insight-box' style='text-align:center; border:2px solid {ai_bk['color']};'>
                        <div style='font-size:3rem; font-weight:900; color:#fff;'>{last:,.0f} <span style='font-size:1.5rem;'>THB</span></div>
                        <div style='font-size:1.5rem; font-weight:bold; color:{ai_bk['color']}; text-transform:uppercase;'>{ai_bk['status']}</div>
                        <p style='margin-top:10px; color:#ccc
