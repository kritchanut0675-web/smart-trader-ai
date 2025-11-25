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

# --- 2. Ultra Black CSS (Enhanced S/R & News) ---
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
        
        /* --- 🛡️ BIG S/R TABLE STYLES --- */
        .sr-container {
            display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;
        }
        .sr-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 15px 25px; border-radius: 15px; font-size: 1.5rem; font-weight: bold;
        }
        .res-row { background: linear-gradient(90deg, rgba(255, 23, 68, 0.1), rgba(0,0,0,0)); border-left: 8px solid #FF1744; color: #FF1744; }
        .sup-row { background: linear-gradient(90deg, rgba(0, 230, 118, 0.1), rgba(0,0,0,0)); border-left: 8px solid #00E676; color: #00E676; }
        .curr-row { background: #222; border: 1px solid #555; color: #fff; justify-content: center; font-size: 1.8rem; text-shadow: 0 0 10px white; }
        .sr-label { font-size: 1rem; opacity: 0.7; letter-spacing: 2px; text-transform: uppercase; }

        /* --- 📰 NEWS CARD STYLES --- */
        .news-card {
            padding: 20px; margin-bottom: 15px; background: #111; border-radius: 15px;
            border-left: 6px solid #888; transition: transform 0.2s;
        }
        .news-card:hover { transform: scale(1.01); background: #161616; }
        .nc-pos { border-left-color: #00E676; box-shadow: -5px 0 15px -5px rgba(0, 230, 118, 0.2); }
        .nc-neg { border-left-color: #FF1744; box-shadow: -5px 0 15px -5px rgba(255, 23, 68, 0.2); }
        .nc-neu { border-left-color: #FFD600; }
        .news-sentiment { font-size: 0.9rem; font-weight: bold; margin-bottom: 5px; display: inline-block; padding: 2px 8px; border-radius: 5px; color:#000; }
        .ns-pos { background: #00E676; }
        .ns-neg { background: #FF1744; color: #fff !important; }
        .ns-neu { background: #FFD600; }

        /* AI VERDICT CARD */
        .ai-card {
            background: linear-gradient(145deg, #111, #0d0d0d);
            border: 2px solid #00E5FF;
            border-radius: 20px; padding: 30px; position: relative;
            box-shadow: 0 0 30px rgba(0, 229, 255, 0.1);
        }
        .ai-score-circle {
            width: 100px; height: 100px; border-radius: 50%;
            border: 5px solid #00E5FF; display: flex; align-items: center; justify-content: center;
            font-size: 2.5rem; font-weight: bold; color: #00E5FF; margin: 0 auto 20px auto;
        }

        /* BUTTONS */
        div.stButton > button {
            font-size: 1.1rem !important; padding: 15px !important; border-radius: 15px !important;
            background: #111; border: 1px solid #333; color: #fff;
        }
        div.stButton > button:hover { background: #00E5FF; color: #000 !important; font-weight: bold; }
        
        /* TABS */
        button[data-baseweb="tab"] { font-size: 1.1rem !important; font-weight: 600 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. Data & Analysis Functions ---

@st.cache_data(ttl=300)
def get_data_full(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df, ticker
    except: return pd.DataFrame(), None

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

# --- NEW: Improved S/R Level Identification ---
def identify_sr_levels(df):
    levels = []
    try:
        # Simple local min/max logic
        window = 5
        for i in range(window, len(df) - window):
            if df['Low'][i] == df['Low'][i-window:i+window+1].min():
                levels.append({'price': df['Low'][i], 'type': 'Support'})
            elif df['High'][i] == df['High'][i-window:i+window+1].max():
                levels.append({'price': df['High'][i], 'type': 'Resistance'})
        
        # Sort and filter close levels
        levels.sort(key=lambda x: x['price'])
        filtered = []
        if levels:
            curr = levels[0]
            for next_lvl in levels[1:]:
                if (next_lvl['price'] - curr['price']) / curr['price'] > 0.02: # 2% difference
                    filtered.append(curr)
                    curr = next_lvl
            filtered.append(curr)
            
        return filtered
    except: return []

# --- NEW: AI News Analysis (Restored & Improved) ---
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
            # Sentiment Analysis
            blob = TextBlob(item.title)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.05:
                sentiment = "ข่าวดี (Positive)"
                color_class = "nc-pos"
                tag_class = "ns-pos"
                icon = "🚀"
            elif sentiment_score < -0.05:
                sentiment = "ข่าวร้าย (Negative)"
                color_class = "nc-neg"
                tag_class = "ns-neg"
                icon = "🔻"
            else:
                sentiment = "ทั่วไป (Neutral)"
                color_class = "nc-neu"
                tag_class = "ns-neu"
                icon = "⚖️"

            # Translate
            title_th = item.title
            if translator:
                try: title_th = translator.translate(item.title)
                except: pass

            news_list.append({
                'title_th': title_th,
                'link': item.link,
                'sentiment': sentiment,
                'class': color_class,
                'tag': tag_class,
                'icon': icon,
                'score': sentiment_score
            })
    except: pass
    return news_list

# --- Guru & Verdict Logic ---
def get_guru_opinion(ticker, current_price):
    try:
        info = ticker.info
        if 'targetMeanPrice' not in info: return None
        return {
            'target_mean': info.get('targetMeanPrice'),
            'target_high': info.get('targetHighPrice'),
            'target_low': info.get('targetLowPrice'),
            'rec': info.get('recommendationKey', 'none').upper(),
            'count': info.get('numberOfAnalystOpinions', 0)
        }
    except: return None

def generate_ai_analysis(df, setup, guru_data, news_list):
    analysis_text = ""
    score = 50
    
    # Technical
    if setup['trend'] == "Uptrend (ขาขึ้น)":
        analysis_text += "📈 **ด้านเทคนิค:** กราฟยังคงรักษาทรงขาขึ้นได้อย่างแข็งแกร่ง ราคายืนเหนือเส้น EMA "
        score += 20
    elif setup['trend'] == "Downtrend (ขาลง)":
        analysis_text += "📉 **ด้านเทคนิค:** กราฟอยู่ในแนวโน้มขาลงชัดเจน ระวังแรงขายกดดัน "
        score -= 20
    else:
        analysis_text += "⚖️ **ด้านเทคนิค:** กราฟแกว่งตัวออกข้าง (Sideways) รอเลือกทาง "

    if setup['rsi'] > 70: score -= 5
    elif setup['rsi'] < 30: score += 5
    
    # News Impact
    news_score = sum([n['score'] for n in news_list]) if news_list else 0
    if news_score > 0.2:
        analysis_text += "\n\n📰 **กระแสข่าว:** ภาพรวมข่าวสารเป็นบวก สนับสนุนราคา"
        score += 10
    elif news_score < -0.2:
        analysis_text += "\n\n📰 **กระแสข่าว:** มีข่าวเชิงลบกดดันตลาด ควรระวัง"
        score -= 10

    score = max(0, min(100, score))
    if score >= 75: verdict = "STRONG BUY"
    elif score >= 55: verdict = "BUY"
    elif score >= 45: verdict = "HOLD / WATCH"
    elif score >= 25: verdict = "SELL"
    else: verdict = "STRONG SELL"
    
    return analysis_text, score, verdict

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00E5FF;'>💎 ULTRA 7</h1>", unsafe_allow_html=True)
    st.caption("AI & Guru Edition")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    if c1.button("BTC-USD"): set_symbol("BTC-USD")
    if c2.button("ETH-USD"): set_symbol("ETH-USD")
    c3, c4 = st.columns(2)
    if c3.button("Gold"): set_symbol("GC=F")
    if c4.button("Oil"): set_symbol("CL=F")
    
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
        df, ticker = get_data_full(symbol, period, "1d")
        
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล '{symbol}'")
    else:
        # Process Data
        curr_price = df['Close'].iloc[-1]
        change = curr_price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        
        setup = calculate_technical_setup(df)
        guru_data = get_guru_opinion(ticker, curr_price)
        news_list = get_ai_analyzed_news_thai(symbol)
        sr_levels = identify_sr_levels(df)
        
        ai_text, ai_score, ai_verdict = generate_ai_analysis(df, setup, guru_data, news_list)

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

        # --- 7 TABS ---
        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "🎯 S/R & Setup", "💰 Entry", "🗣️ Guru View", "🤖 AI Verdict"])

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

        # Tab 2: Stats
        with tabs[1]:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class="stat-box"><div class="stat-label">High</div><div class="stat-value" style="color:#00E676;">{df['High'].max():,.2f}</div></div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="stat-box"><div class="stat-label">Low</div><div class="stat-value" style="color:#FF1744;">{df['Low'].min():,.2f}</div></div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="stat-box"><div class="stat-label">Vol</div><div class="stat-value" style="color:#E040FB;">{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>""", unsafe_allow_html=True)

        # Tab 3: AI News (Restored!)
        with tabs[2]:
            st.markdown("### 📰 AI Sentiment Analysis (วิเคราะห์อารมณ์ข่าว)")
            if news_list:
                for n in news_list:
                    st.markdown(f"""
                    <div class="news-card {n['class']}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="news-sentiment {n['tag']}">{n['icon']} {n['sentiment']}</span>
                        </div>
                        <h4 style="color:#fff; margin:10px 0;">{n['title_th']}</h4>
                        <a href="{n['link']}" target="_blank" style="color:#aaa; font-size:0.9rem;">🔗 อ่านข่าวต้นฉบับ</a>
                    </div>
                    """, unsafe_allow_html=True)
            else: st.info("ไม่พบข่าวในขณะนี้ หรือ API ถูกจำกัดการเข้าถึง")

        # Tab 4: S/R & Setup (Enhanced!)
        with tabs[3]:
            # Big S/R Table
            st.markdown("### 🛡️ Key Levels (แนวรับ-แนวต้าน)")
            res_list = sorted([l['price'] for l in sr_levels if l['price'] > curr_price])[:3]
            sup_list = sorted([l['price'] for l in sr_levels if l['price'] < curr_price], reverse=True)[:3]
            
            sr_html = "<div class='sr-container'>"
            for r in reversed(res_list):
                sr_html += f"<div class='sr-row res-row'><div>RESISTANCE</div><div>{r:,.2f}</div></div>"
            sr_html += f"<div class='sr-row curr-row'><div>CURRENT: {curr_price:,.2f}</div></div>"
            for s in sup_list:
                sr_html += f"<div class='sr-row sup-row'><div>SUPPORT</div><div>{s:,.2f}</div></div>"
            sr_html += "</div>"
            st.markdown(sr_html, unsafe_allow_html=True)
            
            # Technical Setup Box
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

        # Tab 6: Guru
        with tabs[5]:
            st.markdown("### 🗣️ Guru Opinions")
            if guru_data and guru_data['target_mean']:
                st.markdown(f"""<div class="glass-card" style="text-align:center;"><div style="font-size:3rem; font-weight:bold;">{guru_data['rec']}</div><div>Target: {guru_data['target_mean']:,.2f}</div></div>""", unsafe_allow_html=True)
            else: st.warning("No Analyst Data for this asset")

        # Tab 7: AI Verdict
        with tabs[6]:
            st.markdown("### 🤖 AI Market Analysis")
            if ai_score >= 70: score_color = "#00E676"
            elif ai_score <= 30: score_color = "#FF1744"
            else: score_color = "#FFD600"
            st.markdown(f"""<div class="ai-card" style="text-align:center; border-color:{score_color};"><div class="ai-score-circle" style="border-color:{score_color}; color:{score_color};">{ai_score}</div><div style="font-size:2rem; font-weight:bold; color:{score_color};">{ai_verdict}</div><p>{ai_text}</p></div>""", unsafe_allow_html=True)
