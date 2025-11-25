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
# Import translation library (Handling case if not installed)
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
    page_title="Smart Trader AI : Ultra",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# --- 2. Ultra Premium CSS (Bigger & Better) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
        
        /* Global Settings */
        html, body, [class*="css"] {
            font-family: 'Kanit', sans-serif;
        }
        
        /* Main Background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #fff;
        }

        /* --- 🔍 HUGE Search Input --- */
        div[data-testid="stTextInput"] input { 
            background-color: #ffffff !important; 
            color: #000000 !important; 
            font-weight: 700 !important;
            font-size: 1.5rem !important; /* ใหญ่ขึ้น */
            height: 60px !important;
            border: 3px solid #00E5FF !important;
            border-radius: 15px !important;
            padding: 10px 20px !important;
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
        }
        div[data-testid="stTextInput"] label {
            color: #00E5FF !important;
            font-size: 1.3rem !important;
            font-weight: bold;
        }

        /* --- 💎 Glass Cards (Bigger Padding) --- */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border-radius: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 35px; /* เพิ่ม Padding */
            margin-bottom: 30px;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
            transition: transform 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            border-color: rgba(255,255,255,0.3);
        }

        /* --- 📊 Stats Box (Larger) --- */
        .stat-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            border: 2px solid rgba(255,255,255,0.05);
            margin-bottom: 15px;
        }
        .stat-label { 
            color: #aaa; 
            font-size: 1.1rem; /* ใหญ่ขึ้น */ 
            text-transform: uppercase; 
            letter-spacing: 1.5px; 
            margin-bottom: 5px;
        }
        .stat-value { 
            font-size: 2.5rem; /* ใหญ่สะใจ */ 
            font-weight: 800; 
            color: #fff; 
            text-shadow: 0 0 10px rgba(255,255,255,0.3);
        }

        /* --- 📰 News Cards (Thai) --- */
        .news-card {
            padding: 25px;
            margin-bottom: 20px;
            background: rgba(20, 20, 20, 0.6);
            border-radius: 18px;
            border-left: 8px solid #888;
            transition: all 0.3s;
        }
        .news-card:hover { transform: scale(1.02); }
        .nc-pos { border-left-color: #00E676; background: linear-gradient(90deg, rgba(0,230,118,0.1), transparent); }
        .nc-neg { border-left-color: #FF1744; background: linear-gradient(90deg, rgba(255,23,68,0.1), transparent); }
        .nc-neu { border-left-color: #FFD600; background: linear-gradient(90deg, rgba(255,214,0,0.1), transparent); }
        
        .news-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 8px; color: #fff; }
        .news-meta { font-size: 1rem; color: #ccc; }

        /* --- 💰 Entry Strategy Box --- */
        .entry-box {
            background: #1a1a1a;
            border-radius: 20px;
            padding: 30px;
            border-left: 8px solid #555;
            margin-bottom: 20px;
            box-shadow: 5px 5px 20px rgba(0,0,0,0.5);
        }
        .eb-title { font-size: 1.5rem; font-weight: bold; }
        .eb-price { font-size: 2.2rem; font-weight: 800; float: right; }
        .eb-desc { font-size: 1.1rem; color: #aaa; margin-top: 5px; }

        .eb-1 { border-left-color: #00E5FF; }
        .eb-2 { border-left-color: #FFD600; }
        .eb-3 { border-left-color: #FF1744; }

        /* Sidebar Buttons */
        div.stButton > button {
            font-size: 1.1rem !important;
            padding: 15px !important;
            border-radius: 15px !important;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #00E5FF, #2979FF);
            color: #fff !important;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. Data Functions ---

@st.cache_data(ttl=300)
def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    except: return pd.DataFrame()

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

def calculate_trade_setup(df):
    try:
        close = df['Close'].iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        if close > ema50 and ema50 > ema200:
            trend = "Uptrend (ขาขึ้น) 🟢"
            signal = "BUY / LONG"
            color = "#00E676"
            sl = close - (1.5 * atr)
            tp = close + (2.5 * atr)
        elif close < ema50 and ema50 < ema200:
            trend = "Downtrend (ขาลง) 🔴"
            signal = "SELL / SHORT"
            color = "#FF1744"
            sl = close + (1.5 * atr)
            tp = close - (2.5 * atr)
        else:
            trend = "Sideways (ไซด์เวย์) 🟡"
            signal = "WAIT (รอ)"
            color = "#888"
            sl = close - atr
            tp = close + atr
            
        return {'trend': trend, 'signal': signal, 'color': color, 'entry': close, 'sl': sl, 'tp': tp, 'atr': atr}
    except: return None

# --- AI News Analysis Function (With Translation) ---
@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    news_list = []
    clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
    
    # Initialize Translator
    translator = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None

    try:
        # Fetch News
        q = urllib.parse.quote(f"site:bloomberg.com {clean_sym} market")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        if len(feed.entries) == 0:
            q = urllib.parse.quote(f"{clean_sym} finance news")
            rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)

        # Analyze & Translate (Limit to 4 items for speed)
        for item in feed.entries[:4]:
            # 1. Analyze Sentiment (Use original English text for accuracy)
            blob = TextBlob(item.title)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.1:
                sentiment = "ข่าวดี (Positive)"
                color_class = "nc-pos"
                icon = "🚀"
            elif sentiment_score < -0.1:
                sentiment = "ข่าวลบ (Negative)"
                color_class = "nc-neg"
                icon = "🔻"
            else:
                sentiment = "ทั่วไป (Neutral)"
                color_class = "nc-neu"
                icon = "⚖️"

            # 2. Translate Title to Thai
            title_th = item.title
            if translator:
                try:
                    title_th = translator.translate(item.title)
                except:
                    pass # Fallback to English if translation fails

            news_list.append({
                'title_th': title_th,
                'link': item.link,
                'sentiment': sentiment,
                'class': color_class,
                'icon': icon,
                'score': sentiment_score
            })
    except: pass
    return news_list

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00E5FF;'>💎 ULTRA AI</h1>", unsafe_allow_html=True)
    st.caption("Ultimate Edition by KRITCHANUT")
    st.markdown("---")
    
    st.markdown("### 🌎 Global Assets")
    c1, c2 = st.columns(2)
    if c1.button("BTC-USD"): set_symbol("BTC-USD")
    if c2.button("ETH-USD"): set_symbol("ETH-USD")
    
    st.markdown("### 🌟 Popular")
    c3, c4 = st.columns(2)
    if c3.button("Gold (XAU)"): set_symbol("GC=F")
    if c4.button("Oil (WTI)"): set_symbol("CL=F")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    chart_type = st.selectbox("Chart Style", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Period", options=["1mo", "3mo", "6mo", "1y", "5y"], value="1y")

# --- 5. Main Content ---

# Header Section
st.markdown("<h2 style='margin-bottom: 10px; color:#00E5FF;'>🔍 ค้นหาเหรียญ / หุ้น (Premium Search)</h2>", unsafe_allow_html=True)

c_search, c_btn = st.columns([4, 1])
with c_search:
    sym_input = st.text_input("ระบุชื่อย่อ (เช่น BTC-USD, AAPL)", value=st.session_state.symbol, label_visibility="collapsed")
with c_btn:
    st.write("") # Spacer
    if st.button("วิเคราะห์ ⚡", use_container_width=True):
        st.session_state.symbol = sym_input
        st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner('💎 AI กำลังประมวลผลข้อมูลขนาดใหญ่...'):
        df = get_data(symbol, period, "1d")
        
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล '{symbol}' กรุณาลองตรวจสอบชื่อย่อใหม่")
    else:
        # Calculation
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = curr_price - prev_price
        pct = (change / prev_price) * 100
        last_date = df.index[-1].strftime('%d %B %Y')
        
        setup = calculate_trade_setup(df)

        # --- HERO SECTION (Ultra Big) ---
        color_trend = "#00E676" if change >= 0 else "#FF1744"
        arrow = "▲" if change >= 0 else "▼"
        
        st.markdown(f"""
        <div class="glass-card" style="border-top: 6px solid {color_trend}; text-align: center; padding-top:40px;">
            <div style="font-size: 1.5rem; color: #aaa; letter-spacing: 3px; text-transform: uppercase;">ASSET ANALYSIS</div>
            <div style="font-size: 5rem; font-weight: 900; margin: 15px 0; background: -webkit-linear-gradient(45deg, #fff, {color_trend}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(255,255,255,0.2);">
                {symbol}
            </div>
            <div style="font-size: 4.5rem; font-weight: 800; color: {color_trend}; text-shadow: 0 0 20px {color_trend}40;">
                {curr_price:,.2f} 
            </div>
            <div style="color: #ccc; font-size:1.2rem; margin-bottom: 20px;">Price Date: {last_date}</div>
            <div style="background: {color_trend}20; padding: 10px 30px; border-radius: 30px; display: inline-block; border: 2px solid {color_trend};">
                <span style="font-size: 1.8rem; font-weight:bold; color:{color_trend};">
                    {arrow} {abs(change):,.2f} ({pct:+.2f}%)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- TABS (Larger Text) ---
        tabs = st.tabs(["📈 Chart", "📊 Pro Stats", "📰 AI News (TH)", "🎯 Setup", "💰 Entry"])

        # 1. CHART
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['HA_Open'], high=ha['HA_High'], low=ha['HA_Low'], close=ha['HA_Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF', width=2), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=200).mean(), line=dict(color='#FF9100', width=2), name='EMA 200'), row=1, col=1)
            
            rsi = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0,0).rolling(14).mean() / abs(df['Close'].diff().where(lambda x: x<0,0)).rolling(14).mean())))
            fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#E040FB', width=2), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            
            fig.update_layout(height=700, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)

        # 2. PRO STATS (Grid System)
        with tabs[1]:
            st.markdown("### 📊 Market Statistics (สถิติตลาด)")
            high_p = df['High'].max()
            low_p = df['Low'].min()
            volatility = df['Close'].std()
            range_pos = ((curr_price - low_p) / (high_p - low_p)) * 100 if (high_p - low_p) != 0 else 50
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="stat-box"><div class="stat-label">Highest (สูงสุด)</div><div class="stat-value" style="color:#00E676;">{high_p:,.2f}</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="stat-box"><div class="stat-label">Lowest (ต่ำสุด)</div><div class="stat-value" style="color:#FF1744;">{low_p:,.2f}</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="stat-box"><div class="stat-label">Volatility (ความผันผวน)</div><div class="stat-value" style="color:#E040FB;">{volatility:,.2f}</div></div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass-card" style="padding: 30px; margin-top:20px;">
                <div style="display:flex; justify-content:space-between; font-size:1.2rem; color:#aaa; font-weight:bold;">
                    <span>Low: {low_p:,.2f}</span>
                    <span>Range Position</span>
                    <span>High: {high_p:,.2f}</span>
                </div>
                <div style="background: #444; height: 15px; border-radius: 10px; margin-top: 15px; position: relative;">
                    <div style="height: 100%; border-radius: 10px; width: {range_pos}%; background: linear-gradient(90deg, #FF1744, #00E676);"></div>
                    <div style="position: absolute; top: -8px; left: {range_pos}%; width: 6px; height: 30px; background: white; border-radius: 3px; box-shadow: 0 0 15px white;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. AI NEWS ANALYST (THAI)
        with tabs[2]:
            st.markdown("### 📰 AI News Analyst (แปลไทย)")
            st.caption("AI วิเคราะห์หัวข้อข่าวและแปลเป็นภาษาไทยเพื่อคุณ")
            
            analyzed_news = get_ai_analyzed_news_thai(symbol)
            
            if analyzed_news:
                total_score = sum([n['score'] for n in analyzed_news])
                if total_score > 0.1: overall, ov_color = "Bullish (แนวโน้มดี) 🚀", "#00E676"
                elif total_score < -0.1: overall, ov_color = "Bearish (ระวังแรงขาย) 🔻", "#FF1744"
                else: overall, ov_color = "Neutral (ทรงตัว) ⚖️", "#FFD600"
                
                st.markdown(f"""<div style="text-align:center; padding:15px; border:2px solid {ov_color}; border-radius:15px; margin-bottom:30px; background:rgba(0,0,0,0.3);"><span style="color:#ccc; font-size:1.2rem;">ภาพรวมข่าววันนี้:</span> <span style="font-size:2rem; font-weight:bold; color:{ov_color}; margin-left:10px;">{overall}</span></div>""", unsafe_allow_html=True)

                for news in analyzed_news:
                    st.markdown(f"""
                    <div class="news-card {news['class']}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <span style="font-size:1rem; font-weight:bold; color:#fff;">{news['icon']} {news['sentiment']}</span>
                        </div>
                        <a href="{news['link']}" target="_blank" style="text-decoration:none;">
                            <div class="news-title">{news['title_th']}</div>
                        </a>
                        <div class="news-meta">แตะที่หัวข้อเพื่ออ่านข่าวต้นฉบับ (EN)</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ไม่พบข่าวสำคัญ หรือ API ข่าวถูกจำกัดการเข้าถึงในขณะนี้")

        # 4. SETUP
        with tabs[3]:
            if setup:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <div style="color: #aaa; font-size:1.2rem;">SIGNAL</div>
                        <div style="font-size: 3rem; font-weight: 800; color: {setup['color']}; text-shadow:0 0 15px {setup['color']};">{setup['signal']}</div>
                        <hr style="border-color: #333; margin:20px 0;">
                        <div style="color: #aaa; font-size:1.2rem;">TREND</div>
                        <div style="font-size: 1.5rem;">{setup['trend']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="display: flex; gap: 20px; height: 100%;">
                        <div class="glass-card" style="flex: 1; text-align: center; border: 2px solid #FF1744;">
                            <div style="color: #FF1744; font-weight: bold; font-size:1.5rem;">STOP LOSS</div>
                            <div style="font-size: 2.5rem; font-weight:bold;">{setup['sl']:,.2f}</div>
                        </div>
                        <div class="glass-card" style="flex: 1; text-align: center; border: 2px solid #00E676;">
                            <div style="color: #00E676; font-weight: bold; font-size:1.5rem;">TAKE PROFIT</div>
                            <div style="font-size: 2.5rem; font-weight:bold;">{setup['tp']:,.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # 5. ENTRY
        with tabs[4]:
            st.markdown("### 💰 Smart Money Management (แผนการเข้าซื้อ)")
            t1 = curr_price * 0.995 
            t2 = curr_price * 0.98  
            t3 = curr_price * 0.95  
            
            st.markdown(f"""
            <div class="entry-box eb-1">
                <div style="overflow:hidden;">
                    <span class="eb-title" style="color:#00E5FF;">🔹 ไม้ที่ 1 : Probe Buy (20%)</span>
                    <span class="eb-price">{t1:,.2f}</span>
                </div>
                <div class="eb-desc">เข้าซื้อเพื่อทดสอบตลาด หรือเกาะแนวโน้มระยะสั้น</div>
            </div>
            
            <div class="entry-box eb-2">
                <div style="overflow:hidden;">
                    <span class="eb-title" style="color:#FFD600;">🔸 ไม้ที่ 2 : Accumulate (30%)</span>
                    <span class="eb-price">{t2:,.2f}</span>
                </div>
                <div class="eb-desc">จุดเข้าซื้อเพิ่มเมื่อราคาย่อตัวลงมา (Dip Buying)</div>
            </div>
            
            <div class="entry-box eb-3">
                <div style="overflow:hidden;">
                    <span class="eb-title" style="color:#FF1744;">🔻 ไม้ที่ 3 : Sniper Zone (50%)</span>
                    <span class="eb-price">{t3:,.2f}</span>
                </div>
                <div class="eb-desc">จุดซื้อของถูกเมื่อเกิด Panic Sell หรือแนวรับสำคัญมาก</div>
            </div>
            """, unsafe_allow_html=True)
