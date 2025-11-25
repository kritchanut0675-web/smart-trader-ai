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

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- 1. Setup & Design ---
st.set_page_config(
    page_title="Smart Trader AI : Premium",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# --- 2. Premium CSS & UI Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
        
        /* Global Font */
        html, body, [class*="css"] {
            font-family: 'Kanit', sans-serif;
        }

        /* Background */
        .stApp {
            background: radial-gradient(circle at top center, #1a1a2e 0%, #000000 100%);
            color: #fff;
        }

        /* --- Input Box (Black Text) --- */
        div[data-testid="stTextInput"] input { 
            background-color: #ffffff !important; 
            color: #000000 !important; 
            font-weight: 600 !important;
            border: 2px solid #00E5FF !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }
        div[data-testid="stTextInput"] label {
            color: #00E5FF !important;
            font-size: 1.1rem !important;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* --- Stats Card Design --- */
        .stat-box {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s;
        }
        .stat-box:hover {
            transform: scale(1.02);
            border-color: #00E5FF;
        }
        .stat-label { color: #aaa; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
        .stat-value { font-size: 1.8rem; font-weight: bold; margin: 5px 0; color: #fff; }
        
        /* Progress Bar for Range */
        .range-container { background: #333; height: 6px; border-radius: 3px; margin-top: 10px; position: relative; }
        .range-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #FF1744, #00E676); }
        .range-marker { position: absolute; top: -4px; width: 4px; height: 14px; background: #fff; border-radius: 2px; }

        /* --- AI News Cards --- */
        .news-card {
            padding: 15px;
            margin-bottom: 15px;
            background: rgba(20, 20, 20, 0.8);
            border-radius: 12px;
            border-left: 5px solid #888;
        }
        .nc-pos { border-left-color: #00E676; box-shadow: -5px 0 15px -5px rgba(0, 230, 118, 0.2); }
        .nc-neg { border-left-color: #FF1744; box-shadow: -5px 0 15px -5px rgba(255, 23, 68, 0.2); }
        .nc-neu { border-left-color: #FFD600; }
        
        .sentiment-tag { font-size: 0.8rem; padding: 2px 8px; border-radius: 10px; font-weight: bold; margin-bottom: 5px; display: inline-block; }
        .st-pos { background: rgba(0, 230, 118, 0.2); color: #00E676; }
        .st-neg { background: rgba(255, 23, 68, 0.2); color: #FF1744; }
        .st-neu { background: rgba(255, 214, 0, 0.2); color: #FFD600; }

        /* Sidebar Buttons */
        div.stButton > button {
            width: 100%;
            background: transparent;
            border: 1px solid #333;
            color: #aaa;
            border-radius: 12px;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            border-color: #00E5FF;
            color: #00E5FF;
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
        }
        
        /* Entry Strategy Cards */
        .entry-box {
            background: linear-gradient(145deg, #111, #161616);
            border-radius: 15px;
            padding: 20px;
            border-left: 5px solid #555;
            margin-bottom: 15px;
        }
        .eb-1 { border-left-color: #00E5FF; }
        .eb-2 { border-left-color: #FFD600; }
        .eb-3 { border-left-color: #FF1744; }

    </style>
""", unsafe_allow_html=True)

# --- 3. Data Functions (Robust) ---

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
            trend = "Uptrend 🟢"
            signal = "BUY / LONG"
            color = "#00E676"
            sl = close - (1.5 * atr)
            tp = close + (2.5 * atr)
        elif close < ema50 and ema50 < ema200:
            trend = "Downtrend 🔴"
            signal = "SELL / SHORT"
            color = "#FF1744"
            sl = close + (1.5 * atr)
            tp = close - (2.5 * atr)
        else:
            trend = "Sideways 🟡"
            signal = "WAIT"
            color = "#888"
            sl = close - atr
            tp = close + atr
            
        return {'trend': trend, 'signal': signal, 'color': color, 'entry': close, 'sl': sl, 'tp': tp, 'atr': atr}
    except: return None

# --- AI News Analysis Function ---
@st.cache_data(ttl=3600)
def get_ai_analyzed_news(symbol):
    news_list = []
    clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
    try:
        # Fetch News
        q = urllib.parse.quote(f"site:bloomberg.com {clean_sym} market")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        # Fallback if empty
        if len(feed.entries) == 0:
            q = urllib.parse.quote(f"{clean_sym} crypto finance news")
            rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)

        # Analyze
        for item in feed.entries[:5]:
            blob = TextBlob(item.title)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.1:
                sentiment = "Positive (ข่าวดี)"
                color_class = "nc-pos"
                tag_class = "st-pos"
                icon = "🚀"
            elif sentiment_score < -0.1:
                sentiment = "Negative (ข่าวลบ)"
                color_class = "nc-neg"
                tag_class = "st-neg"
                icon = "🔻"
            else:
                sentiment = "Neutral (ทั่วไป)"
                color_class = "nc-neu"
                tag_class = "st-neu"
                icon = "⚖️"
                
            news_list.append({
                'title': item.title,
                'link': item.link,
                'sentiment': sentiment,
                'class': color_class,
                'tag_class': tag_class,
                'icon': icon,
                'score': sentiment_score
            })
    except: pass
    return news_list

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00E5FF;'>💎 SMART AI</h2>", unsafe_allow_html=True)
    st.caption("Premium Edition by KRITCHANUT")
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
st.markdown("<h3 style='margin-bottom: 5px;'>🔍 ค้นหาเหรียญ / หุ้น (Premium Search)</h3>", unsafe_allow_html=True)

c_search, c_btn = st.columns([4, 1])
with c_search:
    sym_input = st.text_input("ระบุชื่อย่อ (เช่น BTC-USD, AAPL)", value=st.session_state.symbol, label_visibility="collapsed")
with c_btn:
    if st.button("วิเคราะห์ ⚡", use_container_width=True):
        st.session_state.symbol = sym_input
        st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner('💎 AI กำลังวิเคราะห์ข้อมูลและข่าวสาร...'):
        df = get_data(symbol, period, "1d")
        
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล '{symbol}' กรุณาลองตรวจสอบชื่อย่อใหม่")
    else:
        # Calculation
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = curr_price - prev_price
        pct = (change / prev_price) * 100
        last_date = df.index[-1].strftime('%d-%m-%Y')
        
        setup = calculate_trade_setup(df)

        # --- HERO SECTION ---
        color_trend = "#00E676" if change >= 0 else "#FF1744"
        arrow = "▲" if change >= 0 else "▼"
        
        st.markdown(f"""
        <div class="glass-card" style="border-top: 5px solid {color_trend}; text-align: center;">
            <div style="font-size: 1.2rem; color: #aaa; letter-spacing: 2px; text-transform: uppercase;">ASSET ANALYSIS</div>
            <div style="font-size: 4rem; font-weight: 800; margin: 10px 0; background: -webkit-linear-gradient(45deg, #fff, {color_trend}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {symbol}
            </div>
            <div style="font-size: 3.5rem; font-weight: bold; color: {color_trend};">
                {curr_price:,.2f} 
            </div>
            <div style="color: #888; margin-bottom: 10px;">Price Date: {last_date}</div>
            <div class="status-badge {'badge-up' if change >= 0 else 'badge-down'}" style="font-size: 1.2rem;">
                {arrow} {abs(change):,.2f} ({pct:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- TABS ---
        tabs = st.tabs(["📈 Smart Chart", "📊 Pro Stats", "🤖 AI News", "🎯 Setup", "💰 Entry"])

        # 1. CHART
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['HA_Open'], high=ha['HA_High'], low=ha['HA_Low'], close=ha['HA_Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF', width=1.5), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=200).mean(), line=dict(color='#FF9100', width=1.5), name='EMA 200'), row=1, col=1)
            
            rsi = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0,0).rolling(14).mean() / abs(df['Close'].diff().where(lambda x: x<0,0)).rolling(14).mean())))
            fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#E040FB'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            
            fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # 2. PRO STATS (NEW DESIGN)
        with tabs[1]:
            st.markdown("### 📊 Market Statistics")
            high_p = df['High'].max()
            low_p = df['Low'].min()
            avg_p = df['Close'].mean()
            volatility = df['Close'].std()
            
            # Position of current price within High-Low Range (0-100%)
            range_pos = ((curr_price - low_p) / (high_p - low_p)) * 100 if (high_p - low_p) != 0 else 50
            
            # Row 1
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-label">Highest ({period})</div>
                    <div class="stat-value" style="color:#00E676;">{high_p:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-label">Lowest ({period})</div>
                    <div class="stat-value" style="color:#FF1744;">{low_p:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-label">Volatility (SD)</div>
                    <div class="stat-value" style="color:#E040FB;">{volatility:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            # Row 2: Range Visualizer
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="glass-card" style="padding: 15px;">
                <div style="display:flex; justify-content:space-between; font-size:0.9rem; color:#aaa;">
                    <span>Low: {low_p:,.2f}</span>
                    <span>Current Range Position</span>
                    <span>High: {high_p:,.2f}</span>
                </div>
                <div class="range-container">
                    <div class="range-fill" style="width: 100%;"></div>
                    <div class="range-marker" style="left: {range_pos}%; background: white; box-shadow: 0 0 10px white;"></div>
                </div>
                <div style="text-align:center; margin-top:5px; font-weight:bold;">
                    {range_pos:.1f}% from Low
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. AI NEWS ANALYST (NEW)
        with tabs[2]:
            st.markdown("### 🤖 AI News & Sentiment Analysis")
            st.caption("AI วิเคราะห์หัวข้อข่าวและประเมินผลกระทบต่อราคา (Source: Bloomberg/Global)")
            
            analyzed_news = get_ai_analyzed_news(symbol)
            
            if analyzed_news:
                # Summary Score
                total_score = sum([n['score'] for n in analyzed_news])
                if total_score > 0.2: 
                    overall = "Bullish (แนวโน้มดี) 🚀"
                    ov_color = "#00E676"
                elif total_score < -0.2: 
                    overall = "Bearish (ระวังแรงขาย) 🔻"
                    ov_color = "#FF1744"
                else: 
                    overall = "Neutral (ทรงตัว) ⚖️"
                    ov_color = "#FFD600"
                
                st.markdown(f"""
                <div style="text-align:center; margin-bottom:20px; padding:10px; border:1px solid {ov_color}; border-radius:10px;">
                    <span style="color:#aaa;">AI Overall Sentiment:</span> 
                    <span style="font-size:1.5rem; font-weight:bold; color:{ov_color}; margin-left:10px;">{overall}</span>
                </div>
                """, unsafe_allow_html=True)

                for news in analyzed_news:
                    st.markdown(f"""
                    <div class="news-card {news['class']}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="sentiment-tag {news['tag_class']}">{news['icon']} {news['sentiment']}</span>
                        </div>
                        <a href="{news['link']}" target="_blank" style="text-decoration:none; color:#fff;">
                            <h4 style="margin:5px 0;">{news['title']}</h4>
                        </a>
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
                        <div style="color: #aaa;">SIGNAL</div>
                        <div style="font-size: 2rem; font-weight: bold; color: {setup['color']};">{setup['signal']}</div>
                        <hr style="border-color: #333;">
                        <div style="color: #aaa;">TREND</div>
                        <div style="font-size: 1.2rem;">{setup['trend']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="display: flex; gap: 10px; height: 100%;">
                        <div class="glass-card" style="flex: 1; text-align: center; border: 1px solid #FF1744;">
                            <div style="color: #FF1744; font-weight: bold;">STOP LOSS</div>
                            <div style="font-size: 1.8rem;">{setup['sl']:,.2f}</div>
                        </div>
                        <div class="glass-card" style="flex: 1; text-align: center; border: 1px solid #00E676;">
                            <div style="color: #00E676; font-weight: bold;">TAKE PROFIT</div>
                            <div style="font-size: 1.8rem;">{setup['tp']:,.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # 5. ENTRY
        with tabs[4]:
            st.markdown("### 💰 Smart Money Management")
            t1 = curr_price * 0.995 
            t2 = curr_price * 0.98  
            t3 = curr_price * 0.95  
            
            st.markdown(f"""
            <div class="entry-box eb-1">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#00E5FF; font-weight:bold; font-size:1.2rem;">🔹 ไม้ที่ 1 : Probe Buy (20%)</span>
                    <span style="font-weight:bold; font-size:1.2rem;">{t1:,.2f}</span>
                </div>
                <small style="color:#aaa;">เข้าซื้อเพื่อทดสอบตลาด หรือเกาะแนวโน้มระยะสั้น</small>
            </div>
            
            <div class="entry-box eb-2">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#FFD600; font-weight:bold; font-size:1.2rem;">🔸 ไม้ที่ 2 : Accumulate (30%)</span>
                    <span style="font-weight:bold; font-size:1.2rem;">{t2:,.2f}</span>
                </div>
                <small style="color:#aaa;">จุดเข้าซื้อเพิ่มเมื่อราคาย่อตัวลงมา (Dip Buying)</small>
            </div>
            
            <div class="entry-box eb-3">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#FF1744; font-weight:bold; font-size:1.2rem;">🔻 ไม้ที่ 3 : Sniper Zone (50%)</span>
                    <span style="font-weight:bold; font-size:1.2rem;">{t3:,.2f}</span>
                </div>
                <small style="color:#aaa;">จุดซื้อของถูกเมื่อเกิด Panic Sell หรือแนวรับสำคัญมาก</small>
            </div>
            """, unsafe_allow_html=True)
