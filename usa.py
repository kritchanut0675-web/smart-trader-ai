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
    page_title="Smart Trader AI : Ultimate",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# --- 2. Ultra Black CSS ---
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
        .stat-box {
            background: #0a0a0a; border-radius: 20px; padding: 25px;
            text-align: center; border: 1px solid #333; margin-bottom: 15px;
        }
        .stat-label { color: #888; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1.5px; }
        .stat-value { font-size: 2.5rem; font-weight: 800; color: #fff; }

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

        /* GURU CARD */
        .guru-card {
            background: #111; border-radius: 15px; padding: 20px; border-left: 5px solid #FFD600;
            margin-bottom: 15px; border: 1px solid #333;
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

# --- FIX: Return ONLY DataFrame (Cacheable) ---
@st.cache_data(ttl=300)
def get_market_data(symbol, period, interval):
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

# --- AI & Guru Analysis ---

def get_guru_opinion(ticker, current_price):
    """
    ดึงข้อมูลจาก Analyst Target ของ Yahoo Finance
    """
    try:
        info = ticker.info
        # Check if keys exist to prevent errors
        if 'targetMeanPrice' not in info: return None
        
        target_mean = info.get('targetMeanPrice')
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        recommendation = info.get('recommendationKey', 'none').upper()
        num_analysts = info.get('numberOfAnalystOpinions', 0)
        
        return {
            'target_mean': target_mean,
            'target_high': target_high,
            'target_low': target_low,
            'rec': recommendation,
            'count': num_analysts
        }
    except:
        return None

def generate_ai_analysis(df, setup, guru_data, sentiment_score):
    """
    สร้างบทวิเคราะห์ภาษาไทยจากข้อมูลที่มี
    """
    analysis_text = ""
    score = 50 # Base score
    
    # 1. Technical Analysis
    if setup['trend'] == "Uptrend (ขาขึ้น)":
        analysis_text += "📈 **ด้านเทคนิค:** กราฟยังคงรักษาทรงขาขึ้นได้อย่างแข็งแกร่ง ราคายืนเหนือเส้นค่าเฉลี่ย EMA50 และ EMA200 แสดงถึงแรงซื้อที่ยังได้เปรียบ "
        score += 20
    elif setup['trend'] == "Downtrend (ขาลง)":
        analysis_text += "📉 **ด้านเทคนิค:** กราฟอยู่ในแนวโน้มขาลงชัดเจน ราคายังไม่สามารถยืนเหนือเส้นค่าเฉลี่ยหลักได้ ระวังแรงขายกดดันต่อเนื่อง "
        score -= 20
    else:
        analysis_text += "⚖️ **ด้านเทคนิค:** กราฟแกว่งตัวออกข้าง (Sideways) ยังไม่มีทิศทางชัดเจน แนะนำให้รอการเลือกทาง "

    # RSI Logic
    if setup['rsi'] > 70:
        analysis_text += "แต่ RSI เริ่มเข้าเขต Overbought (ซื้อมากเกินไป) อาจมีการย่อตัวระยะสั้น "
        score -= 5
    elif setup['rsi'] < 30:
        analysis_text += "RSI เข้าเขต Oversold (ขายมากเกินไป) อาจมีแรงเด้งรีบาวด์ได้ "
        score += 5
    
    # 2. Guru/Fundamental Logic
    if guru_data and guru_data['target_mean']:
        upside = ((guru_data['target_mean'] - setup['entry']) / setup['entry']) * 100
        if upside > 10:
            analysis_text += f"\n\n👥 **มุมมองกูรู:** นักวิเคราะห์มองเป้าหมายเฉลี่ยที่ {guru_data['target_mean']:,.2f} ซึ่งยังมี Upside อีกประมาณ {upside:.1f}% "
            score += 10
        elif upside < -10:
            analysis_text += f"\n\n👥 **มุมมองกูรู:** ราคาปัจจุบันสูงกว่าเป้าหมายเฉลี่ยของนักวิเคราะห์ ({guru_data['target_mean']:,.2f}) ควรระวังแรงขายทำกำไร "
            score -= 10
    else:
        # Crypto or No Data
        analysis_text += "\n\n👥 **มุมมองกูรู:** ไม่มีข้อมูลเป้าหมายราคาจากนักวิเคราะห์ (อาจเป็นสินทรัพย์ทางเลือกหรือคริปโต) ให้เน้นดูปัจจัยทางเทคนิคเป็นหลัก "

    # 3. Sentiment Logic (Simulated)
    if sentiment_score > 0:
        analysis_text += "\n\n📰 **กระแสข่าว:** ภาพรวมข่าวสารอยู่ในเกณฑ์เชิงบวก สนับสนุนราคา"
        score += 10
    elif sentiment_score < 0:
        analysis_text += "\n\n📰 **กระแสข่าว:** มีข่าวเชิงลบกดดันตลาดในช่วงนี้ ควรติดตามสถานการณ์ใกล้ชิด"
        score -= 10
        
    # Clamp score
    score = max(0, min(100, score))
    
    if score >= 75: verdict = "STRONG BUY"
    elif score >= 55: verdict = "BUY"
    elif score >= 45: verdict = "HOLD / WATCH"
    elif score >= 25: verdict = "SELL"
    else: verdict = "STRONG SELL"
    
    return analysis_text, score, verdict

# --- AI News (Simplified for Speed) ---
@st.cache_data(ttl=3600)
def get_news_sentiment(symbol):
    try:
        clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
        q = urllib.parse.quote(f"site:bloomberg.com {clean_sym} market")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        total_score = 0
        count = 0
        news_items = []
        
        for item in feed.entries[:3]:
            blob = TextBlob(item.title)
            total_score += blob.sentiment.polarity
            count += 1
            news_items.append({'title': item.title, 'link': item.link})
            
        avg_score = total_score / count if count > 0 else 0
        return avg_score, news_items
    except: return 0, []

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
    with st.spinner('💎 AI กำลังรวบรวมข้อมูลจากกูรูและตลาด...'):
        # --- FIX: Retrieve only DF from Cache ---
        df = get_market_data(symbol, period, "1d")
        
        # --- FIX: Instantiate Ticker object freshly here (Avoids Cache Error) ---
        ticker = yf.Ticker(symbol)
        
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล '{symbol}'")
    else:
        # Process Data
        curr_price = df['Close'].iloc[-1]
        change = curr_price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        
        setup = calculate_technical_setup(df)
        guru_data = get_guru_opinion(ticker, curr_price)
        sent_score, news_list = get_news_sentiment(symbol)
        
        ai_text, ai_score, ai_verdict = generate_ai_analysis(df, setup, guru_data, sent_score)

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
        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 News", "🎯 Setup", "💰 Entry", "🗣️ Guru View", "🤖 AI Verdict"])

        # Tab 1-5 (Keeping Original Logic for brevity, inserting only specific code for new tabs)
        with tabs[0]: # Chart
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

        with tabs[1]: # Stats
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class="stat-box"><div class="stat-label">High</div><div class="stat-value" style="color:#00E676;">{df['High'].max():,.2f}</div></div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="stat-box"><div class="stat-label">Low</div><div class="stat-value" style="color:#FF1744;">{df['Low'].min():,.2f}</div></div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="stat-box"><div class="stat-label">Vol</div><div class="stat-value" style="color:#E040FB;">{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>""", unsafe_allow_html=True)

        with tabs[2]: # News
            if news_list:
                for n in news_list:
                    st.markdown(f"""<div class="glass-card" style="padding:15px;"><a href="{n['link']}" style="color:#fff; text-decoration:none;"><h4>{n['title']}</h4></a></div>""", unsafe_allow_html=True)
            else: st.info("No News Found")

        with tabs[3]: # Setup
             if setup:
                st.markdown(f"""<div class="glass-card" style="text-align:center; border:2px solid {setup['color']}"><h1 style="color:{setup['color']}">{setup['signal']}</h1><p>{setup['trend']}</p></div>""", unsafe_allow_html=True)

        with tabs[4]: # Entry
            st.markdown("### 💰 Entry Levels")
            t1, t2, t3 = curr_price*0.99, curr_price*0.97, curr_price*0.94
            st.markdown(f"""<div style="background:#111; padding:15px; border-left:5px solid #00E5FF; margin-bottom:10px;"><b>Probe Buy:</b> {t1:,.2f}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:#111; padding:15px; border-left:5px solid #FFD600; margin-bottom:10px;"><b>Accumulate:</b> {t2:,.2f}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:#111; padding:15px; border-left:5px solid #FF1744;"><b>Sniper Zone:</b> {t3:,.2f}</div>""", unsafe_allow_html=True)

        # --- TAB 6: GURU VIEW ---
        with tabs[5]:
            st.markdown("### 🗣️ Guru Opinions (ความเห็นนักวิเคราะห์)")
            
            if guru_data and guru_data['target_mean']:
                # Recommendations
                rec_color = "#00E676" if "BUY" in guru_data['rec'] else "#FF1744" if "SELL" in guru_data['rec'] else "#FFD600"
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="color:#aaa; margin-bottom:5px;">WALL STREET CONSENSUS</div>
                    <div style="font-size:3rem; font-weight:bold; color:{rec_color};">{guru_data['rec']}</div>
                    <div style="color:#888;">Based on {guru_data['count']} Analysts</div>
                </div>
                """, unsafe_allow_html=True)

                # Target Prices
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f"""<div class="guru-card"><div class="stat-label">Low Target</div><div style="font-size:1.5rem; font-weight:bold;">{guru_data['target_low']:,.2f}</div></div>""", unsafe_allow_html=True)
                with c2: st.markdown(f"""<div class="guru-card" style="border-left-color:#00E5FF;"><div class="stat-label">Average Target</div><div style="font-size:1.5rem; font-weight:bold; color:#00E5FF;">{guru_data['target_mean']:,.2f}</div></div>""", unsafe_allow_html=True)
                with c3: st.markdown(f"""<div class="guru-card" style="border-left-color:#00E676;"><div class="stat-label">High Target</div><div style="font-size:1.5rem; font-weight:bold; color:#00E676;">{guru_data['target_high']:,.2f}</div></div>""", unsafe_allow_html=True)
                
                # Upside Calculation
                upside = ((guru_data['target_mean'] - curr_price) / curr_price) * 100
                st.info(f"💡 จากราคาปัจจุบัน ({curr_price:,.2f}) ไปยังราคาเป้าหมายเฉลี่ย ({guru_data['target_mean']:,.2f}) คิดเป็น Upside/Downside ประมาณ **{upside:+.2f}%**")
                
            else:
                st.warning("⚠️ ไม่พบข้อมูลราคาเป้าหมายจากนักวิเคราะห์ (Analyst Targets) สำหรับสินทรัพย์นี้ หรือเป็นสินทรัพย์ประเภท Crypto/Forex ที่ไม่มีข้อมูล Consensus ในระบบนี้")
                st.markdown("แนะนำให้ใช้ **Tab 7 (AI Verdict)** เพื่อดูการวิเคราะห์ทางเทคนิคทดแทน")

        # --- TAB 7: AI VERDICT ---
        with tabs[6]:
            st.markdown("### 🤖 AI Market Analysis (บทวิเคราะห์อัจฉริยะ)")
            
            # Score Color
            if ai_score >= 70: score_color = "#00E676"
            elif ai_score <= 30: score_color = "#FF1744"
            else: score_color = "#FFD600"
            
            c_score, c_text = st.columns([1, 2])
            
            with c_score:
                st.markdown(f"""
                <div class="ai-card" style="text-align:center; border-color:{score_color};">
                    <div style="color:#aaa; margin-bottom:15px;">AI CONFIDENCE SCORE</div>
                    <div class="ai-score-circle" style="border-color:{score_color}; color:{score_color};">
                        {ai_score}
                    </div>
                    <div style="font-size:1.8rem; font-weight:bold; color:{score_color};">{ai_verdict}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_text:
                st.markdown(f"""
                <div class="glass-card" style="min-height: 250px;">
                    <h3 style="color:{score_color}; margin-top:0;">📝 AI Summary Report</h3>
                    <div style="font-size:1.1rem; line-height:1.6; color:#ddd;">
                        {ai_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption("หมายเหตุ: AI ประมวลผลจากข้อมูลทางเทคนิค (EMA, RSI) และความเห็นนักวิเคราะห์ (ถ้ามี) โปรดใช้วิจารณญาณในการลงทุน")
