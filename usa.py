import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from textblob import TextBlob
from deep_translator import GoogleTranslator
import feedparser
from bs4 import BeautifulSoup

# --- 1. ตั้งค่าหน้าเว็บ (Mobile Optimized) ---
st.set_page_config(
    page_title="Smart Trader AI Mobile",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed" # ปิด Sidebar เริ่มต้นเพื่อให้ดูกว้าง
)

# CSS: ปรับแต่งให้เข้ากับมือถือ
st.markdown("""
    <style>
        /* ลดขอบขาวด้านบน-ล่าง */
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        
        /* ปรับช่องค้นหาให้ใหญ่กดง่าย */
        div[data-testid="stTextInput"] input { font-size: 1.2rem; height: 50px; }
        
        /* กล่องราคา */
        .price-box {
            background: #111; padding: 15px; border-radius: 12px;
            text-align: center; border: 1px solid #333; margin-bottom: 15px;
        }
        .price-text { font-size: 2.5rem; font-weight: bold; margin: 0; }
        .symbol-text { font-size: 1rem; color: #888; margin: 0; }
        
        /* สถานะตลาด */
        .status-box {
            padding: 10px; border-radius: 8px; text-align: center; 
            font-weight: bold; margin-bottom: 10px; font-size: 1rem;
        }
        
        /* ตารางแนวรับต้าน */
        .sr-row {
            display: flex; justify-content: space-between; padding: 12px;
            border-bottom: 1px solid #222; font-size: 1.1rem;
        }
        .sr-label { font-size: 0.9rem; padding: 2px 8px; border-radius: 4px; }
        
        /* Tab ตัวหนังสือใหญ่ขึ้น */
        button[data-baseweb="tab"] { font-size: 1.1rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Functions (Logic เดิม แต่ปรับปรุงนิดหน่อย) ---

def clean_html(html_text):
    try: return BeautifulSoup(html_text, "html.parser").get_text()
    except: return html_text

def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        # Fallback for THB
        if df.empty and symbol.endswith("-THB"):
            base = symbol.replace("-THB", "-USD")
            df = yf.Ticker(base).history(period=period, interval=interval)
            usd = yf.Ticker("THB=X").history(period="1d")['Close'].iloc[-1]
            if not df.empty: df[['Open','High','Low','Close']] *= usd
            
        return df
    except: return pd.DataFrame()

def analyze_trend(df):
    close = df['Close'].iloc[-1]
    ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
    ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    if close > ema50 and close > ema200: trend, color = "🚀 ขาขึ้น (Uptrend)", "#00C853"
    elif close < ema50 and close < ema200: trend, color = "🔻 ขาลง (Downtrend)", "#FF1744"
    else: trend, color = "↔️ ไซด์เวย์ (Sideways)", "#FFD600"
    
    rsi_st = "Overbought 🥵" if rsi > 70 else "Oversold 🥶" if rsi < 30 else "Normal 😐"
    return trend, color, rsi_st

def analyze_levels(df):
    levels = []
    # Simple Fractal
    for i in range(2, df.shape[0]-2):
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1]:
            levels.append({'p': df['Low'][i], 't': 'Support'})
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1]:
            levels.append({'p': df['High'][i], 't': 'Resistance'})
            
    # Clustering
    levels.sort(key=lambda x: x['p'])
    clusters = []
    threshold = df['Close'].mean() * 0.015
    for l in levels:
        if not clusters:
            clusters.append({'p': l['p'], 'c': 1, 't': l['t']})
            continue
        if abs(l['p'] - clusters[-1]['p']) < threshold:
            clusters[-1]['c'] += 1
            clusters[-1]['p'] = (clusters[-1]['p'] * (clusters[-1]['c']-1) + l['p']) / clusters[-1]['c']
        else:
            clusters.append({'p': l['p'], 'c': 1, 't': l['t']})
            
    results = []
    for c in clusters:
        label = "แข็งแกร่ง 🔥" if c['c'] >= 3 else "ปกติ"
        if c['c'] == 1: label = "บาง ☁️"
        results.append({'price': c['p'], 'type': c['t'], 'label': label, 'score': c['c']})
    return results

def get_news_mobile(query):
    try:
        q = query.replace("-THB", "").replace("-USD", "")
        url = f"https://news.google.com/rss/search?q={q}+when:3d&hl=en-US&gl=US&ceid=US:en"
        return feedparser.parse(url).entries[:5]
    except: return []

# --- 3. UI Layout (Mobile First) ---

# ส่วนค้นหาอยู่บนสุด (ไม่ต้องเข้า Sidebar)
col_in, col_btn = st.columns([4, 1])
with col_in:
    symbol_input = st.text_input("ชื่อหุ้น/เหรียญ", value="BTC-THB", label_visibility="collapsed")
with col_btn:
    if st.button("🔎", help="ค้นหา", type="primary"):
        st.rerun()

symbol = symbol_input.upper().strip()

# Sidebar: เอาไว้แค่ Setting
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    period = st.selectbox("ช่วงเวลา", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    interval = st.selectbox("Timeframe", ["1d", "1wk"], index=0)
    st.caption("ปรับแต่งการแสดงผลกราฟ")
    show_ema = st.checkbox("EMA 50/200", True)
    show_rsi = st.checkbox("RSI", True)

# Main Logic
if symbol:
    with st.spinner('⏳ กำลังโหลด...'):
        df = get_data(symbol, period, interval)
    
    if df.empty:
        st.error("ไม่พบข้อมูล")
    else:
        # Calcs
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        
        price = df['Close'].iloc[-1]
        change = price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        color_p = "#00E676" if change >= 0 else "#FF1744"
        
        trend_txt, trend_col, rsi_txt = analyze_trend(df)
        levels = analyze_levels(df)
        
        # 1. Price Header (Big & Clear)
        st.markdown(f"""
        <div class="price-box" style="border-color: {color_p};">
            <p class="symbol-text">{symbol}</p>
            <p class="price-text" style="color: {color_p};">{price:,.2f}</p>
            <p style="margin:0; color: {color_p};">{change:+,.2f} ({pct:+.2f}%)</p>
        </div>
        <div class="status-box" style="background: {trend_col}33; color: {trend_col}; border: 1px solid {trend_col};">
            {trend_txt} | RSI: {rsi_txt}
        </div>
        """, unsafe_allow_html=True)
        
        # 2. TABS System (หัวใจของ Mobile UX)
        tab1, tab2, tab3 = st.tabs(["📊 กราฟ", "🧱 แนวรับ/ต้าน", "📰 ข่าว"])
        
        with tab1: # Chart Tab
            fig = make_subplots(rows=2 if show_rsi else 1, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3] if show_rsi else [1.0])
            
            # Candle
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # EMA
            if show_ema:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2979FF', width=1), name="EMA50"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#FF9100', width=1), name="EMA200"), row=1, col=1)
                
            # S/R Lines (บางๆ บนกราฟ)
            for l in levels:
                if l['score'] >= 3:
                    c = 'green' if l['type']=='Support' else 'red'
                    fig.add_hline(y=l['price'], line_dash='solid', line_color=c, opacity=0.5, row=1, col=1)

            # RSI
            if show_rsi:
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AA00FF')), row=2, col=1)
                fig.add_hline(y=70, line_dash='dot', line_color='red', row=2, col=1)
                fig.add_hline(y=30, line_dash='dot', line_color='green', row=2, col=1)

            # Mobile Chart Config
            fig.update_layout(
                height=500, # ความสูงพอดีมือถือ
                margin=dict(l=5, r=5, t=10, b=10), # ลดขอบ
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)'), # Legend แนวนอนด้านบน
                template="plotly_dark",
                dragmode='pan' # ใช้นิ้วลากกราฟได้เลย
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) # ซ่อนเมนูกวนใจ
            
        with tab2: # S/R Levels Tab
            col_res, col_sup = st.columns(2)
            
            # กรองเฉพาะใกล้ๆ
            res = sorted([l for l in levels if l['type']=='Resistance' and l['price']>price], key=lambda x: x['price'])[:4]
            sup = sorted([l for l in levels if l['type']=='Support' and l['price']<price], key=lambda x: x['price'], reverse=True)[:4]
            
            st.markdown("##### 🟥 แนวต้าน (Sell)")
            if not res: st.caption("ไม่พบแนวต้านใกล้เคียง")
            for r in reversed(res):
                st.markdown(f"""
                <div class="sr-row" style="border-left: 4px solid #FF5252;">
                    <span style="color:#aaa;">{r['label']}</span>
                    <span style="color:#FF5252; font-weight:bold;">{r['price']:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            st.markdown("##### 🟩 แนวรับ (Buy)")
            if not sup: st.caption("ไม่พบแนวรับใกล้เคียง")
            for s in sup:
                st.markdown(f"""
                <div class="sr-row" style="border-left: 4px solid #00E676;">
                    <span style="color:#aaa;">{s['label']}</span>
                    <span style="color:#00E676; font-weight:bold;">{s['price']:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
        with tab3: # News Tab
            news = get_news_mobile(symbol)
            if not news:
                st.info("ไม่พบข่าวใหม่")
            else:
                for item in news:
                    blob = TextBlob(item.title)
                    score = blob.sentiment.polarity
                    icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                    
                    try: title_th = GoogleTranslator(source='auto', target='th').translate(item.title)
                    except: title_th = item.title
                    
                    st.markdown(f"""
                    <div style="background:#222; padding:12px; border-radius:8px; margin-bottom:10px;">
                        <div style="font-weight:bold; margin-bottom:5px;">{icon} {title_th}</div>
                        <div style="font-size:0.8rem; color:#888;">{item.title}</div>
                        <a href="{item.link}" style="font-size:0.8rem; color:#448AFF;">อ่านต่อ...</a>
                    </div>
                    """, unsafe_allow_html=True)
