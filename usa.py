import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from deep_translator import GoogleTranslator
import feedparser
from bs4 import BeautifulSoup

# --- 1. ตั้งค่าหน้าเว็บ (Mobile Big Search) ---
st.set_page_config(
    page_title="Smart Trader AI",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed"
)

# CSS: ปรับช่องค้นหาให้ใหญ่ยักษ์ (Big Input)
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        
        /* 1. ปรับช่องกรอกข้อความ (Input) ให้ใหญ่ */
        div[data-testid="stTextInput"] input {
            font-size: 24px !important;      /* ตัวหนังสือใหญ่ */
            height: 60px !important;         /* กล่องสูง จิ้มง่าย */
            border-radius: 15px !important;  /* ขอบมน */
            padding-left: 15px !important;
            background-color: #222 !important; /* พื้นหลังสีเข้ม */
            color: #fff !important;            /* ตัวหนังสือสีขาว */
            border: 2px solid #555 !important;
        }
        
        /* 2. ปรับปุ่มกด (Button) ให้ใหญ่เท่ากัน */
        div[data-testid="stButton"] button {
            height: 60px !important;
            font-size: 24px !important;
            border-radius: 15px !important;
            width: 100% !important;
            background-color: #2962FF !important; /* สีน้ำเงินสด */
            color: white !important;
            border: none !important;
        }
        
        /* 3. กล่องราคา */
        .price-box {
            background: #111; padding: 20px; border-radius: 15px;
            text-align: center; border: 1px solid #333; margin-top: 10px; margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        .price-text { font-size: 3rem; font-weight: bold; margin: 0; line-height: 1.2; }
        .symbol-text { font-size: 1.2rem; color: #aaa; margin: 0; letter-spacing: 2px; }
        
        /* 4. Tab ตัวหนังสือใหญ่ */
        button[data-baseweb="tab"] { 
            font-size: 1.2rem !important; 
            padding: 15px !important;
            flex: 1; /* ขยายให้เต็มความกว้าง */
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. Functions ---

def clean_html(html_text):
    try: return BeautifulSoup(html_text, "html.parser").get_text()
    except: return html_text

def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
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
    
    rsi_st = "Overbought 🥵" if rsi > 70 else "Oversold 🥶" if rsi < 30 else "ปกติ 😐"
    return trend, color, rsi_st

def analyze_levels(df):
    levels = []
    for i in range(2, df.shape[0]-2):
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1]:
            levels.append({'p': df['Low'][i], 't': 'Support'})
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1]:
            levels.append({'p': df['High'][i], 't': 'Resistance'})
            
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

# --- 3. UI Layout ---

# Sidebar Setting
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    period = st.selectbox("ช่วงเวลา", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    interval = st.selectbox("Timeframe", ["1d", "1wk"], index=0)
    show_ema = st.checkbox("EMA 50/200", True)
    show_rsi = st.checkbox("RSI", True)

# Search Bar Area (Big Size)
st.markdown("### 🔎 ค้นหาหุ้น / เหรียญ")
col_in, col_btn = st.columns([3.5, 1]) # แบ่งสัดส่วน กว้าง : แคบ

with col_in:
    # label_visibility="collapsed" เพื่อซ่อนป้ายชื่อ input (ให้ดูคลีนๆ)
    symbol_input = st.text_input("Search", value="BTC-THB", label_visibility="collapsed", placeholder="เช่น BTC-USD, PTT.BK")

with col_btn:
    # ปุ่มกดจะใหญ่ตาม CSS ที่เราเขียนไว้
    search_pressed = st.button("GO")

symbol = symbol_input.upper().strip()

# Main Logic
if symbol:
    with st.spinner('⏳ กำลังโหลด...'):
        df = get_data(symbol, period, interval)
    
    if df.empty:
        st.warning(f"ไม่พบข้อมูล '{symbol}' ลองเติม .BK ถ้าเป็นหุ้นไทย (เช่น PTT.BK)")
    else:
        # Indicators
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        
        price = df['Close'].iloc[-1]
        change = price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        color_p = "#00E676" if change >= 0 else "#FF1744"
        
        trend_txt, trend_col, rsi_txt = analyze_trend(df)
        levels = analyze_levels(df)
        
        # Display Price Box
        st.markdown(f"""
        <div class="price-box" style="border-top: 5px solid {color_p};">
            <p class="symbol-text">{symbol}</p>
            <p class="price-text" style="color: {color_p};">{price:,.2f}</p>
            <p style="margin-top:5px; font-size:1.2rem; color: {color_p};">
                {change:+,.2f} ({pct:+.2f}%)
            </p>
            <div style="margin-top:15px; padding:8px; background:{trend_col}22; border-radius:8px; color:{trend_col}; font-weight:bold;">
                {trend_txt} <br> <span style="font-size:0.9rem; color:#aaa;">RSI: {rsi_txt}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 กราฟ", "🧱 แนวรับต้าน", "📰 ข่าว"])
        
        with tab1:
            fig = make_subplots(rows=2 if show_rsi else 1, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3] if show_rsi else [1.0])
            
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            if show_ema:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2979FF', width=1), name="EMA50"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#FF9100', width=1), name="EMA200"), row=1, col=1)
                
            for l in levels:
                if l['score'] >= 3:
                    c = 'green' if l['type']=='Support' else 'red'
                    fig.add_hline(y=l['price'], line_dash='solid', line_color=c, opacity=0.5, row=1, col=1)

            if show_rsi:
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AA00FF')), row=2, col=1)
                fig.add_hline(y=70, line_dash='dot', line_color='red', row=2, col=1)
                fig.add_hline(y=30, line_dash='dot', line_color='green', row=2, col=1)

            fig.update_layout(
                height=500, margin=dict(l=0, r=0, t=10, b=10),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)'),
                template="plotly_dark", dragmode='pan'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with tab2:
            res = sorted([l for l in levels if l['type']=='Resistance' and l['price']>price], key=lambda x: x['price'])[:4]
            sup = sorted([l for l in levels if l['type']=='Support' and l['price']<price], key=lambda x: x['price'], reverse=True)[:4]
            
            st.markdown("#### 🟥 ต้าน (Sell)")
            if not res: st.caption("- ว่าง -")
            for r in reversed(res):
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:15px; border-bottom:1px solid #333; font-size:1.2rem;">
                    <span style="color:#888;">{r['label']}</span>
                    <span style="color:#FF5252; font-weight:bold;">{r['price']:,.2f}</span>
                </div>""", unsafe_allow_html=True)
                
            st.markdown("#### 🟩 รับ (Buy)")
            if not sup: st.caption("- ว่าง -")
            for s in sup:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:15px; border-bottom:1px solid #333; font-size:1.2rem;">
                    <span style="color:#888;">{s['label']}</span>
                    <span style="color:#00E676; font-weight:bold;">{s['price']:,.2f}</span>
                </div>""", unsafe_allow_html=True)
                
        with tab3:
            news = get_news_mobile(symbol)
            if not news:
                st.info("ไม่พบข่าว")
            else:
                for item in news:
                    blob = TextBlob(item.title)
                    score = blob.sentiment.polarity
                    icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                    try: title_th = GoogleTranslator(source='auto', target='th').translate(item.title)
                    except: title_th = item.title
                    
                    st.markdown(f"""
                    <div style="background:#222; padding:15px; border-radius:10px; margin-bottom:15px;">
                        <div style="font-size:1.1rem; font-weight:bold; margin-bottom:5px;">{icon} {title_th}</div>
                        <div style="font-size:0.9rem; color:#888;">{item.title}</div>
                        <a href="{item.link}" style="display:block; margin-top:10px; color:#448AFF; font-size:1rem;">🔗 อ่านต่อ</a>
                    </div>
                    """, unsafe_allow_html=True)
