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
from newspaper import Article # พระเอกของเราสำหรับดึงเนื้อหา
import nltk

# Config NLTK เล็กน้อยเพื่อป้องกัน Error บน Cloud
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Smart Trader AI",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed"
)

# CSS: แต่งหน้าตาให้สวย อ่านง่าย
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        
        div[data-testid="stTextInput"] input {
            font-size: 22px !important; height: 55px !important;
            border-radius: 12px !important; padding-left: 15px !important;
            background-color: #222 !important; color: #fff !important;
            border: 2px solid #555 !important;
        }
        div[data-testid="stButton"] button {
            height: 55px !important; font-size: 22px !important;
            border-radius: 12px !important; width: 100% !important;
            background-color: #2962FF !important; color: white !important; border: none !important;
        }
        
        .news-card {
            background-color: #1E1E1E; padding: 20px; border-radius: 12px;
            margin-bottom: 20px; border-left: 6px solid #555;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .news-title { font-size: 1.3rem; font-weight: bold; margin-bottom: 10px; color: #fff; }
        .news-body { font-size: 1rem; color: #ccc; line-height: 1.6; }
        .news-meta { font-size: 0.8rem; color: #777; margin-top: 10px; display: flex; justify-content: space-between; }
        
        button[data-baseweb="tab"] { font-size: 1.1rem !important; padding: 15px !important; flex: 1; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Functions ---

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
        if not clusters: clusters.append({'p': l['p'], 'c': 1, 't': l['t']}); continue
        if abs(l['p'] - clusters[-1]['p']) < threshold:
            clusters[-1]['c'] += 1
            clusters[-1]['p'] = (clusters[-1]['p'] * (clusters[-1]['c']-1) + l['p']) / clusters[-1]['c']
        else: clusters.append({'p': l['p'], 'c': 1, 't': l['t']})
    results = []
    for c in clusters:
        label = "แข็งแกร่ง 🔥" if c['c'] >= 3 else "ปกติ"
        if c['c'] == 1: label = "บาง ☁️"
        results.append({'price': c['p'], 'type': c['t'], 'label': label, 'score': c['c']})
    return results

# --- ฟังก์ชันใหม่: ดึงเนื้อหาข่าว + แปล ---
@st.cache_data(ttl=3600) # Cache ไว้ 1 ชม. จะได้ไม่โหลดซ้ำให้ช้า
def fetch_and_translate_news(url, description):
    try:
        # 1. พยายามเจาะเนื้อหาข่าว (Scraping)
        article = Article(url)
        article.download()
        article.parse()
        content = article.text
        
        # ถ้าเจาะไม่เข้า หรือเนื้อหาสั้นเกินไป ให้ใช้ Description จาก RSS แทน
        if len(content) < 100:
            content = BeautifulSoup(description, "html.parser").get_text()
            
        # 2. ตัดเนื้อหาไม่ให้ยาวเกินไป (Google Translate มีลิมิต)
        # เอาสัก 800 ตัวอักษรพอให้อ่านรู้เรื่อง
        summary_en = content[:800] + ("..." if len(content) > 800 else "")
        
        # 3. แปลเป็นไทย
        summary_th = GoogleTranslator(source='auto', target='th').translate(summary_en)
        return summary_th
    except Exception as e:
        # ถ้า Error ให้แปล Description แทน
        try:
            desc_clean = BeautifulSoup(description, "html.parser").get_text()
            return GoogleTranslator(source='auto', target='th').translate(desc_clean[:500])
        except:
            return "ไม่สามารถดึงเนื้อหาได้"

def get_news_feed(query):
    try:
        q = query.replace("-THB", "").replace("-USD", "")
        url = f"https://news.google.com/rss/search?q={q}+when:2d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        return feed.entries[:5] # เอาแค่ 5 ข่าวพอ เดี๋ยวโหลดนาน
    except: return []

# --- 3. UI Layout ---

with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    period = st.selectbox("ช่วงเวลา", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    interval = st.selectbox("Timeframe", ["1d", "1wk"], index=0)
    show_ema = st.checkbox("EMA 50/200", True)
    show_rsi = st.checkbox("RSI", True)

st.markdown("### 🔎 ค้นหาหุ้น / เหรียญ")
col_in, col_btn = st.columns([3.5, 1])
with col_in: symbol_input = st.text_input("Search", value="BTC-THB", label_visibility="collapsed")
with col_btn: search_pressed = st.button("GO")

symbol = symbol_input.upper().strip()

if symbol:
    with st.spinner('⏳ กำลังดึงข้อมูล...'):
        df = get_data(symbol, period, interval)
    
    if df.empty:
        st.warning(f"ไม่พบข้อมูล '{symbol}'")
    else:
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        
        price = df['Close'].iloc[-1]
        change = price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        color_p = "#00E676" if change >= 0 else "#FF1744"
        trend_txt, trend_col, rsi_txt = analyze_trend(df)
        levels = analyze_levels(df)
        
        st.markdown(f"""
        <div style="background:#111; padding:20px; border-radius:15px; border-top:5px solid {color_p}; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.5); margin-bottom:20px;">
            <p style="font-size:1.2rem; color:#aaa; margin:0;">{symbol}</p>
            <p style="font-size:3rem; font-weight:bold; margin:0; line-height:1.2; color:{color_p};">{price:,.2f}</p>
            <p style="margin-top:5px; font-size:1.2rem; color:{color_p};">{change:+,.2f} ({pct:+.2f}%)</p>
            <div style="margin-top:10px; padding:5px; background:{trend_col}22; border-radius:5px; color:{trend_col}; font-weight:bold; font-size:0.9rem;">
                {trend_txt} | RSI: {rsi_txt}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 กราฟ", "🧱 แนวรับต้าน", "📰 ข่าวแปลไทย"])
        
        with tab1:
            fig = make_subplots(rows=2 if show_rsi else 1, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3] if show_rsi else [1.0])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
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
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=10), xaxis_rangeslider_visible=False, template="plotly_dark", dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with tab2:
            res = sorted([l for l in levels if l['type']=='Resistance' and l['price']>price], key=lambda x: x['price'])[:4]
            sup = sorted([l for l in levels if l['type']=='Support' and l['price']<price], key=lambda x: x['price'], reverse=True)[:4]
            st.markdown("#### 🟥 ต้าน (Sell)"); 
            for r in reversed(res): st.markdown(f"<div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #333;'><span style='color:#888;'>{r['label']}</span><span style='color:#FF5252; font-weight:bold;'>{r['price']:,.2f}</span></div>", unsafe_allow_html=True)
            st.markdown("#### 🟩 รับ (Buy)"); 
            for s in sup: st.markdown(f"<div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #333;'><span style='color:#888;'>{s['label']}</span><span style='color:#00E676; font-weight:bold;'>{s['price']:,.2f}</span></div>", unsafe_allow_html=True)

        with tab3:
            st.markdown("waiting... (กำลังเจาะลิงก์ข่าวและแปลไทย อาจใช้เวลาสักครู่)")
            news_items = get_news_feed(symbol)
            if not news_items:
                st.info("ไม่พบข่าวในขณะนี้")
            else:
                progress_bar = st.progress(0)
                for i, item in enumerate(news_items):
                    # Sentiment
                    blob = TextBlob(item.title)
                    score = blob.sentiment.polarity
                    color_bar = "#00C853" if score > 0.1 else "#FF1744" if score < -0.1 else "#9E9E9E"
                    icon = "🟢 ข่าวดี" if score > 0.1 else "🔴 ข่าวร้าย" if score < -0.1 else "⚪ ทั่วไป"

                    # แปลหัวข้อ
                    try: title_th = GoogleTranslator(source='auto', target='th').translate(item.title)
                    except: title_th = item.title
                    
                    # แปลเนื้อหา (ตัวสำคัญ)
                    body_th = fetch_and_translate_news(item.link, item.get('description', ''))
                    
                    # Card UI
                    st.markdown(f"""
                    <div class="news-card" style="border-left-color: {color_bar};">
                        <div class="news-title">{title_th}</div>
                        <div style="margin-bottom:8px; font-weight:bold; color:{color_bar};">{icon}</div>
                        <div class="news-body">{body_th}</div>
                        <div class="news-meta">
                            <span>Original: {item.source.title if 'source' in item else 'Unknown'}</span>
                            <a href="{item.link}" target="_blank" style="color:#448AFF; text-decoration:none;">อ่านต้นฉบับ 🔗</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    progress_bar.progress((i + 1) / len(news_items))
                progress_bar.empty()
