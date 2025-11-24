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
from newspaper import Article, Config
import nltk

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Smart Trader AI : Free News",
    layout="wide",
    page_icon="📰",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        div[data-testid="stTextInput"] input {
            font-size: 20px !important; height: 50px !important;
            border-radius: 12px !important; background-color: #1b1b1b !important;
            color: #fff !important; border: 1px solid #333 !important;
        }
        div[data-testid="stButton"] button {
            height: 50px !important; font-size: 20px !important;
            border-radius: 12px !important; width: 100% !important;
            background-color: #00897B !important; color: white !important; /* สีเขียว Investing */
            border: none !important; font-weight: bold !important;
        }
        .source-tag {
            font-size: 0.8rem; padding: 2px 6px; border-radius: 4px;
            background: #333; color: #ccc; margin-right: 5px; border: 1px solid #555;
        }
        .news-content { 
            font-size: 1rem; line-height: 1.7; color: #ddd; 
            text-align: justify; background: #1a1a1a; padding: 15px; border-radius: 10px;
        }
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

def analyze_ai_signal(df):
    close = df['Close'].iloc[-1]
    ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    if close > ema200:
        if rsi < 30: return "🟢 เข้าซื้อ (Strong Buy)", "#00E676", "เทรนด์ขาขึ้น + ย่อตัวหนัก (Oversold)"
        elif rsi < 50: return "🟢 ทยอยสะสม (Buy)", "#66BB6A", "เทรนด์ขาขึ้น ราคายังไม่แพง"
        elif rsi > 70: return "🔴 ระวังแรงเทขาย", "#FF1744", "ราคา Overbought สูงเกินไป"
        else: return "🟡 ถือรันเทรนด์", "#FFD600", "แนวโน้มยังดี ถือต่อได้"
    else:
        if rsi > 70: return "🔴 ขาย/Short", "#D50000", "เทรนด์ขาลง + ราคาดีดสูงเกินไป"
        else: return "🟠 เลี่ยงการเทรด", "#FF9100", "ราคายังอยู่ใต้เส้นค่าเฉลี่ย 200 วัน"

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
        results.append({'price': c['p'], 'type': c['t'], 'label': label, 'score': c['c']})
    return results

# --- 📰 ADVANCED NEWS FETCHING (เจาะแหล่งฟรี) ---

@st.cache_data(ttl=3600) 
def fetch_content(url, backup=""):
    try:
        # User-Agent หลอกว่าเป็นคนเล่นเว็บ
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 10
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        text = article.text
        if len(text) < 150: return backup if backup else "เนื้อหาถูกจำกัดสิทธิ์ (กรุณาอ่านที่ต้นฉบับ)"
        return text[:4000]
    except: return backup if backup else "ไม่สามารถดึงเนื้อหาได้"

def translate_text(text):
    try: return GoogleTranslator(source='auto', target='th').translate(text[:4500])
    except: return text

def get_special_news(symbol):
    """
    ฟังก์ชันเลือกแหล่งข่าวตามประเภทสินทรัพย์
    - หุ้นไทย/ทองคำไทย -> Settrade, Intergold, Sanook
    - หุ้นนอก/Crypto -> Yahoo (Bloomberg Free), Investing.com
    """
    q = symbol.replace("-THB", "").replace("-USD", "").upper()
    news_list = []
    
    # กรณี 1: หุ้นไทย หรือ ทองคำ (ดูจาก .BK หรือชื่อ)
    if ".BK" in symbol or "THB" in symbol or q in ["SET", "GOLDFUTURES"]:
        # สั่ง Google ให้ไปหาเฉพาะเว็บไทยที่อ่านฟรี
        sites = "site:settrade.com OR site:intergold.co.th OR site:kaohoon.com OR site:thunhoon.com OR site:investing.com"
        rss_url = f"https://news.google.com/rss/search?q={q}+{sites}+when:3d&hl=th&gl=TH&ceid=TH:th"
    
    # กรณี 2: หุ้นนอก / Crypto
    else:
        # Yahoo Finance + Investing.com (Inter)
        rss_url = f"https://news.google.com/rss/search?q={q}+site:finance.yahoo.com OR site:investing.com OR site:coindesk.com+when:2d&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(rss_url)
        for item in feed.entries[:5]: # เอา 5 ข่าว
            # คลีน Summary รอไว้
            soup = BeautifulSoup(item.get('description', ''), "html.parser")
            summary = soup.get_text()
            
            # ระบุแหล่งที่มาให้สวยงาม
            source_name = item.source.title if 'source' in item else 'News'
            if 'yahoo' in item.link: source_name = 'Yahoo Finance'
            elif 'investing' in item.link: source_name = 'Investing.com'
            elif 'settrade' in item.link: source_name = 'Settrade'
            elif 'intergold' in item.link: source_name = 'Intergold'

            news_list.append({
                'title': item.title,
                'link': item.link,
                'summary': summary,
                'source': source_name
            })
    except: pass
    
    return news_list

# --- 3. UI Layout ---

with st.sidebar:
    st.header("⚙️ Setting")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    show_ema = st.checkbox("Show EMA", True)

st.markdown("### 🔎 Smart News Reader (Free Sources)")
col_in, col_btn = st.columns([3.5, 1])
with col_in: symbol_input = st.text_input("Search", value="BTC-THB", label_visibility="collapsed")
with col_btn: search_pressed = st.button("GO")

symbol = symbol_input.upper().strip()

if symbol:
    with st.spinner('🤖 AI กำลังทำงาน...'):
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
        
        levels = analyze_levels(df)
        ai_text, ai_color, ai_reason = analyze_ai_signal(df)
        
        # Header & AI
        st.markdown(f"""
        <div style="background:#111; padding:20px; border-radius:15px; border-top:5px solid {color_p}; text-align:center; margin-bottom:20px;">
            <div style="font-size:1.2rem; color:#aaa;">{symbol}</div>
            <div style="font-size:3rem; font-weight:bold; color:{color_p};">{price:,.2f}</div>
            <div style="font-size:1.1rem; color:{color_p};">{change:+,.2f} ({pct:+.2f}%)</div>
            <div style="margin-top:10px; background:{ai_color}22; color:{ai_color}; padding:8px; border-radius:8px;">
                <b>🤖 AI Signal:</b> {ai_text} <br> <span style="font-size:0.9rem; opacity:0.8;">{ai_reason}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 กราฟ", "🧱 แนวรับต้าน", "📰 ข่าว (แหล่งฟรี)"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            if show_ema:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2979FF', width=1), name="EMA50"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#FF9100', width=1), name="EMA200"), row=1, col=1)
            for l in levels:
                if l['score'] >= 3:
                    c = 'green' if l['type']=='Support' else 'red'
                    fig.add_hline(y=l['price'], line_dash='solid', line_color=c, opacity=0.5, row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AA00FF')), row=2, col=1)
            fig.add_hline(y=70, line_dash='dot', line_color='red', row=2, col=1)
            fig.add_hline(y=30, line_dash='dot', line_color='green', row=2, col=1)
            fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=10), xaxis_rangeslider_visible=False, template="plotly_dark", dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with tab2:
            res = sorted([l for l in levels if l['type']=='Resistance' and l['price']>price], key=lambda x: x['price'])[:4]
            sup = sorted([l for l in levels if l['type']=='Support' and l['price']<price], key=lambda x: x['price'], reverse=True)[:4]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 🟥 ต้าน")
                for r in reversed(res): st.markdown(f"<div style='border-bottom:1px solid #333; padding:10px; display:flex; justify-content:space-between;'><span style='color:#aaa'>{r['label']}</span><span style='color:#FF5252; font-weight:bold;'>{r['price']:,.2f}</span></div>", unsafe_allow_html=True)
            with col_b:
                st.markdown("#### 🟩 รับ")
                for s in sup: st.markdown(f"<div style='border-bottom:1px solid #333; padding:10px; display:flex; justify-content:space-between;'><span style='color:#aaa'>{s['label']}</span><span style='color:#00E676; font-weight:bold;'>{s['price']:,.2f}</span></div>", unsafe_allow_html=True)

        with tab3:
            st.caption(f"ดึงข่าวจากแหล่งฟรีสำหรับ {symbol} (Yahoo/Investing/Settrade/Intergold)")
            news_items = get_special_news(symbol)
            
            if not news_items:
                st.info("ไม่พบข่าวล่าสุดจากแหล่งที่ระบุ")
            else:
                for i, item in enumerate(news_items):
                    # แปลหัวข้อ
                    if any(x in item['source'] for x in ['Settrade', 'Intergold', 'Kaohoon']):
                        # ถ้าเป็นเว็บไทยอยู่แล้ว ไม่ต้องแปล
                        title_show = item['title']
                        is_thai_source = True
                    else:
                        title_show = translate_text(item['title'])
                        is_thai_source = False
                    
                    blob = TextBlob(item['title'])
                    score = blob.sentiment.polarity
                    icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                    
                    with st.expander(f"{icon} {title_show}", expanded=(i==0)):
                        st.markdown(f"<div><span class='source-tag'>{item['source']}</span> <span style='color:#888; font-size:0.9rem;'>{item['title']}</span></div>", unsafe_allow_html=True)
                        
                        with st.spinner("กำลังโหลดเนื้อหา..."):
                            # เจาะเนื้อหา
                            body_raw = fetch_content(item['link'], backup=item['summary'])
                            
                            # ถ้าเป็นเว็บไทย แสดงเลย ถ้าเว็บนอก แปลก่อน
                            if is_thai_source:
                                body_show = body_raw
                            else:
                                body_show = translate_text(body_raw)
                        
                        st.markdown(f"<div class='news-content'>{body_show}</div>", unsafe_allow_html=True)
                        st.markdown(f"<a href='{item['link']}' target='_blank' style='display:inline-block; width:100%; text-align:center; padding:10px; background:#00897B; color:white; border-radius:8px; text-decoration:none; margin-top:10px;'>🔗 อ่านต้นฉบับ ({item['source']})</a>", unsafe_allow_html=True)
