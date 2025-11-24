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
from newspaper import Article
import nltk

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Smart Trader AI : Yahoo Edition",
    layout="wide",
    page_icon="🐂",
    initial_sidebar_state="collapsed"
)

# CSS Styling (Yahoo Style & Mobile Friendly)
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        
        /* Input & Button */
        div[data-testid="stTextInput"] input {
            font-size: 20px !important; height: 50px !important;
            border-radius: 12px !important; background-color: #1b1b1b !important;
            color: #fff !important; border: 1px solid #333 !important;
        }
        div[data-testid="stButton"] button {
            height: 50px !important; font-size: 20px !important;
            border-radius: 12px !important; width: 100% !important;
            background-color: #6001D2 !important; /* Yahoo Purple */
            color: white !important; border: none !important;
            font-weight: bold !important;
        }
        
        /* Guru Box */
        .guru-box {
            background: linear-gradient(135deg, #2c003e 0%, #000000 100%);
            padding: 20px; border-radius: 15px; margin-bottom: 20px;
            border: 1px solid #6001D2; box-shadow: 0 4px 15px rgba(96, 1, 210, 0.3);
        }
        .guru-title { font-size: 1.4rem; font-weight: bold; color: #fff; margin-bottom: 10px; display:flex; align-items:center; }
        .guru-text { font-size: 1.05rem; line-height: 1.6; color: #e0e0e0; margin-bottom: 15px; }
        .guru-stat { display: flex; justify-content: space-around; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; }
        .stat-item { text-align: center; }
        .stat-val { font-size: 1.2rem; font-weight: bold; color: #00E676; }
        .stat-lbl { font-size: 0.8rem; color: #aaa; }

        /* News Content */
        .news-content { 
            font-size: 1rem; line-height: 1.7; color: #ddd; 
            text-align: justify; background: #1a1a1a; padding: 15px; border-radius: 10px;
        }
        
        /* Tabs */
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
        return df, ticker
    except: return pd.DataFrame(), None

# --- 🧐 GURU ANALYSIS (วิเคราะห์พื้นฐาน) ---
def get_guru_analysis(ticker, symbol, current_price):
    """สร้างบทวิเคราะห์จากข้อมูลพื้นฐาน (Wall Street Data)"""
    try:
        info = ticker.info
        
        # ดึงข้อมูลสำคัญ
        target_price = info.get('targetMeanPrice', 0)
        recommendation = info.get('recommendationKey', 'none').replace('_', ' ').upper()
        pe_ratio = info.get('trailingPE', 0)
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Unknown')
        
        # แปลง Market Cap เป็นข้อความ
        if market_cap > 1e12: mcap_str = f"{market_cap/1e12:.2f} Trillion"
        elif market_cap > 1e9: mcap_str = f"{market_cap/1e9:.2f} Billion"
        else: mcap_str = f"{market_cap/1e6:.2f} Million"

        # สร้างบทวิเคราะห์ (Narrative Generation)
        analysis_text = f"หุ้น **{symbol}** อยู่ในกลุ่มอุตสาหกรรม **{sector}** โดยมีมูลค่าตลาดประมาณ **{mcap_str}** "
        
        # 1. วิเคราะห์ราคาเป้าหมาย
        if target_price and target_price > 0:
            upside = ((target_price - current_price) / current_price) * 100
            if upside > 10:
                analysis_text += f"นักวิเคราะห์ Wall Street มองว่าราคายังมีโอกาสเติบโต (Upside) อีกประมาณ **{upside:.1f}%** ไปที่ราคาเป้าหมาย **{target_price:,.2f}** "
            elif upside < -10:
                analysis_text += f"ราคาปัจจุบันสูงกว่าราคาเป้าหมายเฉลี่ยที่ **{target_price:,.2f}** (Overvalued) ควรระมัดระวัง "
            else:
                analysis_text += f"ราคาปัจจุบันใกล้เคียงกับราคาประเมินที่ **{target_price:,.2f}** (Fair Value) "
        else:
            analysis_text += "ไม่มีข้อมูลราคาเป้าหมายจากนักวิเคราะห์ "

        # 2. วิเคราะห์คำแนะนำ
        rec_map = {
            'STRONG BUY': "แนะนำ: 🟢 'ซื้อทันที' (Strong Buy)",
            'BUY': "แนะนำ: 🟢 'ซื้อ' (Buy)",
            'HOLD': "แนะนำ: 🟡 'ถือ' (Hold)",
            'UNDERPERFORM': "แนะนำ: 🔴 'ทำผลงานต่ำกว่าตลาด'",
            'SELL': "แนะนำ: 🔴 'ขาย' (Sell)"
        }
        rec_text = rec_map.get(recommendation, f"สถานะ: {recommendation}")
        
        # 3. วิเคราะห์ P/E (คร่าวๆ)
        if pe_ratio > 0:
            if pe_ratio < 15: analysis_text += "อัตราส่วน P/E อยู่ในเกณฑ์ต่ำ (Value Stock) "
            elif pe_ratio > 50: analysis_text += "อัตราส่วน P/E ค่อนข้างสูง (Growth Stock/High Expectation) "
            
        return analysis_text, rec_text, target_price, pe_ratio
        
    except Exception as e:
        return "ไม่สามารถดึงข้อมูลเชิงลึกได้ (อาจเป็น Crypto หรือ ETF)", "N/A", 0, 0

def analyze_technical(df):
    close = df['Close'].iloc[-1]
    ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
    ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    if close > ema200:
        trend = "Uptrend"
        status = "🟢 แข็งแกร่ง" if rsi < 50 else "🟡 พักตัว" if rsi < 70 else "🔴 ระวังแรงขาย"
    else:
        trend = "Downtrend"
        status = "🔴 ขาลง" if rsi > 50 else "🟡 รีบาวด์"
        
    return trend, status, rsi

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

# --- NEWS FUNCTIONS (Yahoo + Fallback) ---
@st.cache_data(ttl=3600) 
def fetch_full_news_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if len(text) < 100: return None # เนื้อหาน้อยไป
        return text[:4000]
    except: return None

def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='th').translate(text)
    except: return text

def get_yahoo_news(ticker, symbol):
    news_data = []
    try:
        # 1. Try Yahoo Finance First
        yf_news = ticker.news
        if yf_news:
            for item in yf_news[:3]: # เอา 3 ข่าว
                news_data.append({
                    'title': item['title'],
                    'link': item['link'],
                    'pubDate': item.get('providerPublishTime', 0),
                    'source': 'Yahoo Finance'
                })
        
        # 2. If empty (often happens with Crypto), use Google RSS Fallback
        if not news_data:
            q = symbol.replace("-THB", "").replace("-USD", "")
            url = f"https://news.google.com/rss/search?q={q}+when:2d&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for item in feed.entries[:3]:
                news_data.append({
                    'title': item.title,
                    'link': item.link,
                    'pubDate': item.get('published', ''),
                    'source': 'Google News'
                })
                
    except Exception as e: print(e)
    return news_data

# --- 3. UI Layout ---

with st.sidebar:
    st.header("⚙️ Setting")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    show_ema = st.checkbox("Show EMA", True)

st.markdown("### 🔎 Wall Street Analyst & News")
col_in, col_btn = st.columns([3.5, 1])
with col_in: symbol_input = st.text_input("Search", value="NVDA", label_visibility="collapsed")
with col_btn: search_pressed = st.button("GO")

symbol = symbol_input.upper().strip()

if symbol:
    with st.spinner('🐂 กำลังเรียกข้อมูลจาก Wall Street...'):
        df, ticker = get_data(symbol, period, interval)
    
    if df.empty:
        st.warning(f"ไม่พบข้อมูล '{symbol}'")
    else:
        # Tech Indicators
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        
        price = df['Close'].iloc[-1]
        change = price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        color_p = "#00E676" if change >= 0 else "#FF1744"
        
        # Analyses
        levels = analyze_levels(df)
        tech_trend, tech_status, rsi_val = analyze_technical(df)
        
        # Guru Analysis
        guru_text, guru_rec, target_price, pe_ratio = get_guru_analysis(ticker, symbol, price)
        
        # --- UI: Price Header ---
        st.markdown(f"""
        <div style="background:#111; padding:20px; border-radius:15px; border-top:5px solid {color_p}; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.5); margin-bottom:20px;">
            <div style="font-size:1.2rem; color:#aaa;">{symbol}</div>
            <div style="font-size:3rem; font-weight:bold; line-height:1.2; color:{color_p};">{price:,.2f}</div>
            <div style="font-size:1.1rem; color:{color_p}; margin-bottom:10px;">{change:+,.2f} ({pct:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- UI: 🧐 GURU INSIGHT BOX ---
        st.markdown(f"""
        <div class="guru-box">
            <div class="guru-title">🧐 บทวิเคราะห์จากกูรู (Guru Insight)</div>
            <div class="guru-text">
                {guru_text}
            </div>
            <div class="guru-stat">
                <div class="stat-item">
                    <div class="stat-val">{guru_rec}</div>
                    <div class="stat-lbl">ความเห็นนักวิเคราะห์</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">{target_price:,.2f}</div>
                    <div class="stat-lbl">ราคาเป้าหมาย (Target)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">{pe_ratio:.2f}</div>
                    <div class="stat-lbl">P/E Ratio</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 กราฟเทคนิค", "🧱 แนวรับต้าน", "📰 ข่าว Yahoo แปลไทย"])
        
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
                st.markdown("#### 🟥 ต้าน (Sell)")
                for r in reversed(res): st.markdown(f"<div style='border-bottom:1px solid #333; padding:10px; display:flex; justify-content:space-between;'><span style='color:#aaa'>{r['label']}</span><span style='color:#FF5252; font-weight:bold;'>{r['price']:,.2f}</span></div>", unsafe_allow_html=True)
            with col_b:
                st.markdown("#### 🟩 รับ (Buy)")
                for s in sup: st.markdown(f"<div style='border-bottom:1px solid #333; padding:10px; display:flex; justify-content:space-between;'><span style='color:#aaa'>{s['label']}</span><span style='color:#00E676; font-weight:bold;'>{s['price']:,.2f}</span></div>", unsafe_allow_html=True)

        with tab3:
            st.caption("ดึงข่าวจาก Yahoo Finance / Google News และแปลไทย...")
            news_items = get_yahoo_news(ticker, symbol)
            
            if not news_items:
                st.info("ไม่พบข่าวล่าสุด")
            else:
                for i, item in enumerate(news_items):
                    # Translate Title
                    title_th = translate_text(item['title'])
                    
                    # Sentiment Icon
                    blob = TextBlob(item['title'])
                    score = blob.sentiment.polarity
                    icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                    
                    # Expandable News
                    with st.expander(f"{icon} {title_th}", expanded=(i==0)):
                        st.markdown(f"<div style='color:#888; font-size:0.9rem; margin-bottom:10px;'>Source: {item['source']} | {item['title']}</div>", unsafe_allow_html=True)
                        
                        # Fetch & Translate Body
                        with st.spinner("กำลังเจาะลึกเนื้อหา..."):
                            body_en = fetch_full_news_content(item['link'])
                            if body_en:
                                body_th = translate_text(body_en)
                                st.markdown(f"<div class='news-content'>{body_th}</div>", unsafe_allow_html=True)
                            else:
                                st.warning("ไม่สามารถดึงเนื้อหาฉบับเต็มได้ (ติด Paywall หรือ Format ไม่รองรับ)")
                        
                        st.markdown(f"<a href='{item['link']}' target='_blank' style='display:inline-block; width:100%; text-align:center; padding:10px; background:#6001D2; color:white; border-radius:8px; text-decoration:none; margin-top:10px;'>🔗 อ่านต้นฉบับ</a>", unsafe_allow_html=True)
