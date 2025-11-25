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
import nltk
import math
import urllib.parse

# Config NLTK
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

# --- 1. Setup & Design ---
st.set_page_config(
    page_title="Smart Trader AI : Pro Max",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# --- Modern CSS (Neon & Glassmorphism) ---
st.markdown("""
    <style>
        /* Main Theme */
        body { background-color: #050505; color: #fff; }
        .stApp { background: radial-gradient(circle at 10% 20%, #000000 0%, #1a1a1a 90%); }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        }

        /* --- Sentiment Analysis Cards (High Contrast) --- */
        .sentiment-card {
            padding: 20px; border-radius: 15px; margin-bottom: 20px;
            background: #111; /* Darker background for contrast */
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            transition: transform 0.2s;
        }
        .sentiment-card:hover { transform: scale(1.01); }
        
        /* Positive */
        .card-pos { border-left: 6px solid #00E676; border-top: 1px solid rgba(0, 230, 118, 0.3); border-right: 1px solid rgba(0, 230, 118, 0.3); border-bottom: 1px solid rgba(0, 230, 118, 0.3); }
        .badge-pos { background: #00E676; color: #000; padding: 6px 12px; border-radius: 20px; font-weight: 900; box-shadow: 0 0 10px rgba(0, 230, 118, 0.6); }
        
        /* Negative */
        .card-neg { border-left: 6px solid #FF1744; border-top: 1px solid rgba(255, 23, 68, 0.3); border-right: 1px solid rgba(255, 23, 68, 0.3); border-bottom: 1px solid rgba(255, 23, 68, 0.3); }
        .badge-neg { background: #FF1744; color: #fff; padding: 6px 12px; border-radius: 20px; font-weight: 900; box-shadow: 0 0 10px rgba(255, 23, 68, 0.6); }
        
        /* Neutral */
        .card-neu { border-left: 6px solid #2979FF; border-top: 1px solid rgba(41, 121, 255, 0.3); border-right: 1px solid rgba(41, 121, 255, 0.3); border-bottom: 1px solid rgba(41, 121, 255, 0.3); }
        .badge-neu { background: #2979FF; color: #fff; padding: 6px 12px; border-radius: 20px; font-weight: 900; box-shadow: 0 0 10px rgba(41, 121, 255, 0.6); }

        /* Trade Setup Cards */
        .setup-box {
            background: #1e1e1e; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #333;
        }
        .setup-label { color: #888; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
        .setup-val { font-size: 1.8rem; font-weight: bold; margin-top: 5px; }

        /* Inputs & Buttons */
        div[data-testid="stTextInput"] input {
            border-radius: 12px !important; background-color: rgba(255,255,255,0.05) !important;
            color: #fff !important; border: 1px solid rgba(255,255,255,0.2) !important;
        }
        div[data-testid="stButton"] button { border-radius: 12px !important; font-weight: 600 !important; }

        /* S/R Tags */
        .sr-tag { padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; display: inline-block; }
        .sr-strong { background: rgba(0, 230, 118, 0.2); color: #00E676; border: 1px solid #00E676; }
        .sr-psy { background: rgba(41, 98, 255, 0.2); color: #2962FF; border: 1px solid #2962FF; }
        .sr-weak { background: rgba(255, 255, 255, 0.1); color: #aaa; border: 1px solid #555; }
        
        .section-header {
            font-size: 1.5rem; font-weight: bold; background: linear-gradient(90deg, #fff, #888);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. Data & Analysis Functions ---

def get_data(symbol, period, interval):
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

def identify_levels(df, window=5, tolerance=0.02):
    levels = []
    try:
        for i in range(window, len(df) - window):
            is_support = df['Low'][i] == df['Low'][i-window:i+window+1].min()
            is_resistance = df['High'][i] == df['High'][i-window:i+window+1].max()
            if is_support: levels.append({'price': df['Low'][i], 'type': 'Support', 'touches': 1})
            elif is_resistance: levels.append({'price': df['High'][i], 'type': 'Resistance', 'touches': 1})
        
        levels.sort(key=lambda x: x['price'])
        merged = []
        if not levels: return []
        curr = levels[0]
        for next_lvl in levels[1:]:
            if abs(next_lvl['price'] - curr['price']) / curr['price'] < tolerance:
                curr['price'] = (curr['price'] * curr['touches'] + next_lvl['price'] * next_lvl['touches']) / (curr['touches'] + next_lvl['touches'])
                curr['touches'] += next_lvl['touches']
            else:
                merged.append(curr)
                curr = next_lvl
        merged.append(curr)
        
        final = []
        current_price = df['Close'].iloc[-1]
        for lvl in merged:
            price = lvl['price']
            is_psy = False
            if price > 100: is_psy = (abs(price % 100) < 1) or (abs(price % 1000) < 10)
            
            if lvl['touches'] >= 3 or (lvl['touches'] >= 2 and is_psy): strength, desc = "Strong", "🔥🔥 แข็งแกร่ง (Strong Zone)"
            elif is_psy: strength, desc = "Psychological", "🧠 จิตวิทยา (Round Number)"
            else: strength, desc = "Minor", "☁️ เบาบาง (Minor)"
            
            if abs(price - current_price)/current_price > 0.5 and strength == "Minor": continue
            lvl['strength'], lvl['desc'] = strength, desc
            final.append(lvl)
        return final
    except: return []

# --- Trade Setup Logic ---
def calculate_trade_setup(df):
    try:
        close = df['Close'].iloc[-1]
        ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
        
        # ATR Calculation (14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        trend = "Sideways"
        signal = "WAIT"
        color = "#888"
        
        # Logic
        if close > ema50 and ema50 > ema200:
            trend = "Uptrend (ขาขึ้น)"
            signal = "LONG / BUY"
            color = "#00E676"
            entry = close
            stop_loss = close - (1.5 * atr)
            take_profit = close + (2.5 * atr)
        elif close < ema50 and ema50 < ema200:
            trend = "Downtrend (ขาลง)"
            signal = "SHORT / SELL"
            color = "#FF1744"
            entry = close
            stop_loss = close + (1.5 * atr)
            take_profit = close - (2.5 * atr)
        else:
            entry = close
            stop_loss = close - atr
            take_profit = close + atr
            
        return {
            'trend': trend, 'signal': signal, 'color': color,
            'entry': entry, 'sl': stop_loss, 'tp': take_profit, 'atr': atr
        }
    except: return None

# --- News & Sentiment ---
@st.cache_data(ttl=1800)
def get_bloomberg_news(symbol):
    news_list = []
    clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
    try:
        raw_query = f"site:bloomberg.com {clean_sym} market OR {clean_sym} price analysis"
        encoded_query = urllib.parse.quote(raw_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for item in feed.entries[:6]:
            news_list.append({'title': item.title, 'link': item.link, 'summary': item.description, 'source': 'Bloomberg'})
    except: pass
    
    if len(news_list) < 2: # Fallback
        try:
            q = urllib.parse.quote(f"{clean_sym} finance news")
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            for item in feed.entries[:4]:
                 news_list.append({'title': item.title, 'link': item.link, 'summary': item.description, 'source': 'News'})
        except: pass
    return news_list

def analyze_sentiment_advanced(text, title):
    try:
        translator = GoogleTranslator(source='auto', target='th')
        title_th = translator.translate(title)
        soup = BeautifulSoup(text, "html.parser")
        summary_th = translator.translate(soup.get_text()[:300] + "...")
        
        blob = TextBlob(text + " " + title)
        polarity = blob.sentiment.polarity
        
        # Keyword Boost
        txt_low = (text + title).lower()
        if any(w in txt_low for w in ['surge','soar','jump','record','bull','profit','approval','etf']): polarity += 0.25
        if any(w in txt_low for w in ['crash','plunge','drop','bear','loss','ban','lawsuit','hack']): polarity -= 0.25
        
        if polarity > 0.1:
            cat, css_cls, badge_cls, icon = "Positive", "card-pos", "badge-pos", "🚀"
            impact = "ปัจจัยบวก: มีแนวโน้มสนับสนุนราคาให้ปรับตัวขึ้น"
        elif polarity < -0.1:
            cat, css_cls, badge_cls, icon = "Negative", "card-neg", "badge-neg", "🔻"
            impact = "ปัจจัยลบ: ระวังแรงเทขายหรือข่าวร้ายกดดันตลาด"
        else:
            cat, css_cls, badge_cls, icon = "Neutral", "card-neu", "badge-neu", "⚖️"
            impact = "ทรงตัว: ตลาดยังรอความชัดเจน (Wait & See)"
            
        return {'title': title_th, 'summary': summary_th, 'cat': cat, 'css': css_cls, 'badge': badge_cls, 'icon': icon, 'impact': impact, 'link': item['link'], 'source': item['source']}
    except: return None

# --- 3. Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=60)
    st.title("Smart Trader AI")
    st.caption("Pro Max Edition")
    st.markdown("### 🏆 Top Assets")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("BTC-USD"): set_symbol("BTC-USD")
        if st.button("ETH-USD"): set_symbol("ETH-USD")
        if st.button("Gold"): set_symbol("GC=F")
    with c2:
        if st.button("🇹🇭 BTC-THB"): set_symbol("BTC-THB")
        if st.button("🇹🇭 ETH-THB"): set_symbol("ETH-THB")
        if st.button("Oil"): set_symbol("CL=F")
    st.markdown("---")
    st.subheader("🛠 Configuration")
    chart_type = st.selectbox("รูปแบบกราฟ", ["Candlestick (Standard)", "Heikin Ashi (Trend)"], index=0)
    period = st.select_slider("ย้อนหลัง", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], value="1y")
    interval = st.selectbox("Timeframe", ["1d", "1wk", "1h"], index=0)

# --- 4. Main Interface ---
c_search, c_act = st.columns([3, 1])
with c_search: sym_input = st.text_input("🔍 Search Symbol", value=st.session_state.symbol)
with c_act:
    st.write("")
    st.write("")
    if st.button("Analyze Now ⚡", use_container_width=True):
        st.session_state.symbol = sym_input
        st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner(f'🤖 AI กำลังประมวลผลกลยุทธ์สำหรับ {symbol}...'):
        df, ticker = get_data(symbol, period, interval)
    
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล {symbol}")
    else:
        # Indicators
        curr_price = df['Close'].iloc[-1]
        change = curr_price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        sr_levels = identify_levels(df)
        setup = calculate_trade_setup(df)

        # Header
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 4px solid {'#00E676' if change>=0 else '#FF1744'};">
                <h1 style="margin:0; font-size: 3rem;">{symbol}</h1>
                <h2 style="margin:0; font-size: 4rem; color: {'#00E676' if change>=0 else '#FF1744'};">{curr_price:,.2f}</h2>
                <p style="font-size: 1.5rem; color: #aaa;">{change:+,.2f} ({pct:+.2f}%)</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Tabs (REPLACED Tab 3)
        t1, t2, t3, t4, t5 = st.tabs(["📈 Smart Chart", "🛡️ S/R Levels", "🎯 Smart Trade Setup", "📊 Fundamentals", "🧠 AI Sentiment"])
        
        with t1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            if "Heikin" in chart_type:
                plot_df = calculate_heikin_ashi(df)
                o, h, l, c = plot_df['HA_Open'], plot_df['HA_High'], plot_df['HA_Low'], plot_df['HA_Close']
                c_inc, c_dec = '#00F2B6', '#FF3B30'
            else:
                plot_df = df
                o, h, l, c = plot_df['Open'], plot_df['High'], plot_df['Low'], plot_df['Close']
                c_inc, c_dec = '#26A69A', '#EF5350'
            fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=l, close=c, name='Price', increasing_line_color=c_inc, decreasing_line_color=c_dec), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2979FF', width=1), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#FF9100', width=1), name='EMA 200'), row=1, col=1)
            for lvl in sr_levels:
                if abs(lvl['price'] - curr_price)/curr_price < 0.2:
                    c_line = 'rgba(0,230,118,0.5)' if lvl['type']=='Support' else 'rgba(255,23,68,0.5)'
                    fig.add_hline(y=lvl['price'], line_dash='dash', line_color=c_line, row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AB47BC', width=1.5), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.markdown("<div class='section-header'>🛡️ Support & Resistance (Advanced)</div>", unsafe_allow_html=True)
            c_res, c_sup = st.columns(2)
            res = sorted([l for l in sr_levels if l['price'] > curr_price], key=lambda x:x['price'])[:5]
            sup = sorted([l for l in sr_levels if l['price'] < curr_price], key=lambda x:x['price'], reverse=True)[:5]
            with c_res:
                st.error("🟥 RESISTANCE (แนวต้าน)")
                for r in reversed(res):
                    tag = "sr-strong" if r['strength']=="Strong" else "sr-psy" if r['strength']=="Psychological" else "sr-weak"
                    st.markdown(f"<div style='border-bottom:1px solid #333; padding:10px; display:flex; justify-content:space-between;'><span style='font-family:monospace; font-size:1.1rem;'>{r['price']:,.2f}</span><span class='sr-tag {tag}'>{r['desc']}</span></div>", unsafe_allow_html=True)
            with c_sup:
                st.success("🟩 SUPPORT (แนวรับ)")
                for s in sup:
                    tag = "sr-strong" if s['strength']=="Strong" else "sr-psy" if s['strength']=="Psychological" else "sr-weak"
                    st.markdown(f"<div style='border-bottom:1px solid #333; padding:10px; display:flex; justify-content:space-between;'><span style='font-family:monospace; font-size:1.1rem;'>{s['price']:,.2f}</span><span class='sr-tag {tag}'>{s['desc']}</span></div>", unsafe_allow_html=True)

        # --- TAB 3: SMART TRADE SETUP (NEW) ---
        with t3:
            st.markdown("<div class='section-header'>🎯 Smart Trade Setup (AI Plan)</div>", unsafe_allow_html=True)
            if setup:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 10px solid {setup['color']};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:2rem; font-weight:bold; color:{setup['color']};">{setup['signal']}</div>
                        <div style="font-size:1.2rem; color:#aaa;">Trend: {setup['trend']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c_en, c_sl, c_tp = st.columns(3)
                with c_en:
                    st.markdown(f"<div class='setup-box'><div class='setup-label'>🔵 ENTRY PRICE</div><div class='setup-val' style='color:#2979FF'>{setup['entry']:,.2f}</div><div style='font-size:0.8rem; color:#666;'>Current Market Price</div></div>", unsafe_allow_html=True)
                with c_sl:
                    st.markdown(f"<div class='setup-box'><div class='setup-label'>🔴 STOP LOSS</div><div class='setup-val' style='color:#FF1744'>{setup['sl']:,.2f}</div><div style='font-size:0.8rem; color:#666;'>Risk Based on ATR ({setup['atr']:,.2f})</div></div>", unsafe_allow_html=True)
                with c_tp:
                    st.markdown(f"<div class='setup-box'><div class='setup-label'>🟢 TAKE PROFIT</div><div class='setup-val' style='color:#00E676'>{setup['tp']:,.2f}</div><div style='font-size:0.8rem; color:#666;'>Reward Ratio 1:1.5+</div></div>", unsafe_allow_html=True)
                
                st.info("⚠️ **คำเตือน:** แผนการเทรดนี้คำนวณจากความผันผวน (ATR) และเทรนด์ (EMA) โดยอัตโนมัติ ไม่ใช่คำแนะนำทางการเงิน ผู้ลงทุนควรบริหารความเสี่ยงด้วยตนเอง")
            else:
                st.warning("ข้อมูลไม่เพียงพอสำหรับการคำนวณแผนการเทรด")

        with t4:
            info = ticker.info
            st.markdown("<div class='section-header'>📊 Fundamentals</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Market Cap", f"{info.get('marketCap',0):,}")
            with c2: st.metric("PE Ratio", f"{info.get('trailingPE',0):.2f}")
            with c3: st.metric("52W High", f"{info.get('fiftyTwoWeekHigh',0):,.2f}")
            with c4: st.metric("52W Low", f"{info.get('fiftyTwoWeekLow',0):,.2f}")

        # --- TAB 5: SENTIMENT (IMPROVED UI) ---
        with t5:
            st.markdown("<div class='section-header'>🧠 AI Sentiment Analysis (Thai)</div>", unsafe_allow_html=True)
            
            raw_news = get_bloomberg_news(symbol)
            if not raw_news:
                st.warning("ไม่พบข่าวในขณะนี้")
            else:
                processed = []
                pos, neg, neu = 0, 0, 0
                
                bar = st.progress(0)
                for i, item in enumerate(raw_news):
                    bar.progress((i+1)/len(raw_news))
                    res = analyze_sentiment_advanced(item['summary'], item['title'])
                    if res:
                        res['link'] = item['link']
                        res['source'] = item['source']
                        processed.append(res)
                        if res['cat']=='Positive': pos+=1
                        elif res['cat']=='Negative': neg+=1
                        else: neu+=1
                bar.empty()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Positive News", f"{pos}", delta="Bullish", delta_color="normal")
                c2.metric("Negative News", f"{neg}", delta="-Bearish", delta_color="inverse")
                c3.metric("Neutral News", f"{neu}", delta="Wait", delta_color="off")
                
                st.markdown("---")
                
                for p in processed:
                    st.markdown(f"""
                    <div class="sentiment-card {p['css']}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                            <span class="{p['badge']}">{p['icon']} {p['cat']}</span>
                            <span style="color:#666; font-size:0.8rem;">Source: {p['source']}</span>
                        </div>
                        <div style="font-size:1.1rem; font-weight:bold; color:#fff; margin-bottom:10px;">{p['title']}</div>
                        <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px; color:#ccc; font-size:0.9rem; margin-bottom:10px;">
                            {p['summary']}
                        </div>
                        <div style="font-weight:bold; margin-top:5px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.1);">
                            💥 {p['impact']}
                        </div>
                        <div style="text-align:right; margin-top:5px;">
                             <a href="{p['link']}" target="_blank" style="color:#aaa; font-size:0.8rem; text-decoration:none;">🔗 อ่านข่าวต้นฉบับ</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
