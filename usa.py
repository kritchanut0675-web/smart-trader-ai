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
import math

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

# --- Modern CSS (Glassmorphism UI) ---
st.markdown("""
    <style>
        /* Main Theme */
        body { background-color: #0E1117; }
        .stApp { background: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(20, 20, 20) 90.2%); }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        /* Inputs & Buttons */
        div[data-testid="stTextInput"] input {
            border-radius: 12px !important; background-color: rgba(255,255,255,0.05) !important;
            color: #fff !important; border: 1px solid rgba(255,255,255,0.2) !important;
        }
        div[data-testid="stButton"] button {
            border-radius: 12px !important; font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,200,83,0.3);
        }

        /* S/R Tags */
        .sr-tag {
            padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; display: inline-block;
        }
        .sr-strong { background: rgba(0, 230, 118, 0.2); color: #00E676; border: 1px solid #00E676; }
        .sr-psy { background: rgba(41, 98, 255, 0.2); color: #2962FF; border: 1px solid #2962FF; }
        .sr-weak { background: rgba(255, 255, 255, 0.1); color: #aaa; border: 1px solid #555; }

        /* News */
        .news-item {
            border-left: 4px solid #2962FF; background: rgba(255,255,255,0.03);
            padding: 15px; margin-bottom: 10px; border-radius: 0 10px 10px 0;
        }
        
        /* Custom Headers */
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

# --- Advanced S/R Algorithm ---
def identify_levels(df, window=5, tolerance=0.02):
    """
    Identify S/R levels based on fractals, count touches, and classify strength.
    """
    levels = []
    
    # 1. Find Pivot Highs/Lows
    for i in range(window, len(df) - window):
        is_support = df['Low'][i] == df['Low'][i-window:i+window+1].min()
        is_resistance = df['High'][i] == df['High'][i-window:i+window+1].max()
        
        if is_support:
            levels.append({'price': df['Low'][i], 'type': 'Support', 'score': 1, 'touches': 1})
        elif is_resistance:
            levels.append({'price': df['High'][i], 'type': 'Resistance', 'score': 1, 'touches': 1})
            
    # 2. Clustering (Merge close levels)
    levels.sort(key=lambda x: x['price'])
    merged_levels = []
    
    if not levels: return []
    
    curr = levels[0]
    for next_lvl in levels[1:]:
        # If prices are within tolerance %
        if abs(next_lvl['price'] - curr['price']) / curr['price'] < tolerance:
            # Merge: Weighted average based on touches
            total_touches = curr['touches'] + next_lvl['touches']
            new_price = ((curr['price'] * curr['touches']) + (next_lvl['price'] * next_lvl['touches'])) / total_touches
            
            curr['price'] = new_price
            curr['touches'] = total_touches
            # If types conflict, the one with more recent/more touches wins, or keep current
        else:
            merged_levels.append(curr)
            curr = next_lvl
    merged_levels.append(curr)
    
    # 3. Classification Logic
    final_levels = []
    current_price = df['Close'].iloc[-1]
    
    for lvl in merged_levels:
        price = lvl['price']
        touches = lvl['touches']
        
        # Determine if Psychological (ends in 00, 000, 50 etc based on magnitude)
        magnitude = 10 ** int(math.log10(price))
        is_psy = False
        if price > 1000:
            if abs(price % 1000) < 10 or abs(price % 1000) > 990: is_psy = True # e.g., 65000
        elif price > 100:
            if abs(price % 100) < 1 or abs(price % 100) > 99: is_psy = True
            
        # Determine Strength
        strength = "Weak"
        desc = "แนวรับ/ต้านย่อย"
        
        if touches >= 3 or (touches >= 2 and is_psy):
            strength = "Strong"
            desc = "🔥🔥 แข็งแกร่ง (Strong Zone)"
        elif is_psy:
            strength = "Psychological"
            desc = "🧠 จิตวิทยา (Round Number)"
        else:
            strength = "Minor"
            desc = "☁️ เบาบาง (Minor)"

        # Filter noise: Ignore very weak levels far from price
        if abs(price - current_price) / current_price > 0.5 and strength == "Minor":
            continue

        lvl['strength'] = strength
        lvl['desc'] = desc
        final_levels.append(lvl)
        
    return final_levels

# --- Bloomberg News via Google RSS ---
@st.cache_data(ttl=1800)
def get_bloomberg_news(symbol):
    # 1. Build query tailored for Bloomberg results
    clean_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
    query = f"site:bloomberg.com {clean_sym} market OR {clean_sym} price analysis"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    news_list = []
    try:
        feed = feedparser.parse(rss_url)
        for item in feed.entries[:6]:
            # Google RSS links need decoding or just use as is (redirects)
            news_list.append({
                'title': item.title,
                'link': item.link,
                'pubDate': item.published,
                'source': 'Bloomberg (via Google)',
                'summary': item.description
            })
    except Exception as e:
        print(e)
        
    # Fallback: If Bloomberg specific yields nothing, get general high-quality finance news
    if len(news_list) < 2:
        query_bk = f"{clean_sym} finance news"
        rss_bk = f"https://news.google.com/rss/search?q={query_bk}&hl=en-US&gl=US&ceid=US:en"
        feed_bk = feedparser.parse(rss_bk)
        for item in feed_bk.entries[:4]:
             news_list.append({
                'title': item.title,
                'link': item.link,
                'pubDate': item.published,
                'source': item.source.title if 'source' in item else 'News',
                'summary': item.description
            })
            
    return news_list

def translate_and_summarize(text, title):
    # Due to limits, we just translate the title and a short summary
    try:
        translator = GoogleTranslator(source='auto', target='th')
        th_title = translator.translate(title)
        
        # Remove HTML tags from summary
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text()[:300] + "..."
        th_sum = translator.translate(clean_text)
        
        return th_title, th_sum
    except:
        return title, "ไม่สามารถแปลเนื้อหาได้"

# --- 3. Sidebar & Controls ---
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
    
    st.markdown("---")
    st.info("💡 **Tip:** 'Strong Zone' คือแนวรับที่มีนัยยะสำคัญ เหมาะแก่การพิจารณาเข้าซื้อหากมีสัญญาณกลับตัว")

# --- 4. Main Interface ---
c_search, c_act = st.columns([3, 1])
with c_search:
    sym_input = st.text_input("🔍 Search Symbol (e.g. AAPL, TSLA, DOGE-USD)", value=st.session_state.symbol)
with c_act:
    st.write("")
    st.write("")
    if st.button("Analyze Now ⚡", use_container_width=True):
        st.session_state.symbol = sym_input
        st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    # Fetch Data
    with st.spinner(f'🤖 AI กำลังวิเคราะห์ข้อมูล {symbol} จาก Bloomberg และ Market Data...'):
        df, ticker = get_data(symbol, period, interval)
    
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูล {symbol} โปรดตรวจสอบชื่อย่อ")
    else:
        # Pre-calc
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = current_price - prev_price
        pct_change = (change / prev_price) * 100
        
        # Calculate Indicators
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # S/R Analysis
        sr_levels = identify_levels(df)
        
        # Display Header
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 4px solid {'#00E676' if change>=0 else '#FF1744'};">
                <h1 style="margin:0; font-size: 3rem;">{symbol}</h1>
                <h2 style="margin:0; font-size: 4rem; color: {'#00E676' if change>=0 else '#FF1744'};">
                    {current_price:,.2f}
                </h2>
                <p style="font-size: 1.5rem; color: {'#aaa'};">
                    {change:+,.2f} ({pct_change:+.2f}%)
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Tabs
        t1, t2, t3, t4 = st.tabs(["📈 Smart Chart", "🛡️ Support/Resistance PRO", "📰 Bloomberg News (AI)", "📊 Fundamentals"])
        
        with t1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Chart Type Logic
            if "Heikin" in chart_type:
                plot_df = calculate_heikin_ashi(df)
                o, h, l, c = plot_df['HA_Open'], plot_df['HA_High'], plot_df['HA_Low'], plot_df['HA_Close']
                c_inc, c_dec = '#00F2B6', '#FF3B30'
            else:
                plot_df = df
                o, h, l, c = plot_df['Open'], plot_df['High'], plot_df['Low'], plot_df['Close']
                c_inc, c_dec = '#26A69A', '#EF5350'
                
            fig.add_trace(go.Candlestick(x=df.index, open=o, high=h, low=l, close=c, name='Price',
                                         increasing_line_color=c_inc, decreasing_line_color=c_dec), row=1, col=1)
            
            # EMAs
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2979FF', width=1), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#FF9100', width=1), name='EMA 200'), row=1, col=1)
            
            # Plot Significant S/R on Chart
            for lvl in sr_levels:
                if abs(lvl['price'] - current_price) / current_price < 0.2: # Show only nearby levels
                    color = 'rgba(0, 230, 118, 0.5)' if lvl['type'] == 'Support' else 'rgba(255, 23, 68, 0.5)'
                    width = 2 if lvl['strength'] == 'Strong' else 1
                    dash = 'solid' if lvl['strength'] == 'Strong' else 'dash'
                    
                    fig.add_hline(y=lvl['price'], line_dash=dash, line_color=color, line_width=width, row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AB47BC', width=1.5), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dot', row=2, col=1)
            
            fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            st.markdown("<div class='section-header'>🛡️ Advanced Support & Resistance Analysis</div>", unsafe_allow_html=True)
            
            col_res, col_sup = st.columns(2)
            
            # Filter levels
            res_levels = sorted([l for l in sr_levels if l['price'] > current_price], key=lambda x: x['price'])[:5]
            sup_levels = sorted([l for l in sr_levels if l['price'] < current_price], key=lambda x: x['price'], reverse=True)[:5]
            
            with col_res:
                st.error("🟥 แนวต้าน (Resistance)")
                for r in reversed(res_levels):
                    tag_class = "sr-strong" if r['strength']=="Strong" else "sr-psy" if r['strength']=="Psychological" else "sr-weak"
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; padding:10px;">
                        <span style="font-family:monospace; font-size:1.2rem;">{r['price']:,.2f}</span>
                        <span class="sr-tag {tag_class}">{r['desc']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_sup:
                st.success("🟩 แนวรับ (Support)")
                for s in sup_levels:
                    tag_class = "sr-strong" if s['strength']=="Strong" else "sr-psy" if s['strength']=="Psychological" else "sr-weak"
                    buy_msg = "🛒 จุดเข้าซื้อได้" if s['strength'] == "Strong" else ""
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; padding:10px;">
                        <span style="font-family:monospace; font-size:1.2rem;">{s['price']:,.2f}</span>
                        <div>
                            <span style="color:#00E676; font-size:0.8rem; margin-right:5px;">{buy_msg}</span>
                            <span class="sr-tag {tag_class}">{s['desc']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with t3:
            st.markdown("<div class='section-header'>📰 Bloomberg & Global News (AI Translated)</div>", unsafe_allow_html=True)
            news = get_bloomberg_news(symbol)
            
            if not news:
                st.warning("ไม่พบข่าวใหม่ในช่วงนี้")
            else:
                for item in news:
                    th_title, th_sum = translate_and_summarize(item['summary'], item['title'])
                    st.markdown(f"""
                    <div class="news-item">
                        <div style="font-size:1.1rem; font-weight:bold; color:#fff;">{th_title}</div>
                        <div style="font-size:0.9rem; color:#aaa; margin-top:5px;">{th_sum}</div>
                        <div style="margin-top:10px; font-size:0.8rem; display:flex; justify-content:space-between;">
                            <span style="color:#2962FF;">Source: {item['source']}</span>
                            <a href="{item['link']}" target="_blank" style="color:#00E676; text-decoration:none;">อ่านต่อฉบับเต็ม 🔗</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with t4:
            info = ticker.info
            st.markdown("<div class='section-header'>📊 Fundamental & Key Stats</div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Market Cap", f"{info.get('marketCap', 0):,}")
            with col2:
                st.metric("PE Ratio", f"{info.get('trailingPE', 0):.2f}")
            with col3:
                st.metric("52 Week High", f"{info.get('fiftyTwoWeekHigh', 0):,.2f}")
            with col4:
                st.metric("52 Week Low", f"{info.get('fiftyTwoWeekLow', 0):,.2f}")
                
            st.markdown("---")
            st.markdown(f"**Business Summary:** {info.get('longBusinessSummary', 'No summary available.')[:500]}...")

# --- Footer ---
st.markdown("""
    <div style="text-align:center; margin-top:50px; color:#666; border-top:1px solid #333; padding-top:20px;">
        Smart Trader AI Pro Max © 2024 | Designed for Precision Trading<br>
        <small>Data delayed by 15 mins. Investment involves risk.</small>
    </div>
""", unsafe_allow_html=True)
