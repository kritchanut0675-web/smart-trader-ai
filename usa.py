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
    page_title="Smart Trader AI : Visual Pro",
    layout="wide",
    page_icon="✨",
    initial_sidebar_state="expanded"
)

if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'

def set_symbol(sym):
    st.session_state.symbol = sym

# --- CSS Styling (Visual Upgrade) ---
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        
        /* Input & Button */
        div[data-testid="stTextInput"] input {
            font-size: 20px !important; height: 50px !important;
            border-radius: 12px !important; background-color: #151515 !important;
            color: #fff !important; border: 1px solid #333 !important;
        }
        div[data-testid="stButton"] button {
            height: 50px !important; font-size: 20px !important;
            border-radius: 12px !important; width: 100% !important;
            background: linear-gradient(90deg, #00C853, #69F0AE); 
            color: #000 !important; border: none !important; font-weight: bold !important;
        }
        
        /* AI Report Design */
        .ai-card {
            background: #1E1E1E; border-radius: 15px; padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 1px solid #333;
            margin-bottom: 20px;
        }
        .ai-header {
            display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid #333; padding-bottom: 15px; margin-bottom: 15px;
        }
        .verdict-badge {
            padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 1.2rem;
            color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .section-title { font-size: 1.1rem; font-weight: bold; color: #aaa; margin-top: 15px; margin-bottom: 8px; }
        .analysis-text { color: #e0e0e0; line-height: 1.6; font-size: 1rem; }
        
        /* Grid Stats inside AI Report */
        .ai-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px; }
        .ai-stat-box { background: #252525; padding: 15px; border-radius: 10px; text-align: center; }
        
        /* Guru & News */
        .guru-card {
            background: linear-gradient(135deg, #1a237e 0%, #000000 100%);
            padding: 20px; border-radius: 15px; border: 1px solid #304FFE;
        }
        .news-content { font-size: 1rem; line-height: 1.7; color: #ddd; background: #1a1a1a; padding: 15px; border-radius: 10px; }
        
        button[data-baseweb="tab"] { font-size: 1.1rem !important; padding: 15px !important; flex: 1; }
        .stButton button { width: 100%; }
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

def analyze_levels(df):
    levels = []
    # Fractal Logic
    for i in range(2, df.shape[0]-2):
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1]:
            levels.append({'p': df['Low'][i], 't': 'Support'})
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1]:
            levels.append({'p': df['High'][i], 't': 'Resistance'})
    
    # Clustering (รวมราคาใกล้เคียงกัน)
    levels.sort(key=lambda x: x['p'])
    clusters = []
    threshold = df['Close'].mean() * 0.02 # 2% threshold
    
    for l in levels:
        if not clusters: clusters.append(l); continue
        if abs(l['p'] - clusters[-1]['p']) < threshold:
            # Keep the more recent one or average? Let's take average for smoother lines
            clusters[-1]['p'] = (clusters[-1]['p'] + l['p']) / 2
        else: clusters.append(l)
        
    return clusters

def get_guru_insight(ticker, price):
    try:
        info = ticker.info
        name = info.get('longName', 'Unknown')
        target = info.get('targetMeanPrice', 0)
        rec = info.get('recommendationKey', '-').upper().replace('_', ' ')
        pe = info.get('trailingPE', 0)
        beta = info.get('beta', 0)
        div_yield = info.get('dividendYield', 0)
        high52 = info.get('fiftyTwoWeekHigh', 0)
        low52 = info.get('fiftyTwoWeekLow', 0)
        
        insight = f"**{name}**"
        if target > 0:
            upside = ((target - price) / price) * 100
            insight += f" (Upside: {upside:.1f}%)"
        
        return insight, rec, target, pe, beta, div_yield, high52, low52
    except: return "No Data", "-", 0, 0, 0, 0, 0, 0

# --- News ---
@st.cache_data(ttl=3600) 
def fetch_content(url, backup=""):
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 10
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text
        if len(text) < 150: return backup if backup else "เนื้อหาถูกจำกัดสิทธิ์"
        return text[:4000]
    except: return backup if backup else "ไม่สามารถดึงเนื้อหาได้"

def translate_text(text):
    try: return GoogleTranslator(source='auto', target='th').translate(text[:4500])
    except: return text

def get_hybrid_news(ticker, symbol):
    news_list = []
    seen = set()
    try:
        yf_news = ticker.news
        if yf_news:
            for item in yf_news[:3]:
                if item['link'] not in seen:
                    news_list.append({'title': item['title'], 'link': item['link'], 'summary': item.get('title',''), 'source': 'Yahoo'})
                    seen.add(item['link'])
    except: pass
    if len(news_list) < 3:
        try:
            q = symbol.replace("-THB", "").replace("-USD", "").upper()
            rss_url = f"https://news.google.com/rss/search?q={q}+stock+news+when:3d&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for item in feed.entries[:5]:
                if item.link not in seen:
                    soup = BeautifulSoup(item.get('description', ''), "html.parser")
                    news_list.append({'title': item.title, 'link': item.link, 'summary': soup.get_text(), 'source': 'Google'})
                    seen.add(item.link)
        except: pass
    return news_list[:5]

# --- 🤖 AI Report Generator (Beautiful HTML) ---
def generate_ai_html(symbol, price, df, target, pe, rec):
    close = df['Close'].iloc[-1]
    ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    # Logic
    score = 50
    trend_txt = "ไซด์เวย์"
    if close > ema200: 
        score += 20
        trend_txt = "ขาขึ้น (Uptrend)"
    else: 
        score -= 20
        trend_txt = "ขาลง (Downtrend)"
        
    if rsi < 30: score += 15
    elif rsi > 70: score -= 15
    
    if target > 0 and price < target: score += 15
    
    # Verdict
    if score >= 70: 
        verdict = "🟢 STRONG BUY"
        color = "#00E676"
        desc = "กราฟเป็นขาขึ้นและมีปัจจัยบวกสนับสนุน เป็นจังหวะที่ดีในการเข้าสะสม"
    elif score >= 55:
        verdict = "🟢 BUY"
        color = "#66BB6A"
        desc = "แนวโน้มดี แต่อาจรอจังหวะย่อตัวเล็กน้อยก่อนเข้า"
    elif score <= 30:
        verdict = "🔴 STRONG SELL"
        color = "#FF1744"
        desc = "กราฟเสียทรงและมีความเสี่ยงสูง ควรระมัดระวังหรือตัดขาดทุน"
    elif score <= 45:
        verdict = "🔴 SELL/WAIT"
        color = "#FF5252"
        desc = "แรงส่งยังไม่ดี ควรชะลอการลงทุนหรือขายทำกำไร"
    else:
        verdict = "🟡 HOLD/WAIT"
        color = "#FFD600"
        desc = "ตลาดยังเลือกทางไม่ชัดเจน แนะนำให้ถือรอหรือ Wait & See"

    html = f"""
    <div class="ai-card">
        <div class="ai-header">
            <div style="font-size:1.5rem; font-weight:bold; color:#fff;">🤖 AI Analysis: {symbol}</div>
            <div class="verdict-badge" style="background:{color};">{verdict}</div>
        </div>
        
        <div style="margin-bottom:20px; text-align:center;">
            <span style="font-size:3rem; font-weight:bold; color:{color};">{score}/100</span>
            <div style="color:#aaa;">AI Score Confidence</div>
            <p style="margin-top:10px; color:#ddd;">"{desc}"</p>
        </div>
        
        <div class="ai-grid">
            <div class="ai-stat-box">
                <div style="color:#aaa;">Trend</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#fff;">{trend_txt}</div>
            </div>
            <div class="ai-stat-box">
                <div style="color:#aaa;">RSI Momentum</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#fff;">{rsi:.1f}</div>
            </div>
            <div class="ai-stat-box">
                <div style="color:#aaa;">Analyst View</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#fff;">{rec}</div>
            </div>
            <div class="ai-stat-box">
                <div style="color:#aaa;">Fair Value</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#fff;">{target:,.2f}</div>
            </div>
        </div>
    </div>
    """
    return html

# --- 3. UI Layout ---

with st.sidebar:
    st.markdown("### ⭐ รายการโปรด (Watchlist)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Bitcoin"): set_symbol("BTC-USD")
        if st.button("Gold"): set_symbol("GC=F")
        if st.button("Nvidia"): set_symbol("NVDA")
    with col2:
        if st.button("Tesla"): set_symbol("TSLA")
        if st.button("Apple"): set_symbol("AAPL")
        if st.button("Oil"): set_symbol("CL=F")
        
    st.markdown("---")
    st.header("⚙️ Setting")
    # Default Heikin Ashi for smoothness
    chart_type = st.radio("Chart Type", ["Heikin Ashi (Smooth)", "Candlestick (Real)"], index=0)
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=4) 
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    show_ema = st.checkbox("Show EMA 50/200", True)
    show_sr = st.checkbox("Show S/R Lines", True)

st.markdown("### ✨ Smart Trader AI : Visual Pro")
col_in, col_btn = st.columns([3.5, 1])

with col_in: 
    symbol_input = st.text_input("Search", value=st.session_state.symbol, label_visibility="collapsed")
with col_btn: 
    if st.button("GO"):
        st.session_state.symbol = symbol_input
        st.rerun()

symbol = st.session_state.symbol.upper().strip()

if symbol:
    with st.spinner(f'✨ Rendering beautiful charts for {symbol}...'):
        df, ticker = get_data(symbol, period, interval)
    
    if df.empty:
        st.warning(f"ไม่พบข้อมูล '{symbol}'")
    else:
        # Indicators
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        
        price = df['Close'].iloc[-1]
        change = price - df['Close'].iloc[-2]
        pct = (change / df['Close'].iloc[-2]) * 100
        color_p = "#00E676" if change >= 0 else "#FF1744"
        
        levels = analyze_levels(df)
        insight, rec, target, pe, beta, div_yield, high52, low52 = get_guru_insight(ticker, price)
        
        # --- Header ---
        st.markdown(f"""
        <div style="background:#111; padding:20px; border-radius:15px; border-top:5px solid {color_p}; text-align:center; margin-bottom:20px;">
            <div style="font-size:1.2rem; color:#aaa;">{symbol}</div>
            <div style="font-size:3.5rem; font-weight:bold; color:{color_p}; letter-spacing: 2px;">{price:,.2f}</div>
            <div style="font-size:1.2rem; color:{color_p};">{change:+,.2f} ({pct:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 กราฟ & S/R", "🧱 ตาราง S/R", "📰 ข่าว", "🧐 กูรู", "🤖 AI Analysis"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
            
            # 1. Price Chart (Heikin Ashi / Candle)
            if "Heikin Ashi" in chart_type:
                ha_df = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(
                    x=ha_df.index, open=ha_df['HA_Open'], high=ha_df['HA_High'], low=ha_df['HA_Low'], close=ha_df['HA_Close'], 
                    name="Price", increasing_line_color='#00F2B6', decreasing_line_color='#FF3B30'
                ), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                    name="Price", increasing_line_color='#00F2B6', decreasing_line_color='#FF3B30'
                ), row=1, col=1)

            # 2. EMA Lines (Smooth)
            if show_ema:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2962FF', width=1.5), name="EMA50"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#FF9100', width=1.5), name="EMA200"), row=1, col=1)
            
            # 3. Support / Resistance Lines (On Chart)
            if show_sr:
                # กรองเอาเฉพาะที่ใกล้ราคาปัจจุบัน +/- 15% เพื่อไม่ให้รก
                nearby_levels = [l for l in levels if abs(l['p'] - price) / price < 0.15]
                for l in nearby_levels:
                    c = 'rgba(0, 230, 118, 0.6)' if l['t'] == "Support" else 'rgba(255, 23, 68, 0.6)'
                    fig.add_hline(y=l['p'], line_dash="dash", line_color=c, line_width=1, row=1, col=1,
                                  annotation_text=f"{l['p']:,.2f}", annotation_position="top right", annotation_font_color=c)

            # 4. RSI with Visual Signals
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#D1C4E9', width=1.5), name="RSI"), row=2, col=1)
            
            # RSI Thresholds
            fig.add_hline(y=70, line_color='#FF3B30', line_width=1, line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_color='#00F2B6', line_width=1, line_dash='dot', row=2, col=1)
            
            # RSI Background Zones
            fig.add_hrect(y0=70, y1=100, fillcolor="#FF3B30", opacity=0.1, layer="below", row=2, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="#00F2B6", opacity=0.1, layer="below", row=2, col=1)
            
            # RSI Signal Markers (จุดไข่ปลาแจ้งเตือน)
            # หาจุดที่ RSI ตัด 30 ขึ้น (Buy) หรือตัด 70 ลง (Sell)
            rsi_buy = df[ (df['RSI'] < 30) ]
            rsi_sell = df[ (df['RSI'] > 70) ]
            
            fig.add_trace(go.Scatter(
                x=rsi_buy.index, y=rsi_buy['RSI'], mode='markers', name='Oversold',
                marker=dict(color='#00F2B6', size=6, symbol='circle')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=rsi_sell.index, y=rsi_sell['RSI'], mode='markers', name='Overbought',
                marker=dict(color='#FF3B30', size=6, symbol='circle')
            ), row=2, col=1)

            # Zoom Buttons
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=5, label="1W", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(step="all", label="All")
                    ]),
                    font=dict(color="black"), bgcolor="#DDDDDD", activecolor="#2962FF"
                ),
                row=1, col=1
            )

            # Clean Layout
            fig.update_layout(
                height=600, margin=dict(l=10, r=10, t=10, b=10), 
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_showgrid=False, yaxis_showgrid=True, yaxis_gridcolor='rgba(255,255,255,0.1)',
                dragmode='pan', hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
            
        with tab2:
            st.markdown("#### 🧱 Key Levels")
            res = sorted([l for l in levels if l['type']=='Resistance' and l['price']>price], key=lambda x: x['price'])[:4]
            sup = sorted([l for l in levels if l['type']=='Support' and l['price']<price], key=lambda x: x['price'], reverse=True)[:4]
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown("<div style='text-align:center; color:#FF3B30; font-weight:bold; margin-bottom:10px;'>🟥 RESISTANCE</div>", unsafe_allow_html=True)
                for r in reversed(res): st.markdown(f"<div style='border-bottom:1px solid #333; padding:8px; text-align:center; font-family:monospace; font-size:1.1rem;'>{r['price']:,.2f}</div>", unsafe_allow_html=True)
            with c_b:
                st.markdown("<div style='text-align:center; color:#00F2B6; font-weight:bold; margin-bottom:10px;'>🟩 SUPPORT</div>", unsafe_allow_html=True)
                for s in sup: st.markdown(f"<div style='border-bottom:1px solid #333; padding:8px; text-align:center; font-family:monospace; font-size:1.1rem;'>{s['price']:,.2f}</div>", unsafe_allow_html=True)

        with tab3:
            news_items = get_hybrid_news(ticker, symbol)
            if not news_items: st.info("ไม่พบข่าวในขณะนี้")
            else:
                for i, item in enumerate(news_items):
                    blob = TextBlob(item['title'])
                    icon = "🟢" if blob.sentiment.polarity > 0.1 else "🔴" if blob.sentiment.polarity < -0.1 else "⚪"
                    try: title_th = translate_text(item['title'])
                    except: title_th = item['title']
                    with st.expander(f"{icon} {title_th}", expanded=(i==0)):
                        st.caption(f"Source: {item['source']}")
                        with st.spinner("Loading..."):
                            body = translate_text(fetch_content(item['link'], item['summary']))
                        st.markdown(f"<div class='news-content'>{body}</div>", unsafe_allow_html=True)
                        st.markdown(f"<a href='{item['link']}' target='_blank' style='display:block; text-align:center; padding:10px; background:#333; color:white; border-radius:8px; margin-top:10px; text-decoration:none;'>🔗 อ่านต้นฉบับ</a>", unsafe_allow_html=True)

        with tab4:
            st.markdown(f"""
            <div class="guru-card">
                <div style="font-size:1.3rem; font-weight:bold; color:white; margin-bottom:10px;">🧐 Guru Analysis</div>
                <div style="background:rgba(0,0,0,0.3); padding:15px; border-radius:10px; color:#ddd; line-height:1.6; margin-bottom:15px;">
                    {insight}
                </div>
                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; text-align:center;">
                    <div class="stat-card"><div class="stat-value" style="color:#00E676">{rec}</div><div class="stat-label">Consensus</div></div>
                    <div class="stat-card"><div class="stat-value">{target:,.2f}</div><div class="stat-label">Target Price</div></div>
                    <div class="stat-card"><div class="stat-value">{pe:.2f}</div><div class="stat-label">P/E Ratio</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # --- 🔥 Tab 5: Visual AI Report ---
        with tab5:
            ai_html = generate_ai_html(symbol, price, df, target, pe, rec)
            st.markdown(ai_html, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** ข้อมูลทั้งหมดเพื่อการศึกษาเท่านั้น (Educational Purpose Only)")
