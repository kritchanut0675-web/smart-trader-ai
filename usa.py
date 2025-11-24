import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from textblob import TextBlob
from googletrans import Translator
import feedparser
from bs4 import BeautifulSoup

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Smart Trader AI (Pro Max)",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .sr-table { 
            width: 100%; border-collapse: separate; border-spacing: 0; 
            border-radius: 8px; overflow: hidden; background-color: #000000; font-size: 0.95em;
        }
        .sr-table td { padding: 8px 10px; border-bottom: 1px solid #222; }
        .metric-box {
            background-color: #1E1E1E; border: 1px solid #333; padding: 15px; 
            border-radius: 10px; text-align: center; margin-bottom: 10px;
        }
        .trend-up { color: #00FF00; font-weight: bold; }
        .trend-down { color: #FF0000; font-weight: bold; }
        .trend-neu { color: #FFFF00; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Smart Trader AI (Complete Edition)")


# --- 2. Functions ---

def clean_html(html_text):
    try:
        return BeautifulSoup(html_text, "html.parser").get_text()
    except:
        return html_text


def get_data(symbol, period, interval):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty and symbol.endswith("-THB"):
        base_symbol = symbol.replace("-THB", "-USD")
        st.toast(f"🔄 กำลังดึง {base_symbol} และแปลงค่าเงิน...", icon="💱")
        df = yf.Ticker(base_symbol).history(period=period, interval=interval)
        try:
            usd_thb = yf.Ticker("THB=X").history(period="1d")['Close'].iloc[-1]
        except:
            usd_thb = 34.0

        if not df.empty:
            cols = ['Open', 'High', 'Low', 'Close']
            df[cols] = df[cols] * usd_thb

    return df


def analyze_trend_summary(df):
    """วิเคราะห์ภาพรวมตลาด"""
    last_close = df['Close'].iloc[-1]
    ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
    ema200 = df['Close'].ewm(span=200).mean().iloc[-1]
    rsi = df['RSI'].iloc[-1]

    # 1. Trend Analysis
    if last_close > ema50 and last_close > ema200:
        trend = "ขาขึ้นแข็งแกร่ง (Strong Uptrend)"
        trend_color = "trend-up"
    elif last_close < ema50 and last_close < ema200:
        trend = "ขาลง (Downtrend)"
        trend_color = "trend-down"
    elif last_close > ema200:
        trend = "กำลังกลับตัวขึ้น (Bullish Bias)"
        trend_color = "trend-up"
    else:
        trend = "ไซด์เวย์ (Sideways)"
        trend_color = "trend-neu"

    # 2. RSI Status
    if rsi > 70:
        rsi_stat = "Overbought (ระวังแรงขาย)"
        rsi_color = "trend-down"
    elif rsi < 30:
        rsi_stat = "Oversold (รอสัญญาณซื้อ)"
        rsi_color = "trend-up"
    else:
        rsi_stat = "ปกติ (Neutral)"
        rsi_color = "trend-neu"

    return trend, trend_color, rsi_stat, rsi_color


def is_psychological(price):
    if price > 10000:
        return price % 1000 < 100 or price % 1000 > 900
    elif price > 100:
        return price % 10 < 0.5 or price % 10 > 9.5
    else:
        return abs(price - round(price)) < 0.1


def analyze_levels(df):
    raw_levels = []
    # Fractal Logic
    for i in range(2, df.shape[0] - 2):
        if df['Low'][i] < df['Low'][i - 1] and df['Low'][i] < df['Low'][i + 1] and \
                df['Low'][i + 1] < df['Low'][i + 2] and df['Low'][i - 1] < df['Low'][i - 2]:
            raw_levels.append({'price': df['Low'][i], 'type': 'Support'})

        if df['High'][i] > df['High'][i - 1] and df['High'][i] > df['High'][i + 1] and \
                df['High'][i + 1] > df['High'][i + 2] and df['High'][i - 1] > df['High'][i - 2]:
            raw_levels.append({'price': df['High'][i], 'type': 'Resistance'})

    if not raw_levels: return []

    raw_levels.sort(key=lambda x: x['price'])
    clusters = []
    threshold = df['Close'].mean() * 0.015

    for lvl in raw_levels:
        if not clusters:
            clusters.append({'price': lvl['price'], 'count': 1, 'type': lvl['type']})
            continue
        last = clusters[-1]
        if abs(lvl['price'] - last['price']) < threshold:
            last['count'] += 1
            last['price'] = (last['price'] * (last['count'] - 1) + lvl['price']) / last['count']
        else:
            clusters.append({'price': lvl['price'], 'count': 1, 'type': lvl['type']})

    final_levels = []
    for c in clusters:
        price = c['price']
        count = c['count']
        lvl_type = c['type']
        label = "แนวต้านปกติ" if lvl_type == 'Resistance' else "แนวรับปกติ"
        color = "white"
        width = 1
        is_psych = is_psychological(price)

        if count >= 4:
            label = "🔥 แข็งแกร่งมาก"
            color = "#FF0000" if lvl_type == 'Resistance' else "#00FF00"
            width = 3
        elif count == 3:
            label = "💪 แข็งแกร่ง"
            color = "#FF4500" if lvl_type == 'Resistance' else "#32CD32"
            width = 2
        elif is_psych:
            label = "🧠 จิตวิทยา"
            color = "#FFD700"
            width = 2
        elif count == 1:
            label = "☁️ บางที่สุด"
            color = "rgba(255, 255, 255, 0.3)"
            width = 1

        final_levels.append({
            'price': price, 'type': lvl_type, 'label': label,
            'color': color, 'width': width, 'score': count + (2 if is_psych else 0)
        })
    return final_levels


def get_news(query):
    q = query.replace("-THB", "").replace("-USD", "")
    url = f"https://news.google.com/rss/search?q={q}+when:3d&hl=en-US&gl=US&ceid=US:en"
    return feedparser.parse(url).entries[:5]


def translate_content(text):
    try:
        translator = Translator()
        return translator.translate(text[:500], src='en', dest='th').text
    except:
        return text


# --- 3. Sidebar Controls ---
st.sidebar.header("🔍 ค้นหา")
col_search, col_btn = st.sidebar.columns([3, 1])
symbol = col_search.text_input("Symbol", value="BTC-THB", label_visibility="collapsed").upper().strip()
if col_btn.button("↻", help="Refresh Data"):
    st.rerun()

with st.sidebar.expander("⚙️ ตั้งค่ากราฟ (Indicators)", expanded=False):
    period = st.selectbox("ย้อนหลัง", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox("Timeframe", ["1d", "1wk"], index=0)
    show_ema = st.checkbox("Show EMA (50/200)", True)
    show_rsi = st.checkbox("Show RSI", True)
    show_vol = st.checkbox("Show Volume", True)
    show_trend = st.checkbox("Trend Line", True)
    st.markdown("---")
    show_weak = st.checkbox("แสดงแนวรับบางๆ", False)

# --- 4. Main Process ---
if symbol:
    with st.spinner('กำลังโหลดข้อมูล...'):
        df = get_data(symbol, period, interval)

    if df.empty:
        st.error("❌ ไม่พบข้อมูล")
    else:
        df.reset_index(inplace=True)
        df['Date'] = df.index if 'Date' not in df.columns else df['Date']

        # --- Calculate Indicators ---
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # EMA
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # Levels
        levels_data = analyze_levels(df)
        current_price = df['Close'].iloc[-1]
        display_levels = [l for l in levels_data if show_weak or l['label'] != "☁️ บางที่สุด"]

        # Trend Analysis Summary
        trend_txt, trend_col, rsi_txt, rsi_col = analyze_trend_summary(df)

        # --- Sidebar Display ---
        price_color = "green" if df['Close'].iloc[-1] > df['Close'].iloc[-2] else "red"
        st.sidebar.markdown(f"""
        <div class="metric-box">
            <h3 style="margin:0; color: #aaa;">{symbol}</h3>
            <h1 style="margin:0; font-size: 36px; color: {price_color};">{current_price:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

        # AI Summary Box
        st.sidebar.markdown("### 🤖 AI Market Status")
        st.sidebar.markdown(f"""
        <div style="background:#222; padding:10px; border-radius:5px; margin-bottom:15px; border-left: 4px solid #fff;">
            <div>📈 เทรนด์: <span class="{trend_col}">{trend_txt}</span></div>
            <div style="margin-top:5px;">📊 RSI: <span class="{rsi_col}">{rsi_txt}</span></div>
        </div>
        """, unsafe_allow_html=True)

        supports = sorted([l for l in display_levels if l['type'] == "Support" and l['price'] < current_price],
                          key=lambda x: x['price'], reverse=True)[:3]
        resistances = sorted([l for l in display_levels if l['type'] == "Resistance" and l['price'] > current_price],
                             key=lambda x: x['price'])[:3]

        # Resistance Table
        st.sidebar.markdown('<div class="res-header">🟥 แนวต้าน (Resistance)</div>', unsafe_allow_html=True)
        res_table = """<table class="sr-table" style="border: 1px solid #ff4b4b;">"""
        for r in reversed(resistances):
            res_table += f"""<tr style="color: #ff4b4b;"><td style="text-align:left;">{r['label']}</td><td style="text-align:right; font-family:monospace; font-weight:bold;">{r['price']:,.2f}</td></tr>"""
        res_table += "</table>"
        if not resistances: res_table = "<p style='color:#777; font-style:italic;'>- ไม่มีแนวต้านใกล้เคียง -</p>"
        st.sidebar.markdown(res_table, unsafe_allow_html=True)

        # Support Table
        st.sidebar.markdown('<div class="sup-header">🟩 แนวรับ (Support)</div>', unsafe_allow_html=True)
        sup_table = """<table class="sr-table" style="border: 1px solid #00c853;">"""
        for s in supports:
            sup_table += f"""<tr style="color: #00c853;"><td style="text-align:left;">{s['label']}</td><td style="text-align:right; font-family:monospace; font-weight:bold;">{s['price']:,.2f}</td></tr>"""
        sup_table += "</table>"
        if not supports: sup_table = "<p style='color:#777; font-style:italic;'>- ไม่มีแนวรับใกล้เคียง -</p>"
        st.sidebar.markdown(sup_table, unsafe_allow_html=True)

        # --- Chart Display ---
        # ปรับความสูง Row: ถ้ามี RSI ให้ Row 2 สูงหน่อย, Volume ซ้อนกับ Price หรือแยก?
        # เพื่อความสวยงาม: Row 1 = Price+EMA, Row 2 = Volume, Row 3 = RSI

        rows = 1
        row_heights = [1.0]

        if show_vol and show_rsi:
            rows = 3
            row_heights = [0.6, 0.2, 0.2]
        elif show_vol or show_rsi:
            rows = 2
            row_heights = [0.7, 0.3]

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

        # 1. Price Candle
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)

        # EMA Lines
        if show_ema:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], name='EMA 50', line=dict(color='#2962FF', width=1.5)),
                          row=1, col=1)
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['EMA200'], name='EMA 200', line=dict(color='#FF6D00', width=1.5)), row=1,
                col=1)

        # S/R Levels
        for lvl in display_levels:
            dash_style = "solid" if "แข็งแกร่ง" in lvl['label'] else "dot" if "บาง" in lvl['label'] else "dash"
            opacity = 0.9 if "แข็งแกร่ง" in lvl['label'] else 0.4
            fig.add_hline(
                y=lvl['price'], line_dash=dash_style, line_color=lvl['color'], line_width=lvl['width'], opacity=opacity,
                annotation_text=f"{lvl['price']:,.2f}", annotation_position="top right",
                annotation_font_color=lvl['color'], annotation_font_size=11, row=1, col=1
            )

        if show_trend:
            x_nums = np.arange(len(df))
            slope, intercept, _, _, _ = linregress(x_nums, df['Close'])
            fig.add_trace(go.Scatter(x=df['Date'], y=slope * x_nums + intercept, mode='lines', name='Trend',
                                     line=dict(color='#FFFF00', width=2)), row=1, col=1)

        # 2. Volume & RSI Placement logic
        next_row = 2

        if show_vol:
            colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name='Volume', opacity=0.8),
                          row=next_row, col=1)
            next_row += 1

        if show_rsi:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#A569BD')),
                          row=next_row if show_vol else 2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=next_row if show_vol else 2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=next_row if show_vol else 2, col=1)

        fig.update_layout(height=700, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

        # --- News ---
        st.markdown("### 📰 ข่าว & Sentiment Analysis")
        col1, col2 = st.columns([3, 1])

        news_items = get_news(symbol)
        if not news_items:
            st.info("ไม่พบข่าวล่าสุดในช่วง 3 วัน")
        else:
            for item in news_items:
                title_en = item.title
                summary_en = clean_html(item.get('description', ''))
                link = item.link
                blob = TextBlob(title_en)
                polarity = blob.sentiment.polarity

                sentiment_icon = "😐"
                box_color = "#333"
                if polarity > 0.1:
                    sentiment_icon = "🚀"
                    box_color = "rgba(0, 200, 0, 0.1)"
                elif polarity < -0.1:
                    sentiment_icon = "🔻"
                    box_color = "rgba(200, 0, 0, 0.1)"

                title_th = translate_content(title_en)
                summary_th = translate_content(summary_en)

                st.markdown(f"""
                <div style="background-color: {box_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid {'green' if polarity > 0.1 else 'red' if polarity < -0.1 else 'gray'};">
                    <h4 style="margin:0;">{sentiment_icon} {title_th}</h4>
                    <p style="font-size:0.9em; color:#ccc; margin-top:5px;">{summary_th}</p>
                    <div style="font-size:0.8em; color:#777; margin-top:10px; display:flex; justify-content:space-between;">
                         <span>Original: {title_en[:50]}...</span>
                         <a href="{link}" target="_blank" style="color:#4fa8ff; text-decoration:none;">อ่านฉบับเต็ม 🔗</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)