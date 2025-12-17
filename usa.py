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
import requests
import datetime
import re

# --- Libraries Setup ---
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try: 
    nltk.data.find('tokenizers/punkt')
except LookupError: 
    nltk.download('punkt')

# --- 1. Setup & Design ---
st.set_page_config(
    page_title="Smart Trader AI : Ultra Black",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'symbol' not in st.session_state: st.session_state.symbol = 'GOOGL'

def set_symbol(sym): st.session_state.symbol = sym

# --- 2. CSS Styling (Ultra Modern UI) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
        
        .stApp { background-color: #050505 !important; color: #e0e0e0; }
        
        /* Input Field */
        div[data-testid="stTextInput"] input { 
            background-color: #111 !important; color: #fff !important; 
            font-weight: bold !important; font-size: 1.2rem !important;
            border: 2px solid #00E5FF !important; border-radius: 10px;
        }

        /* Cards */
        .glass-card {
            background: linear-gradient(145deg, #1a1a1a, #0d0d0d);
            border: 1px solid #333; border-radius: 20px;
            padding: 25px; margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        
        /* Stat Metric Box */
        .metric-box {
            background: #111; border-radius: 15px; padding: 20px;
            border-left: 4px solid #333; position: relative; overflow: hidden;
            transition: transform 0.2s;
            height: 100%;
        }
        .metric-box:hover { transform: translateY(-5px); border-left-color: #00E5FF; }
        .metric-label { font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .metric-val { font-size: 1.8rem; font-weight: 800; color: #fff; margin-top: 5px; }
        
        /* S/R Dynamic Cards */
        .sr-card {
            padding: 15px 20px; border-radius: 12px; margin-bottom: 10px;
            display: flex; justify-content: space-between; align-items: center;
            border: 1px solid rgba(255,255,255,0.05); backdrop-filter: blur(5px);
        }
        .sr-res { background: linear-gradient(90deg, rgba(255, 23, 68, 0.2), rgba(0,0,0,0)); border-left: 5px solid #FF1744; }
        .sr-sup { background: linear-gradient(90deg, rgba(0, 230, 118, 0.2), rgba(0,0,0,0)); border-left: 5px solid #00E676; }
        .sr-piv { background: linear-gradient(90deg, rgba(255, 214, 0, 0.2), rgba(0,0,0,0)); border-left: 5px solid #FFD600; }
        
        /* Static Grid Card */
        .static-card {
            background: #161616; padding: 15px; border-radius: 10px; 
            border: 1px solid #333; margin-bottom: 8px;
            display: flex; justify-content: space-between;
        }
        .static-label { color: #aaa; font-weight: 600; }
        .static-val { color: #00E5FF; font-weight: bold; }
        
        /* AI Verdict Ring */
        .verdict-ring {
            width: 140px; height: 140px; border-radius: 50%;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            font-size: 3rem; font-weight: 900; margin: 0 auto 20px auto;
            border: 8px solid #333; background: #000;
            box-shadow: 0 0 40px rgba(0,0,0,0.5);
        }
        
        /* AI Insight Box */
        .ai-insight-box {
            background: linear-gradient(135deg, #111, #0a0a0a);
            border: 1px solid #333; border-radius: 15px; padding: 25px;
            position: relative; overflow: hidden;
        }
        .ai-insight-icon { font-size: 2rem; margin-bottom: 10px; }
        
        /* NEWS CARD */
        .news-card { 
            padding: 20px; margin-bottom: 15px; background: #111; 
            border-radius: 15px; border-left: 5px solid #888; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .news-card:hover { transform: translateX(5px); background: #161616; }
        .nc-pos { border-left-color: #00E676; }
        .nc-neg { border-left-color: #FF1744; }
        .nc-neu { border-left-color: #FFD600; }
        
        /* GURU CARD */
        .guru-card {
            background: #111; padding: 15px; border-radius: 12px; 
            border: 1px solid #333; margin-bottom: 10px; font-size: 0.95rem;
        }
        
        .ai-article {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px; border-radius: 15px;
            border-left: 4px solid #00E5FF;
            font-size: 1rem; line-height: 1.8; color: #ddd;
            margin-top: 20px;
        }

        /* Custom Tabs */
        button[data-baseweb="tab"] { 
            font-size: 1rem !important; font-weight: 600 !important; 
            border-radius: 8px !important; margin: 0 4px !important;
            background: #111 !important; border: 1px solid #333 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: #00E5FF !important; color: #000 !important; border-color: #00E5FF !important;
        }
        
        /* Centered Button */
        div.stButton > button {
            width: 100%; justify-content: center; font-size: 1.1rem !important; 
            padding: 12px !important; border-radius: 12px !important;
            background: linear-gradient(45deg, #00E5FF, #2979FF); 
            border: none !important; color: #000 !important; font-weight: 800 !important;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
        }
        div.stButton > button:hover {
            transform: scale(1.02); box-shadow: 0 0 25px rgba(0, 229, 255, 0.6);
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. Functions ---

@st.cache_data(ttl=300)
def get_market_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass
                
        if not df.empty and 'Close' in df.columns and len(df) > 0:
            return df
            
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info and len(info) > 5: return info
        return {} 
    except: return {}

# [UPDATED] Super Robust Financial Data (Revenue, Net Income, Cash Flow)
@st.cache_data(ttl=3600)
def get_financial_data_robust(symbol):
    try:
        if "-USD" in symbol or "=F" in symbol:
            return None # Crypto/Forex usually don't have this
            
        ticker = yf.Ticker(symbol)
        
        # 1. Income Statement
        try:
            inc = ticker.financials
            if inc.empty: inc = ticker.income_statement
        except: inc = pd.DataFrame()
        
        # 2. Cash Flow Statement
        try:
            cf = ticker.cashflow
            if cf.empty: cf = ticker.cash_flow
        except: cf = pd.DataFrame()

        if inc.empty and cf.empty:
            return None

        # Helper to find row with fuzzy matching
        def get_row_data(df, keywords):
            for idx in df.index:
                for k in keywords:
                    if k.lower() == str(idx).lower() or k.lower() in str(idx).lower():
                        return df.loc[idx]
            return None

        # Extract Revenue
        rev_data = get_row_data(inc, ['Total Revenue', 'Revenue', 'Gross Revenue', 'Operating Revenue'])
        
        # Extract Net Income
        net_data = get_row_data(inc, ['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing And Discontinued Operation'])
        
        # Extract Operating Cash Flow
        ocf_data = get_row_data(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Cash Flow From Continuing Operating Activities'])

        # Combine into DataFrame
        combined = pd.DataFrame()
        if rev_data is not None: combined['Revenue'] = rev_data
        if net_data is not None: combined['Net Income'] = net_data
        if ocf_data is not None: combined['Operating Cash Flow'] = ocf_data
        
        if not combined.empty:
            # Sort by date ascending (old -> new)
            combined.sort_index(inplace=True)
            # Filter only last 4 years/periods if too many
            if len(combined) > 5:
                combined = combined.tail(5)
            return combined
            
        return None
    except Exception as e:
        return None

# [NEW] AI Financial Analyst Logic
def analyze_financial_health_score(df):
    score = 0
    reasons = []
    
    if len(df) < 2:
        return 5, ["⚠️ ข้อมูลย้อนหลังไม่เพียงพอสำหรับการวิเคราะห์แนวโน้ม"], "Insufficient Data", "#888"

    latest = df.iloc[-1]
    prev = df.iloc[0] # Compare with oldest available in the window
    
    # 1. Revenue Growth (3 Points)
    if 'Revenue' in df.columns:
        rev_growth = ((latest['Revenue'] - prev['Revenue']) / prev['Revenue']) * 100
        if rev_growth > 50: score += 3; reasons.append(f"✅ รายได้เติบโตอย่างก้าวกระโดด (+{rev_growth:.0f}%)")
        elif rev_growth > 20: score += 2; reasons.append(f"✅ รายได้เติบโตดี (+{rev_growth:.0f}%)")
        elif rev_growth > 0: score += 1; reasons.append(f"⚖️ รายได้เติบโตเล็กน้อย (+{rev_growth:.0f}%)")
        else: score -= 1; reasons.append(f"⚠️ รายได้หดตัว ({rev_growth:.0f}%)")
        
    # 2. Net Income (3 Points)
    if 'Net Income' in df.columns:
        if latest['Net Income'] > 0:
            score += 1; reasons.append("✅ บริษัทมีกำไรสุทธิ (Profitable)")
            net_growth = ((latest['Net Income'] - prev['Net Income']) / abs(prev['Net Income'])) * 100
            if net_growth > 20: score += 2; reasons.append("✅ กำไรเติบโตแข็งแกร่ง")
            elif net_growth > 0: score += 1
        else:
            score -= 2; reasons.append("❌ ขาดทุนสุทธิ (Net Loss)")
            
    # 3. Cash Flow & Quality (4 Points)
    if 'Operating Cash Flow' in df.columns:
        cf = latest['Operating Cash Flow']
        ni = latest.get('Net Income', 0)
        
        if cf > 0: 
            score += 2; reasons.append("✅ กระแสเงินสดเป็นบวก (Cash Flow Positive)")
            if ni > 0 and cf > ni: 
                score += 2; reasons.append("💎 Quality of Earnings ดีเยี่ยม (CFO > Net Income)")
            elif ni > 0 and cf < ni:
                score -= 1; reasons.append("⚠️ ระวัง: กำไรทางบัญชีสูงกว่าเงินสดที่ได้รับจริง (Accrual Hazard)")
        else:
            score -= 2; reasons.append("❌ กระแสเงินสดติดลบ (Cash Burn)")
            
    # Final Adjustments
    score = max(0, min(10, score))
    
    verdict = "Strong Buy / Growth" if score >= 8 else "Buy / Stable" if score >= 6 else "Hold / Watch" if score >= 4 else "High Risk / Avoid"
    color = "#00E676" if score >= 7 else "#FFD600" if score >= 4 else "#FF1744"
    
    return score, reasons, verdict, color

def get_sector_pe_benchmark(sector):
    benchmarks = {
        'Technology': 25, 'Financial Services': 15, 'Healthcare': 22, 
        'Consumer Cyclical': 20, 'Industrials': 20, 'Energy': 12,
        'Communication Services': 20, 'Basic Materials': 15,
        'Real Estate': 30, 'Utilities': 18
    }
    return benchmarks.get(sector, 20) 

def calculate_strategic_supports(price, setup_data=None):
    if price > 2000000: step = 50000       
    elif price > 100000: step = 10000      
    elif price > 50000: step = 2000        
    elif price > 10000: step = 1000        
    elif price > 1000: step = 100          
    elif price > 100: step = 10            
    elif price > 10: step = 1              
    elif price > 1: step = 0.1             
    else: step = 0.01

    base = (price // step) * step
    if (price - base) < (step * 0.05): base = base - step

    is_uptrend = False
    if setup_data and "UPTREND" in setup_data.get('trend', ''):
        is_uptrend = True
    
    if is_uptrend:
        l1_act, l1_desc = "ไม้ที่ 1: ย่อซื้อ (Buy on Dip)", "เทรนด์ขาขึ้น แนวรับแรกมีโอกาสเด้งสูง"
        l2_act, l3_act = "ไม้ที่ 2: สะสมเพิ่ม (Add Position)", "ไม้ที่ 3: รับแน่น (Strong Buy)"
        allocs = ["30%", "40%", "30%"] 
    else: 
        l1_act, l1_desc = "ไม้ที่ 1: แหย่เบาๆ (Risky)", "เทรนด์ไม่ชัดเจน/ขาลง เสี่ยงหลุดสูง"
        l2_act, l3_act = "ไม้ที่ 2: รอเด้ง (Play Bounce)", "ไม้ที่ 3: ถัวเฉลี่ย (DCA)"
        allocs = ["10%", "30%", "60%"] 

    levels = [
        {"name": "🛡️ แนวรับแรก (First Sup)", "price": base, "action": l1_act, "alloc": allocs[0], "color": "#FFD600", "bar": 30 if is_uptrend else 15, "desc": l1_desc},
        {"name": "🧠 แนวรับจิตวิทยา (Psych Sup)", "price": base - step, "action": l2_act, "alloc": allocs[1], "color": "#FF9100", "bar": 40, "desc": "โซนแนวรับจิตวิทยา ตัวเลขกลมๆ"},
        {"name": "💎 แนวรับแข็งแกร่ง (Strong Sup)", "price": base - (step * 2.5), "action": l3_act, "alloc": allocs[2], "color": "#00E676", "bar": 80, "desc": "โซน Deep Value ปลอดภัยสูง"}
    ]
    return levels, step

def generate_ai_trade_reasoning(price, setup, strat_levels, val_score):
    first_sup = strat_levels[0]['price']
    gap_first = ((price - first_sup) / price) * 100
    
    if setup['trend'] == "UPTREND (ขาขึ้น)":
        if gap_first < 2.0: return "✅ BUY ON DIP (ย่อซื้อในขาขึ้น)", "กราฟเป็นขาขึ้นและราคาย่อตัวลงมาใกล้ 'แนวรับแรก' เป็นจังหวะที่ดีในการเข้าซื้อเพื่อเก็งกำไรตามเทรนด์ (Trend Following)", "#00E676", "🚀"
        elif setup['rsi_val'] > 70: return "⚠️ WAIT / TAKE PROFIT (ระวังแรงเทขาย)", "แม้จะเป็นขาขึ้น แต่ราคาและ RSI เข้าโซน Overbought (ซื้อมากเกินไป) เสี่ยงต่อการพักตัว ห้ามไล่ราคา", "#FFD600", "✋"
        else: return "📈 HOLD / RUN TREND", "กราฟยังแข็งแกร่ง ใครมีของให้ถือต่อ (Let Profit Run) ใครไม่มีของรอจังหวะย่อตัว", "#2979FF", "💎"
    elif setup['trend'] == "DOWNTREND (ขาลง)":
        if val_score >= 8: return "💎 VALUE BUY (ของดีราคาถูก)", "กราฟระยะสั้นเป็นขาลง แต่ในมุมมอง Valuation ถือว่าถูกมาก (Deep Value) ทยอยสะสมได้", "#00E676", "💰"
        elif gap_first < 1.0: return "⚔️ PLAY BOUNCE (เก็งกำไรเด้ง)", "ราคาชนแนวรับจิตวิทยา มีโอกาสเด้งสั้นๆ แต่เนื่องจากเป็นขาลงหลัก ให้เล่นรอบเร็ว (Hit & Run)", "#FF9100", "⚡"
        else: return "⛔ AVOID (อย่าเพิ่งรับมีด)", "กราฟเป็นขาลงชัดเจนและราคายังลอยตัวกลางอากาศ แนะนำให้นั่งทับมือรอไปก่อน", "#FF1744", "🛑"
    else: return "⚖️ SIDEWAY (รอเลือกทาง)", "กราฟออกข้าง ไม่ชัดเจน แนะนำให้ซื้อที่แนวรับและขายที่แนวต้าน (Swing Trade)", "#E0E0E0", "⚖️"

def analyze_stock_guru(info, setup, symbol):
    if not info: info = {}
    pe, roe = info.get('trailingPE'), info.get('returnOnEquity')
    
    if pe is None:
        val_score = 5
        reasons_q, reasons_v = ["ℹ️ ไม่พบข้อมูล P/E (Switch to Technical Mode)"], []
        if "UPTREND" in setup['trend']: val_score += 3; reasons_v.append("✅ Trend เป็นขาขึ้น (Bullish)")
        elif "DOWNTREND" in setup['trend']: val_score -= 2; reasons_v.append("❌ Trend เป็นขาลง (Bearish)")
        if setup['rsi_val'] < 30: val_score += 2; reasons_v.append("✅ RSI Oversold (ขายมากเกินไป)")
        elif setup['rsi_val'] > 70: val_score -= 2; reasons_v.append("⚠️ RSI Overbought (แพงระยะสั้น)")
        return {"verdict": "Technical Speculation", "color": "#2979FF", "val_score": max(0, min(10, val_score)), "article": f"เนื่องจากระบบไม่พบข้อมูลพื้นฐานของ **{symbol}** (อาจเป็นหุ้น Growth, Crypto หรือข้อมูลมาช้า) \n\nAI จึงเปลี่ยนมาวิเคราะห์ด้วย **Technical Analysis** แทน โดยพบว่าแนวโน้มปัจจุบันเป็น **{setup['trend']}** และ RSI อยู่ที่ **{setup['rsi_val']:.1f}** ซึ่งเป็นปัจจัยหลักในการตัดสินใจซื้อขายขณะนี้", "reasons_q": reasons_q, "reasons_v": reasons_v}

    peg, pb = info.get('pegRatio'), info.get('priceToBook')
    profit_margin, rev_growth = info.get('profitMargins', 0), info.get('revenueGrowth', 0)
    sector = info.get('sector', 'General')
    
    val_score, reasons_q, reasons_v = 0, [], []
    if roe and roe > 0.15: reasons_q.append("✅ ROE สูง (>15%) บริหารทุนเก่ง")
    elif roe and roe < 0: reasons_q.append("❌ ROE ติดลบ ขาดทุน")
    if profit_margin and profit_margin > 0.10: reasons_q.append("✅ อัตรากำไรดี (>10%)")
    if rev_growth and rev_growth > 0: reasons_q.append("✅ รายได้เติบโต")
    else: reasons_q.append("⚠️ รายได้ไม่โต หรือหดตัว")
    
    if pe:
        if pe < 15: val_score += 3; reasons_v.append("✅ P/E ต่ำ (ถูก)")
        elif pe < 25: val_score += 2; reasons_v.append("⚖️ P/E เหมาะสม")
        elif pe < 40: val_score += 1; reasons_v.append("⚠️ P/E เริ่มสูง")
    else: val_score += 1
    
    if peg:
        if peg < 1.0: val_score += 3; reasons_v.append("✅ PEG < 1 (โตคุ้มราคา)")
        elif peg < 2.0: val_score += 2; reasons_v.append("⚖️ PEG ปกติ")
        else: val_score += 0; reasons_v.append("❌ PEG สูง (โตไม่ทันราคา)")
    
    if pb and pb < 3: val_score += 2
    if roe and roe > 0.15: val_score += 2
    val_score = min(10, val_score)

    intro = f"จากการวิเคราะห์หุ้น **{symbol}** ในกลุ่มอุตสาหกรรม **{sector}** ด้วยระบบ AI Guru พบข้อมูลที่น่าสนใจดังนี้:\n\n"
    val_text = ""
    if pe:
        if pe < 15: val_text = f"ในมุมมองความคุ้มค่า (Valuation) หุ้นตัวนี้ถือว่า **'ราคาถูก (Undervalued)'** เมื่อเทียบกับกำไรที่ทำได้ โดยมีค่า P/E อยู่ที่ {pe:.2f} ซึ่งต่ำกว่าเกณฑ์มาตรฐาน "
        elif pe > 40: val_text = f"ในมุมมองความคุ้มค่า ราคาหุ้นปัจจุบันค่อนข้าง **'แพง (Overvalued)'** มีค่า P/E สูงถึง {pe:.2f} สะท้อนความคาดหวังของนักลงทุนที่สูงมาก "
        else: val_text = f"ราคาหุ้นปัจจุบันถือว่า **'สมเหตุสมผล (Fair Price)'** มีค่า P/E อยู่ที่ {pe:.2f} เป็นระดับที่ยอมรับได้ "

    qual_text = ""
    if roe and roe > 0.15: qual_text = f"\n\nด้านคุณภาพบริษัท (Quality) จัดว่ายอดเยี่ยม มี ROE สูงถึง {roe*100:.1f}% แสดงถึงความสามารถในการบริหารเงินทุนของผู้บริหารที่เก่งกาจ "
    elif profit_margin and profit_margin < 0.05: qual_text = f"\n\nด้านคุณภาพอาจต้องระวังเรื่องอัตรากำไรที่ค่อนข้างบาง ({profit_margin*100:.1f}%) ซึ่งอาจเปราะบางต่อเศรษฐกิจ "

    tech_text = f"\n\n**คำแนะนำเชิงกลยุทธ์:** เมื่อประกอบกับกราฟเทคนิคที่เป็น **{setup['trend']}** "
    if setup['trend'] == "UPTREND (ขาขึ้น)":
        if val_score >= 7: tech_text += "และพื้นฐานที่แข็งแกร่ง **'แนะนำให้ทยอยสะสม (Buy)'** ได้ทันที เพราะทั้งพื้นฐานและเทคนิคสนับสนุนกัน ราคาเป้าหมายยังมี Upside"
        else: tech_text += "แม้เทคนิคจะดูดี แต่พื้นฐานเริ่มตึงตัว **'แนะนำให้เก็งกำไรระยะสั้น (Trading)'** และวางจุด Stop Loss อย่างเคร่งครัด ไม่ควรถือยาว"
    elif setup['trend'] == "DOWNTREND (ขาลง)":
        if val_score >= 8: tech_text += "แม้พื้นฐานจะดีและราคาถูกมาก แต่กราฟยังเป็นขาลง **'แนะนำให้ Wait & See'** รอให้กราฟสร้างฐานหรือยืนเหนือเส้น EMA ก่อนค่อยเข้าซื้อ จะได้ของดีในราคาที่ปลอดภัยกว่า"
        else: tech_text += "ประกอบกับพื้นฐานที่อ่อนแอ/แพง **'แนะนำให้หลีกเลี่ยง (Avoid)'** ไปก่อน จนกว่าจะมีสัญญาณการกลับตัวที่ชัดเจน"
    else: tech_text += "ควรรอให้ราคาเลือกทางที่ชัดเจนก่อนเข้าลงทุน (Wait for Breakout)"

    full_article = intro + val_text + qual_text + tech_text
    if val_score >= 8: status, color = "💎 Hidden Gem (ของดีราคาถูก)", "#00E676"
    elif val_score >= 5: status, color = "⚖️ Fair Value (เหมาะสม)", "#FFD600"
    else: status, color = "⚠️ High Risk / Expensive", "#FF1744"
    return {"verdict": status, "color": color, "val_score": val_score, "article": full_article, "reasons_q": reasons_q, "reasons_v": reasons_v}

@st.cache_data(ttl=15)
def get_bitkub_ticker():
    try:
        r = requests.get("https://api.bitkub.com/api/market/ticker", timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

# --- NEWS SYSTEM (FREE & KEYLESS) ---
@st.cache_data(ttl=3600)
def get_ai_analyzed_news_thai(symbol):
    news_list = []
    translator = GoogleTranslator(source='auto', target='th') if HAS_TRANSLATOR else None
    
    # Use Google News RSS (Free, No Key)
    try:
        cl_sym = symbol.replace("-THB","").replace("-USD","").replace("=F","")
        q = urllib.parse.quote(f"site:bloomberg.com {cl_sym} market")
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
        
        # Fallback to general search if bloomberg is empty
        if not feed.entries:
            q = urllib.parse.quote(f"{cl_sym} finance news")
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
            
        for i in feed.entries[:8]:
            t, s = i.title, re.sub(re.compile('<.*?>'), '', getattr(i, 'summary', '') or getattr(i, 'description', ''))[:300]
            sc = TextBlob(t).sentiment.polarity
            
            if sc > 0.05: lbl, icon, cls = "ข่าวดี (Positive)", "🚀", "nc-pos"
            elif sc < -0.05: lbl, icon, cls = "ข่าวร้าย (Negative)", "🔻", "nc-neg"
            else: lbl, icon, cls = "ทั่วไป (Neutral)", "⚖️", "nc-neu"
            
            t_th, s_th = t, s
            if translator:
                try: 
                    t_th = translator.translate(t)
                    if s: s_th = translator.translate(s) 
                except: pass
            
            news_list.append({'title': t_th, 'summary': s_th, 'link': i.link, 'icon': icon, 'class': cls, 'label': lbl, 'score': sc, 'source': 'Google News'})
    except Exception as e: 
        pass
        
    return news_list[:10]

def calculate_technical_setup(df):
    try:
        delta = df['Close'].diff()
        gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        close, ema50, ema200 = df['Close'].iloc[-1], df['Close'].ewm(span=50).mean().iloc[-1], df['Close'].ewm(span=200).mean().iloc[-1]
        atr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        
        if close > ema50 and ema50 > ema200: trend, sig, col, sc = "UPTREND (ขาขึ้น)", "BUY", "#00E676", 2
        elif close < ema50 and ema50 < ema200: trend, sig, col, sc = "DOWNTREND (ขาลง)", "SELL", "#FF1744", -2
        else: trend, sig, col, sc = "SIDEWAYS (ออกข้าง)", "WAIT", "#FFD600", 0
        return {'trend': trend, 'signal': sig, 'color': col, 'rsi_series': rsi_series, 'rsi_val': rsi_series.iloc[-1], 'entry': close, 'sl': close-(1.5*atr) if sc>=0 else close+(1.5*atr), 'tp': close+(2.5*atr) if sc>=0 else close-(2.5*atr)}
    except: return None

def calculate_pivot_points(df):
    try:
        p = df.iloc[-2]
        pp = (p['High']+p['Low']+p['Close'])/3
        return {"PP":pp, "R1":(2*pp)-p['Low'], "S1":(2*pp)-p['High'], "R2":pp+(p['High']-p['Low']), "S2":pp-(p['High']-p['Low'])}
    except: return None

def calculate_dynamic_levels(df):
    try:
        sma, std = df['Close'].rolling(20).mean().iloc[-1], df['Close'].rolling(20).std().iloc[-1]
        return {"EMA 20": df['Close'].ewm(span=20).mean().iloc[-1], "EMA 50": df['Close'].ewm(span=50).mean().iloc[-1], "EMA 200": df['Close'].ewm(span=200).mean().iloc[-1], "BB Upper": sma+(2*std), "BB Lower": sma-(2*std), "Current": df['Close'].iloc[-1]}
    except: return None

def generate_dynamic_insight(price, pivots, dynamics):
    e200, e20 = dynamics['EMA 200'], dynamics['EMA 20']
    if price > e200: msg, col, icon = "Bullish Strong (แกร่งมาก)", "#00E676", "🐂" if price > e20 else ("Bullish Retrace (ย่อตัว)", "#00E676", "📉")
    else: msg, col, icon = "Bearish Strong (ลงหนัก)", "#FF1744", "🐻" if price < e20 else ("Bearish Correction (ดีดตัว)", "#FF1744", "📈")
    
    all_lvls = {**pivots, **{k:v for k,v in dynamics.items() if k!='Current'}}
    n_name, n_price, min_d = "", 0, float('inf')
    for k,v in all_lvls.items():
        if abs(price-v) < min_d: min_d, n_name, n_price = abs(price-v), k, v
    
    dist_pct = (min_d / price) * 100
    act = f"⚠️ ราคากำลังทดสอบแนว **{n_name}** ({n_price:,.2f}) ระยะห่างเพียง {dist_pct:.2f}%" if dist_pct < 0.8 else f"มีพื้นที่วิ่ง (Room to run) ไปหา **{n_name}** ({n_price:,.2f})"
    return msg, col, icon, act

def calculate_heikin_ashi(df):
    ha = df.copy()
    ha['Close'] = (df['Open']+df['High']+df['Low']+df['Close'])/4
    ha['Open'] = [ (df['Open'][0]+df['Close'][0])/2 ] + [0]*(len(df)-1)
    for i in range(1, len(df)): ha['Open'].iloc[i] = (ha['Open'].iloc[i-1]+ha['Close'].iloc[i-1])/2
    ha['High'] = ha[['High','Open','Close']].max(axis=1)
    ha['Low'] = ha[['Low','Open','Close']].min(axis=1) 
    return ha

def gen_ai_verdict(setup, news):
    score, t_txt, n_txt = 50, "", ""
    if setup['trend'] == "UPTREND (ขาขึ้น)": score += 20; t_txt = "กราฟเป็นขาขึ้นชัดเจน ยืนเหนือ EMA"
    elif setup['trend'] == "DOWNTREND (ขาลง)": score -= 20; t_txt = "กราฟเป็นขาลง หลุดแนวรับสำคัญ"
    else: t_txt = "กราฟออกข้าง รอเลือกทาง"
    
    if setup['rsi_val'] > 70: score -= 5; t_txt += " (Overbought ระวังย่อ)"
    elif setup['rsi_val'] < 30: score += 5; t_txt += " (Oversold ลุ้นเด้ง)"
    
    n_score = sum([n['score'] for n in news]) if news else 0
    if n_score > 0.3: score += 15; n_txt = "ข่าวสารเชิงบวก สนับสนุนราคา"
    elif n_score < -0.3: score -= 15; n_txt = "ข่าวสารเชิงลบ กดดันตลาด"
    else: n_txt = "ข่าวสารทรงตัว ไม่มีประเด็นใหญ่"
    
    score = max(0, min(100, score))
    verd = "STRONG BUY" if score>=80 else "BUY" if score>=60 else "SELL" if score<=40 else "STRONG SELL" if score<=20 else "HOLD"
    return t_txt, n_txt, score, verd

def calculate_static_round_numbers(price):
    if price <= 0: return {"Res 1": 0, "Sup 1": 0, "Step": 0}
    magnitude = 10 ** (len(str(int(price))) - 1)
    if magnitude == 0: magnitude = 1
    
    if price < 10: step = 0.5
    elif price < 100: step = 5
    elif price < 1000: step = 50
    elif price < 10000: step = 500
    elif price < 100000: step = 5000
    else: step = 10000 
    
    ceil_val = (int(price) // step + 1) * step
    floor_val = (int(price) // step) * step
    
    return {"Res 1": ceil_val, "Sup 1": floor_val, "Step": step}

def calculate_bitkub_ai_levels(high24, low24, last):
    diff = high24 - low24
    if diff == 0: diff = 1
    
    fib_levels = {
        "0.0": high24,
        "0.236": high24 - (diff * 0.236),
        "0.382": high24 - (diff * 0.382),
        "0.5": high24 - (diff * 0.5),
        "0.618": high24 - (diff * 0.618),
        "1.0": low24
    }
    
    status = "NEUTRAL"
    color = "#FFD600"
    
    if last > fib_levels["0.236"]:
        status = "BULLISH BREAKOUT"
        color = "#00E676"
    elif last < fib_levels["0.618"]:
        status = "BEARISH DIP"
        color = "#FF1744"
    
    levels = [
        {"name": "High 24h", "price": high24},
        {"name": "Fib 0.236", "price": fib_levels["0.236"]},
        {"name": "Mid 50%", "price": fib_levels["0.5"]},
        {"name": "Fib 0.618", "price": fib_levels["0.618"]},
        {"name": "Low 24h", "price": low24}
    ]
    
    return {
        "status": status, 
        "color": color, 
        "fib": {"top": fib_levels["0.236"], "bot": fib_levels["0.618"]},
        "levels": levels
    }

def analyze_bitkub_static_guru(price, static_lvls):
    r1 = static_lvls['Res 1']
    s1 = static_lvls['Sup 1']
    mid = (r1 + s1) / 2
    
    if price > mid:
        verdict = "Uptrend Bias"
        color = "#00E676"
        desc = f"ราคาอยู่ครึ่งบนของกรอบ {s1:,} - {r1:,} มีโอกาสทดสอบแนวต้าน"
        strat = "Buy on Dip"
    else:
        verdict = "Downtrend Bias"
        color = "#FF1744"
        desc = f"ราคาอยู่ครึ่งล่างของกรอบ {s1:,} - {r1:,} ระวังหลุดแนวรับ"
        strat = "Wait & See"
        
    return verdict, color, desc, strat


# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;color:#00E5FF;'>💎 ULTRA</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("BTC"): set_symbol("BTC-USD")
    if c2.button("ETH"): set_symbol("ETH-USD")
    st.markdown("---")
    st.markdown("### 🇹🇭 Bitkub Rate")
    bk_data = get_bitkub_ticker()
    if bk_data:
        b, e = bk_data.get('THB_BTC',{}), bk_data.get('THB_ETH',{})
        st.markdown(f"**BTC:** <span style='color:#00E676'>{b.get('last',0):,.0f}</span>", unsafe_allow_html=True)
        st.markdown(f"**ETH:** <span style='color:#00E676'>{e.get('last',0):,.0f}</span>", unsafe_allow_html=True)
    st.markdown("---")
    chart_type = st.selectbox("Chart Style", ["Candlestick", "Heikin Ashi"])
    period = st.select_slider("Period", ["1mo","3mo","6mo","1y"], value="6mo")
    interval = st.selectbox("Timeframe", ["1d", "1wk", "1h", "15m", "5m"], index=0)

# --- 5. Main ---
st.markdown("<h2 style='color:#00E5FF;'>🔍 Smart Search</h2>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 1]) 
with c1: sym_input = st.text_input("Symbol", st.session_state.symbol, label_visibility="collapsed")
with c2: 
    if st.button("วิเคราะห์ ⚡", use_container_width=True): 
        set_symbol(sym_input); st.rerun()

symbol = st.session_state.symbol.upper()

if symbol:
    with st.spinner("🚀 AI Analyzing..."):
        df = get_market_data(symbol, period, interval)
    
    if not df.empty:
        # [NEW FEATURE] Download Button in Sidebar
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        
        csv_data = convert_df(df)
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Download Market Data (CSV)",
            data=csv_data,
            file_name=f'{symbol}_data.csv',
            mime='text/csv',
        )

        curr, chg = df['Close'].iloc[-1], df['Close'].iloc[-1] - df['Close'].iloc[-2]
        pct, color = (chg / df['Close'].iloc[-2]) * 100, "#00E676" if chg >= 0 else "#FF1744"
        
        setup = calculate_technical_setup(df)
        news = get_ai_analyzed_news_thai(symbol)
        info = get_stock_info(symbol) 
        
        t_txt, n_txt, ai_sc, ai_vd = gen_ai_verdict(setup, news)
        sc_col, sc_glow = ("#00E676", "0, 230, 118") if ai_sc >= 70 else ("#FF1744", "255, 23, 68") if ai_sc <= 30 else ("#FFD600", "255, 214, 0")
        trend_status, trend_icon, trend_color_css = ("BULLISH (กระทิง)", "🐂", "#00E676") if "UPTREND" in setup['trend'] else ("BEARISH (หมี)", "🐻", "#FF1744") if "DOWNTREND" in setup['trend'] else ("SIDEWAY", "⚖️", "#FFD600")

        st.markdown(f"""<div class="glass-card" style="border-top:5px solid {color};text-align:center;"><div style="font-size:3.5rem;font-weight:900;line-height:1;margin-bottom:10px;">{symbol}</div><div style="font-size:3rem;color:{color};font-weight:bold;">{curr:,.2f}</div><div style="background:rgba({sc_glow}, 0.2);padding:5px 20px;border-radius:20px;display:inline-block;margin-top:10px;"><span style="color:{color};font-weight:bold;font-size:1.1rem;">{chg:+.2f} ({pct:+.2f}%)</span></div></div>""", unsafe_allow_html=True)

        tabs = st.tabs(["📈 Chart", "📊 Stats", "📰 AI News", "🎯 Setup", "🤖 Verdict", "🛡️ S/R Dynamic", "🧠 AI Guru", "💰 Financials", "🇹🇭 Bitkub AI", "🧮 Calc"])

        with tabs[0]:
            exp12 = df['Close'].ewm(span=12, adjust=False).mean()
            exp26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal_line

            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.6, 0.15, 0.25],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
            )

            if chart_type == "Heikin Ashi":
                ha = calculate_heikin_ashi(df)
                fig.add_trace(go.Candlestick(x=df.index, open=ha['Open'], high=ha['High'], low=ha['Low'], close=ha['Close'], name="HA"), row=1, col=1)
            else:
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), line=dict(color='#2979FF', width=1.5), name="EMA50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=200).mean(), line=dict(color='#FF9100', width=1.5), name="EMA200"), row=1, col=1)

            colors_vol = ['#00E676' if r.Open < r.Close else '#FF1744' for i, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol, showlegend=False), row=2, col=1)

            fig.add_trace(go.Bar(x=df.index, y=macd_hist, name='MACD Hist', marker_color='#00E5FF'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd, line=dict(color='#fff', width=1), name='MACD'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=signal_line, line=dict(color='#FFD600', width=1), name='Signal'), row=3, col=1)

            fig.update_layout(
                template='plotly_dark', 
                height=700, 
                margin=dict(l=0,r=0,t=0,b=0), 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            st.markdown(f"""<div style="background:{trend_color_css}20; border:2px solid {trend_color_css}; padding:15px; border-radius:15px; text-align:center; margin-bottom:20px;"><h2 style="margin:0; color:{trend_color_css}; font-size:2rem;">{trend_icon} {trend_status}</h2><p style="margin:5px 0 0 0; color:#ddd;">Market Trend Indicator</p></div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-box'><div class='metric-label'>High (สูงสุด)</div><div class='metric-val' style='color:#00E676'>{df['High'].max():,.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><div class='metric-label'>Low (ต่ำสุด)</div><div class='metric-val' style='color:#FF1744'>{df['Low'].min():,.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-box'><div class='metric-label'>Volume (ปริมาณ)</div><div class='metric-val' style='color:#E040FB'>{df['Volume'].iloc[-1]/1e6:.1f}M</div></div>", unsafe_allow_html=True)
            if info:
                sector, pe = info.get('sector', 'Unknown'), info.get('trailingPE')
                if pe:
                    st.markdown("---")
                    st.markdown(f"<h3 style='color:#00E5FF;'>📊 AI Valuation & P/E Analysis</h3>", unsafe_allow_html=True)
                    st.markdown(f"**Industry:** {sector}")
                    c_pe1, c_pe2 = st.columns(2)
                    with c_pe1: st.markdown(f"""<div class='metric-box'><div class='metric-label'>P/E Ratio (ปัจจุบัน)</div><div class='metric-val'>{pe:.2f}</div><div style='color:#888; font-size:0.8rem;'>ระยะเวลาคืนทุนโดยประมาณ (ปี)</div></div>""", unsafe_allow_html=True)
                    with c_pe2:
                        avg_pe = get_sector_pe_benchmark(sector)
                        diff = ((pe - avg_pe) / avg_pe) * 100
                        status, color, icon = ("Overvalued (แพงกว่ากลุ่ม)", "#FF1744", "🔺") if diff > 15 else ("Undervalued (ถูกกว่ากลุ่ม)", "#00E676", "💎") if diff < -15 else ("Fair Price (ราคาเหมาะสม)", "#FFD600", "⚖️")
                        st.markdown(f"""<div class='metric-box' style='border-left-color:{color}'><div class='metric-label'>AI Sector Compare (Avg {avg_pe})</div><div class='metric-val' style='color:{color}; font-size:1.6rem;'>{icon} {status}</div><div style='color:#ccc; font-size:0.9rem;'>Difference: {diff:+.1f}%</div></div>""", unsafe_allow_html=True)

        with tabs[2]:
            st.markdown("### 📰 Market Sentiment (Free Source)")
            if news:
                for n in news: st.markdown(f"""<div class="news-card {n['class']}"><div style="display:flex;justify-content:space-between;margin-bottom:5px;"><div style="display:flex;align-items:center;gap:10px;"><span style="font-size:1rem;">{n['icon']}</span><span style="font-weight:bold;color:#fff;">{n['label']}</span></div><span style="font-size:0.8rem;background:#333;padding:2px 8px;border-radius:5px;">{n['source']}</span></div><h4 style="margin:10px 0;color:#e0e0e0;">{n['title']}</h4><p style="color:#aaa;font-size:0.9rem;line-height:1.5;">{n['summary']}</p><div style="text-align:right;margin-top:10px;"><a href="{n['link']}" target="_blank" style="color:#00E5FF;text-decoration:none;">🔗 อ่านต่อ</a></div></div>""", unsafe_allow_html=True)
            else: st.info("ไม่พบข่าว หรือ Internet มีปัญหา")

        with tabs[3]:
            if setup:
                st.markdown(f"""<div class='ai-insight-box' style='border-left: 5px solid {setup['color']}; margin-bottom:20px;'><h2 style='margin:0; color:{setup['color']};'>{setup['signal']}</h2><p style='font-size:1.2rem; color:#ccc; margin-top:5px;'>{setup['trend']}</p><div style='margin-top:15px; display:flex; gap:10px;'><span style='background:#111; padding:5px 15px; border-radius:10px; border:1px solid #333;'>RSI: {setup['rsi_val']:.1f}</span><span style='background:#111; padding:5px 15px; border-radius:10px; border:1px solid #333;'>Entry: {setup['entry']:,.2f}</span></div></div>""", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-box' style='border-left-color:#00E5FF'><div class='metric-label'>Buy Zone</div><div class='metric-val'>{curr*0.99:,.2f}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-box' style='border-left-color:#00E676'><div class='metric-label'>Target (TP)</div><div class='metric-val'>{setup['tp']:,.2f}</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-box' style='border-left-color:#FF1744'><div class='metric-label'>Stop Loss</div><div class='metric-val'>{setup['sl']:,.2f}</div></div>", unsafe_allow_html=True)

        with tabs[4]:
            col_v1, col_v2 = st.columns([1, 1.5])
            with col_v1: st.markdown(f"""<div class="verdict-ring" style="border-color:{sc_col}; color:{sc_col}; box-shadow:0 0 30px rgba({sc_glow}, 0.5);">{ai_sc}</div><div style="text-align:center; font-size:2rem; font-weight:900; color:{sc_col}; text-transform:uppercase; letter-spacing:2px;">{ai_vd}</div>""", unsafe_allow_html=True)
            with col_v2:
                st.markdown("### 🔍 AI Analysis Breakdown")
                st.markdown(f"""<div class="factor-card" style="border-left-color:{sc_col};"><h4 style="margin:0;color:#fff;">📈 Technical Insight</h4><p style="margin-top:5px;color:#ccc;">{t_txt}</p></div><div class="factor-card" style="border-left-color:{'#00E676' if 'บวก' in n_txt else '#FF1744'};"><h4 style="margin:0;color:#fff;">📰 News Sentiment</h4><p style="margin-top:5px;color:#ccc;">{n_txt}</p></div>""", unsafe_allow_html=True)

        with tabs[5]:
            st.markdown("### 🧠 AI Strategic Support (วางแผนการรับของ)")
            strat_levels, step_size = calculate_strategic_supports(curr, setup)
            gap_pct = ((curr - strat_levels[0]['price']) / curr) * 100
            st.markdown(f"""<div style="background:rgba(0, 229, 255, 0.1); padding:15px; border-radius:10px; border-left:4px solid #00E5FF; margin-bottom:20px;"><h4 style="margin:0; color:#00E5FF;">💡 AI Strategy Advisor</h4><p style="margin:5px 0 0 0; color:#ddd;">ราคาปัจจุบันห่างจากแนวรับแรก <b>{gap_pct:.2f}%</b> (Step: {step_size:,.2f})<br>แนะนำให้แบ่งไม้ซื้อตามระดับแนวรับเพื่อบริหารต้นทุน (DCA/Grid Trading)</p></div>""", unsafe_allow_html=True)
            for lvl in strat_levels:
                l_gap = ((curr - lvl['price']) / curr) * 100
                is_near = "ใกล้ถึงแล้ว! 🚨" if l_gap < 1.0 else f"อีก {l_gap:.2f}%"
                st.markdown(f"""<div style="background: linear-gradient(145deg, #1a1a1a, #111); border: 1px solid #333; border-left: 6px solid {lvl['color']}; border-radius: 15px; padding: 20px; margin-bottom: 15px; position: relative; overflow: hidden;"><div style="display:flex; justify-content:space-between; align-items:flex-start;"><div><div style="font-size:1.1rem; font-weight:bold; color:{lvl['color']}; text-transform:uppercase; margin-bottom:5px;">{lvl['name']}</div><div style="font-size:2rem; font-weight:900; color:#fff; line-height:1;">{lvl['price']:,.2f}</div><div style="font-size:0.9rem; color:#888; margin-top:5px;">📉 ระยะห่าง: {is_near}</div></div><div style="text-align:right;"><span style="background:{lvl['color']}20; color:{lvl['color']}; padding:5px 12px; border-radius:20px; font-weight:bold; font-size:0.9rem;">แนะนำ: {lvl['alloc']}</span></div></div><div style="margin-top:15px; padding-top:15px; border-top:1px solid rgba(255,255,255,0.1);"><div style="font-weight:600; color:#eee; font-size:1rem;">{lvl['action']}</div><div style="font-size:0.9rem; color:#aaa;">{lvl['desc']}</div></div><div style="margin-top:10px; background:#333; height:6px; border-radius:3px; width:100%;"><div style="width:{lvl['bar']}%; background:{lvl['color']}; height:100%; border-radius:3px; box-shadow: 0 0 10px {lvl['color']};"></div></div></div>""", unsafe_allow_html=True)
            st.markdown("---")
            pivots = calculate_pivot_points(df)
            dynamic = calculate_dynamic_levels(df)
            if pivots and dynamic:
                msg, col, icon, act = generate_dynamic_insight(curr, pivots, dynamic)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🎯 Pivot Points (Day Trading)")
                    for k, v in pivots.items(): st.markdown(f"<div class='sr-card {'sr-res' if 'R' in k else 'sr-sup' if 'S' in k else 'sr-piv'}'><b>{k}</b><span>{v:,.2f}</span></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("#### 🌊 Dynamic Levels (EMA/Trend)")
                    for k, v in dynamic.items():
                        if k!="Current":
                            dist, cl = ((curr-v)/v)*100, "#00E676" if curr > v else "#FF1744"
                            st.markdown(f"<div class='sr-card' style='border-left:4px solid {cl}; background:rgba({255 if cl=='#FF1744' else 0}, {230 if cl=='#00E676' else 23}, {118 if cl=='#00E676' else 68}, 0.1);'><span>{k}</span><div style='text-align:right;'>{v:,.2f}<br><small style='color:{cl}'>{dist:+.2f}%</small></div></div>", unsafe_allow_html=True)

        with tabs[6]:
            st.markdown("### 🧠 AI Guru: Fundamental & Valuation")
            safe_info = info if info else {}
            summary = safe_info.get('longBusinessSummary')
            if summary:
                if HAS_TRANSLATOR:
                    try: summary = GoogleTranslator(source='auto', target='th').translate(summary[:2000])
                    except: pass
                st.info(f"**🏢 รู้จักกับ {symbol}:** {summary}")
            
            pe = safe_info.get('trailingPE')
            if pe:
                sector = safe_info.get('sector', 'Unknown')
                avg_pe = get_sector_pe_benchmark(sector)
                diff_pct = ((pe - avg_pe) / avg_pe) * 100
                pe_status, pe_color = ("แพงกว่ากลุ่ม (Overvalued)", "#FF1744") if diff_pct > 15 else ("ถูกกว่ากลุ่ม (Undervalued)", "#00E676") if diff_pct < -15 else ("ราคาเหมาะสม (Fair Value)", "#FFD600")
                st.markdown("#### ⚖️ Price vs Sector (เปรียบเทียบความถูกแพง)")
                col_pe1, col_pe2, col_pe3 = st.columns(3)
                with col_pe1: st.markdown(f"<div class='metric-box'><div class='metric-label'>{symbol} P/E</div><div class='metric-val'>{pe:.2f}</div></div>", unsafe_allow_html=True)
                with col_pe2: st.markdown(f"<div class='metric-box'><div class='metric-label'>Sector ({sector})</div><div class='metric-val' style='color:#888'>{avg_pe:.2f}</div></div>", unsafe_allow_html=True)
                with col_pe3: st.markdown(f"<div class='metric-box' style='border-left-color:{pe_color}'><div class='metric-label'>Verdict</div><div class='metric-val' style='color:{pe_color}; font-size:1.4rem;'>{pe_status}</div></div>", unsafe_allow_html=True)
                st.markdown("---")
            
            guru = analyze_stock_guru(safe_info, setup, symbol)
            strat_lvls, _ = calculate_strategic_supports(curr, setup)
            why_title, why_desc, why_color, why_icon = generate_ai_trade_reasoning(curr, setup, strat_lvls, guru['val_score'])
            st.markdown(f"""<div class='ai-insight-box' style='border:2px solid {guru['color']}; text-align:center; margin-bottom:20px;'><h1 style='color:{guru['color']}; font-size:3rem; margin:0;'>{guru['verdict']}</h1><div style="margin:20px 0; background:#333; border-radius:10px; height:10px; width:100%;"><div style="width:{guru['val_score']*10}%; background:{guru['color']}; height:100%; border-radius:10px;"></div></div><p style='font-size:1.1rem; color:#ccc;'>Valuation Score: {guru['val_score']}/10</p></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class='ai-insight-box' style='border-color:{why_color}; background:rgba(0,0,0,0.3); margin-bottom:20px;'><div style="display:flex; gap:15px; align-items:flex-start;"><span style="font-size:2.5rem;">{why_icon}</span><div><h3 style="margin:0; color:{why_color};">{why_title}</h3><p style="margin:5px 0 0 0; font-size:1.1rem; color:#ddd; line-height:1.5;">{why_desc}</p></div></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class='ai-article'><h4 style='margin-top:0; color:#fff;'>📝 บทวิเคราะห์โดย AI (AI Analyst Report)</h4>{guru['article']}</div>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🏢 Quality Score (พื้นฐาน)")
                for r in guru['reasons_q']: st.markdown(f"<div class='guru-card' style='border-left:4px solid {'#00E676' if '✅' in r else '#FF1744'};'>{r}</div>", unsafe_allow_html=True)
            with c2:
                for r in guru['reasons_v']: st.markdown(f"<div class='guru-card' style='border-left:4px solid {'#00E676' if '✅' in r else '#FF1744'};'>{r}</div>", unsafe_allow_html=True)

        with tabs[7]: # [NEW TAB] Financials
            st.markdown("### 💰 Financial Performance (งบการเงินย้อนหลัง)")
            
            fin_df = get_financial_data_robust(symbol)
            
            if fin_df is not None:
                # [NEW FEATURE] AI Financial Health Check
                f_score, f_reasons, f_verdict, f_color = analyze_financial_health_score(fin_df)
                
                st.markdown(f"""<div class='ai-insight-box' style='border-left: 5px solid {f_color}; margin-bottom:20px;'><h3 style='margin:0; color:{f_color};'>🏥 AI Financial Health Check: {f_score}/10</h3><p style='font-size:1.1rem; font-weight:bold; color:#fff;'>{f_verdict}</p><hr style='border-color:#333;'>{"".join([f"<div style='margin-bottom:5px;'>{r}</div>" for r in f_reasons])}</div>""", unsafe_allow_html=True)

                # 1. Income Statement Chart
                fig_inc = go.Figure()
                if 'Revenue' in fin_df.columns:
                    fig_inc.add_trace(go.Bar(x=fin_df.index.year, y=fin_df['Revenue'], name='Revenue', marker_color='#2979FF'))
                if 'Net Income' in fin_df.columns:
                    fig_inc.add_trace(go.Bar(x=fin_df.index.year, y=fin_df['Net Income'], name='Net Income', marker_color='#00E676'))
                
                fig_inc.update_layout(
                    template='plotly_dark', barmode='group', height=400,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    title="Revenue vs Net Income (รายได้ vs กำไรสุทธิ)",
                    xaxis_title="Year", yaxis_title="Amount"
                )
                st.plotly_chart(fig_inc, use_container_width=True)
                
                # 2. Cash Flow Chart (Separated)
                if 'Operating Cash Flow' in fin_df.columns:
                    fig_cf = go.Figure()
                    fig_cf.add_trace(go.Bar(x=fin_df.index.year, y=fin_df['Operating Cash Flow'], name='Operating Cash Flow', marker_color='#AA00FF'))
                    fig_cf.update_layout(
                        template='plotly_dark', height=400,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        title="Operating Cash Flow (กระแสเงินสดจากการดำเนินงาน)",
                        xaxis_title="Year", yaxis_title="Amount"
                    )
                    st.plotly_chart(fig_cf, use_container_width=True)
                
                # [NEW FEATURE] Financial Cheat Sheet (คำอธิบาย)
                st.markdown("---")
                st.markdown("#### 💡 Financial Cheat Sheet (คู่มืออ่านงบฉบับย่อ)")
                
                c_t1, c_t2, c_t3 = st.columns(3)
                with c_t1:
                    st.info("""
                    **1. Revenue (รายได้)**
                    * 💰 ยอดขายรวมก่อนหักค่าใช้จ่าย
                    * **ความหมาย:** บอกว่าสินค้าขายดีไหม บริษัทโตไหม
                    * **ดีคือ:** กราฟควรสูงขึ้นทุกปี 📈
                    """)
                with c_t2:
                    st.info("""
                    **2. Net Income (กำไรสุทธิ)**
                    * 💵 รายได้หลังหักต้นทุน+ภาษีแล้ว
                    * **ความหมาย:** เงินที่เหลือเข้ากระเป๋าจริง
                    * **ดีคือ:** เป็นบวก (+) และโตตามรายได้
                    """)
                with c_t3:
                    st.info(f"""
                    **3. Cash Flow (กระแสเงินสด)**
                    * 💧 เงินสดที่หมุนเวียนจริงในธุรกิจ
                    * **ที่มา:** มาจากกิจกรรมหลัก (Core Business) ของ {symbol} เช่น การขายสินค้า/บริการ เก็บเงินลูกค้าได้จริง ไม่ใช่แค่ตัวเลขทางบัญชี
                    * **ดีคือ:** ต้องเป็นบวก (+) เสมอ ถ้าติดลบแปลว่าฝืดเคือง
                    """)

                st.markdown("---")
                with st.expander("📄 ดูข้อมูลดิบ (Raw Data Table)"):
                    st.dataframe(fin_df.style.format("{:,.0f}"))

            else:
                if "-USD" in symbol:
                    st.info("ℹ️ สินทรัพย์ประเภท Crypto/Currency ไม่มีงบการเงินให้แสดง")
                else:
                    st.warning("⚠️ ไม่พบข้อมูลงบการเงิน (อาจเป็นหุ้นใหม่ ETF หรือข้อมูลยังไม่มา)")

        with tabs[8]:
            bk_sel = st.radio("เลือกเหรียญ (THB)", ["BTC", "ETH"], horizontal=True)
            if bk_data:
                pair = f"THB_{bk_sel}"
                d = bk_data.get(pair, {})
                if d:
                    last, h24, l24 = d.get('last',0), d.get('high24hr',0), d.get('low24hr',0)
                    ai_bk = calculate_bitkub_ai_levels(h24, l24, last)
                    static_lvls = calculate_static_round_numbers(last)
                    bk_verd, bk_col, bk_desc, bk_strat = analyze_bitkub_static_guru(last, static_lvls)
                    
                    st.markdown(f"""<div class='ai-insight-box' style='text-align:center; border:2px solid {ai_bk['color']}; margin-bottom:20px;'><div style='font-size:3rem; font-weight:900; color:#fff;'>{last:,.0f} <span style='font-size:1.5rem;'>THB</span></div><div style='font-size:1.5rem; font-weight:bold; color:{ai_bk['color']}; text-transform:uppercase;'>{ai_bk['status']}</div></div>""", unsafe_allow_html=True)
                    st.markdown("#### 🧠 AI Strategic Support (แผนการรับของ - THB)")
                    bk_strat_levels, bk_step = calculate_strategic_supports(last, None)
                    bk_gap_pct = ((last - bk_strat_levels[0]['price']) / last) * 100
                    st.markdown(f"""<div style="background:rgba(0, 229, 255, 0.1); padding:15px; border-radius:10px; border-left:4px solid #00E5FF; margin-bottom:20px;"><h4 style="margin:0; color:#00E5FF;">💡 AI Strategy Advisor (THB)</h4><p style="margin:5px 0 0 0; color:#ddd;">ราคาปัจจุบันห่างจากแนวรับแรก <b>{bk_gap_pct:.2f}%</b> (Step: {bk_step:,.0f})<br>แนะนำให้แบ่งไม้ซื้อตามระดับแนวรับเพื่อบริหารต้นทุน</p></div>""", unsafe_allow_html=True)
                    for lvl in bk_strat_levels:
                        l_gap = ((last - lvl['price']) / last) * 100
                        is_near = "ใกล้ถึงแล้ว! 🚨" if l_gap < 1.0 else f"อีก {l_gap:.2f}%"
                        st.markdown(f"""<div style="background: linear-gradient(145deg, #1a1a1a, #111); border: 1px solid #333; border-left: 6px solid {lvl['color']}; border-radius: 12px; padding: 15px; margin-bottom: 10px;"><div style="display:flex; justify-content:space-between; align-items:center;"><div><div style="font-size:1rem; font-weight:bold; color:{lvl['color']};">{lvl['name']}</div><div style="font-size:1.6rem; font-weight:900; color:#fff;">{lvl['price']:,.0f}</div></div><div style="text-align:right;"><span style="font-size:0.8rem; color:#888;">{is_near}</span><br><span style="background:{lvl['color']}20; color:{lvl['color']}; padding:3px 10px; border-radius:10px; font-weight:bold; font-size:0.8rem;">{lvl['alloc']}</span></div></div></div>""", unsafe_allow_html=True)
                    st.markdown("---")
                    div_s1, div_s2 = st.columns(2)
                    with div_s1:
                        st.markdown("#### 🧱 Static S/R")
                        st.markdown(f"<div class='static-card'><span class='static-label'>Res 1</span><span class='static-val' style='color:#FF5252'>{static_lvls['Res 1']:,.0f}</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='static-card'><span class='static-label'>Sup 1</span><span class='static-val' style='color:#69F0AE'>{static_lvls['Sup 1']:,.0f}</span></div>", unsafe_allow_html=True)
                    with div_s2:
                        st.markdown("#### 🤖 Intraday")
                        st.markdown(f"<div class='sr-card sr-res'><b>R1</b><span>{ai_bk['levels'][1]['price']:,.0f}</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='sr-card sr-sup'><b>S1</b><span>{ai_bk['levels'][3]['price']:,.0f}</span></div>", unsafe_allow_html=True)
                    with st.expander("ℹ️ Bitkub Golden Zone"): st.info(f"**Zone:** {ai_bk['fib']['bot']:,.0f} - {ai_bk['fib']['top']:,.0f}")
                else: st.error("ไม่พบข้อมูล")
            else: st.warning("Connecting...")
        
        with tabs[9]:
            st.markdown("### 🧮 Money Management (คำนวณไม้เทรด)")
            col_calc1, col_calc2 = st.columns(2)
            with col_calc1:
                balance = st.number_input("💰 เงินทุนในพอร์ต (Portfolio Size)", value=100000.0, step=1000.0)
                risk_pct = st.number_input("⚠️ ความเสี่ยงที่รับได้ (%)", value=1.0, step=0.1, max_value=100.0)
            with col_calc2:
                def_entry = setup['entry'] if setup else curr
                def_sl = setup['sl'] if setup else curr*0.95
                entry_price = st.number_input("🎯 ราคาเข้าซื้อ (Entry Price)", value=def_entry)
                stop_loss = st.number_input("🛑 จุดตัดขาดทุน (Stop Loss)", value=def_sl)

            if st.button("🧮 คำนวณเดี๋ยวนี้ (Calculate)", use_container_width=True):
                if entry_price > 0 and stop_loss > 0:
                    risk_per_share = abs(entry_price - stop_loss)
                    risk_amount = balance * (risk_pct / 100)
                    if risk_per_share > 0:
                        position_size = risk_amount / risk_per_share
                        total_cost = position_size * entry_price
                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f"<div class='metric-box' style='border-left-color:#00E5FF'><div class='metric-label'>จำนวนหุ้น/เหรียญ</div><div class='metric-val'>{position_size:,.2f}</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='metric-box' style='border-left-color:#FFD600'><div class='metric-label'>เงินลงทุน (Cost)</div><div class='metric-val'>{total_cost:,.2f}</div></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='metric-box' style='border-left-color:#FF1744'><div class='metric-label'>ความเสี่ยง (Risk)</div><div class='metric-val'>{risk_amount:,.2f}</div></div>", unsafe_allow_html=True)
                        st.info(f"💡 แผนการเทรด: คุณจะซื้อจำนวน **{position_size:,.2f} หน่วย** ใช้เงิน **{total_cost:,.2f} บาท** \n\nหากราคาชน Stop Loss คุณจะขาดทุนเพียง **{risk_amount:,.2f} บาท** ({risk_pct}% ของพอร์ต) ซึ่งอยู่ในแผนที่วางไว้")
                    else: st.error("⚠️ ราคาเข้าซื้อต้องไม่เท่ากับราคา Stop Loss")
                else: st.error("⚠️ กรุณากรอกราคาให้ถูกต้อง")

    else: st.error("❌ ไม่พบข้อมูลหุ้น/เหรียญนี้")
