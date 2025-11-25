if df.empty:
    st.error(f"❌ ไม่พบข้อมูล {symbol}")
else:
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
    entries = calculate_tiered_entries(df, sr_levels)

    st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-top: 4px solid {'#00E676' if change>=0 else '#FF1744'};">
            <h1 style="margin:0; font-size: 3rem;">{symbol}</h1>
            <h2 style="margin:0; font-size: 4rem; color: {'#00E676' if change>=0 else '#FF1744'};">{curr_price:,.2f}</h2>
            <p style="font-size: 1.5rem; color: #aaa;">{change:+,.2f} ({pct:+.2f}%)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs (Added Tab 6)
    t1, t2, t3, t4, t5, t6 = st.tabs(["📈 Smart Chart", "🛡️ S/R Levels", "🎯 Smart Trade Setup", "📊 Fundamentals", "🧠 AI Sentiment", "💰 AI Entry Strategy"])
    
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
            with c_en: st.markdown(f"<div class='setup-box'><div class='setup-label'>🔵 ENTRY PRICE</div><div class='setup-val' style='color:#2979FF'>{setup['entry']:,.2f}</div><div style='font-size:0.8rem; color:#666;'>Current Market Price</div></div>", unsafe_allow_html=True)
            with c_sl: st.markdown(f"<div class='setup-box'><div class='setup-label'>🔴 STOP LOSS</div><div class='setup-val' style='color:#FF1744'>{setup['sl']:,.2f}</div><div style='font-size:0.8rem; color:#666;'>Risk Based on ATR ({setup['atr']:,.2f})</div></div>", unsafe_allow_html=True)
            with c_tp: st.markdown(f"<div class='setup-box'><div class='setup-label'>🟢 TAKE PROFIT</div><div class='setup-val' style='color:#00E676'>{setup['tp']:,.2f}</div><div style='font-size:0.8rem; color:#666;'>Reward Ratio 1:1.5+</div></div>", unsafe_allow_html=True)

    with t4:
        info = ticker.info
        st.markdown("<div class='section-header'>📊 Fundamentals Analysis</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f"""<div class='fund-card' style='border-top: 3px solid #2979FF;'><div class='fund-lbl'>Market Cap</div><div class='fund-val' style='color:#2979FF'>{info.get('marketCap',0):,}</div></div>""", unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class='fund-card' style='border-top: 3px solid #AB47BC;'><div class='fund-lbl'>P/E Ratio</div><div class='fund-val' style='color:#AB47BC'>{info.get('trailingPE',0):.2f}</div></div>""", unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class='fund-card' style='border-top: 3px solid #00E676;'><div class='fund-lbl'>52 Week High</div><div class='fund-val' style='color:#00E676'>{info.get('fiftyTwoWeekHigh',0):,.2f}</div></div>""", unsafe_allow_html=True)
        with c4: st.markdown(f"""<div class='fund-card' style='border-top: 3px solid #FF1744;'><div class='fund-lbl'>52 Week Low</div><div class='fund-val' style='color:#FF1744'>{info.get('fiftyTwoWeekLow',0):,.2f}</div></div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.info(f"ℹ️ **Business Summary:** {info.get('longBusinessSummary', 'No description available.')[:600]}...")

    with t5:
        st.markdown("<div class='section-header'>🧠 AI Sentiment Analysis (Thai)</div>", unsafe_allow_html=True)
        raw_news = get_bloomberg_news(symbol)
        if not raw_news: st.warning("ไม่พบข่าวในขณะนี้")
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
            with c1: st.markdown(f"""<div class="sent-box sb-pos"><div class="sent-box-val">{pos}</div><div class="sent-box-lbl">Positive News (ข่าวดี)</div></div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="sent-box sb-neg"><div class="sent-box-val">{neg}</div><div class="sent-box-lbl">Negative News (ข่าวร้าย)</div></div>""", unsafe_allow_html=True)
            with c3: st.markdown(f"""<div class="sent-box sb-neu"><div class="sent-box-val">{neu}</div><div class="sent-box-lbl">Neutral News (ทั่วไป)</div></div>""", unsafe_allow_html=True)
            st.markdown("---")
            for p in processed:
                st.markdown(f"""<div class="sentiment-card {p['css']}"><div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;"><span class="{p['badge']}">{p['icon']} {p['cat']}</span><span style="color:#666; font-size:0.8rem;">Source: {p['source']}</span></div><div style="font-size:1.1rem; font-weight:bold; color:#fff; margin-bottom:10px;">{p['title']}</div><div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px; color:#ccc; font-size:0.9rem; margin-bottom:10px;">{p['summary']}</div><div style="font-weight:bold; margin-top:5px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.1);">💥 {p['impact']}</div><div style="text-align:right; margin-top:5px;"><a href="{p['link']}" target="_blank" style="color:#aaa; font-size:0.8rem; text-decoration:none;">🔗 อ่านข่าวต้นฉบับ</a></div></div>""", unsafe_allow_html=True)

    # --- TAB 6: AI ENTRY STRATEGY (NEW) ---
    with t6:
        st.markdown("<div class='section-header'>💰 AI Entry Strategy (Money Management)</div>", unsafe_allow_html=True)
        if entries:
            # Tier 1
            st.markdown(f"""
            <div class="entry-card ec-tier1">
                <div class="ec-allocation">Allocation: 20%</div>
                <div class="ec-title-1">🪵 ไม้แรก (Probe Buy)</div>
                <div class="ec-price">{entries['t1']:,.2f}</div>
                <div class="ec-desc">เข้าซื้อเพื่อหยั่งเชิง หรือ Testing Position บริเวณแนวรับแรก หากราคายืนไม่อยู่ให้เตรียมไม้สอง</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tier 2
            st.markdown(f"""
            <div class="entry-card ec-tier2">
                <div class="ec-allocation">Allocation: 30%</div>
                <div class="ec-title-2">🪵 ไม้สอง (Accumulate)</div>
                <div class="ec-price">{entries['t2']:,.2f}</div>
                <div class="ec-desc">จุดเข้าซื้อเพิ่มเมื่อราคาย่อตัวลงมา (Correction) เพื่อถัวเฉลี่ยต้นทุนในจุดที่มีนัยยะสำคัญ</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tier 3
            st.markdown(f"""
            <div class="entry-card ec-tier3">
                <div class="ec-allocation">Allocation: 50%</div>
                <div class="ec-title-3">💎 ไม้หนัก (Strong / Sniper)</div>
                <div class="ec-price">{entries['t3']:,.2f}</div>
                <div class="ec-desc">โซนแนวรับแข็งแกร่งที่สุด หรือ Panic Sell จุดนี้คือจุดที่ควรใช้เงินหนัก (All-in zone) หากพื้นฐานยังดี</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"💡 **AI Note:** แผนนี้คำนวณจากแนวรับ (Support) ร่วมกับค่าความผันผวน (ATR = {entries['atr']:,.2f}) เพื่อกระจายความเสี่ยงอย่างเป็นระบบ")
        else:
            st.warning("ข้อมูลไม่เพียงพอสำหรับคำนวณ Entry Strategy")
