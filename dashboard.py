import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import plotly.graph_objects as go
from datetime import datetime
from config import SIGNALS_CSV, TRADES_CSV, SYMBOL, INITIAL_PAPER_BALANCE

st.set_page_config(
    page_title="ETH Trading Terminal",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Light Paper Minimalist Theme ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Montserrat:wght@600;700;800&display=swap');
    
    /* Main Background with subtle texture */
    .stApp {
        background-color: #e5e7eb;
        background-image: url("https://www.transparenttextures.com/patterns/felt.png");
        color: #1f2937 !important;
    }
    
    /* Typography */
    h1, h2, h3, .stTitle {
        font-family: 'Montserrat', sans-serif !important;
        color: #111827 !important;
        text-shadow: none !important;
        letter-spacing: -0.5px;
    }
    
    body, p, div {
        font-family: 'Inter', sans-serif !important;
        color: #374151 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(243, 244, 246, 0.8) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Metrics Cards - Clean and Minimal */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }

    /* Styled Containers (Custom Metrics) */
    .metric-card {
        background: #ffffff !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        background: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 8px 8px 0px 0px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        color: #4b5563 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        border-bottom: 2px solid #3b82f6 !important;
        color: #111827 !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px !important;
        background-color: #374151 !important;
        color: white !important;
        border: none !important;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1f2937 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    /* Glow bypass for light theme */
    .glow-green, .glow-red, .glow-blue {
        box-shadow: none !important;
        border: 2px solid currentColor !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 TRADING TERMINAL")
st.markdown("<p style='color: #6b7280; font-family: Inter; margin-top: -20px;'>Institutional Grade ETH/USDT Intelligence</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Workspace Loaded ──

# ── Sidebar ──
st.sidebar.image("https://img.icons8.com/isometric/512/cryptocurrency.png", width=100)
st.sidebar.title("⚡ TERMINAL CONTROLS")
st.sidebar.markdown("---")

source = st.sidebar.selectbox("📡 Data Source", ["Live (SQLite)", "Historical (CSV)"])
from config import MOCK_TRADING
if not MOCK_TRADING:
    st.sidebar.error("🚨 LIVE TRADING ACTIVE - REAL MONEY AT RISK")
else:
    st.sidebar.success("💡 PAPER TRADING MODE")
    
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Settings")
auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)

# ── Mission Protocol Selector ──
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 MISSION PROTOCOL")
from utils.settings_manager import load_settings, save_settings
from config import TRADING_MODES

current_settings = load_settings()
selected_mode = st.sidebar.selectbox(
    "Active Protocol", 
    options=list(TRADING_MODES.keys()),
    index=list(TRADING_MODES.keys()).index(current_settings["active_mode"])
)

# Mode Details Card
mode_info = TRADING_MODES[selected_mode]
st.sidebar.markdown(f"""
<div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.1); 
            border-left: 5px solid {mode_info['color']};
            border-radius: 8px; padding: 15px; margin-top: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
    <p style="color: #374151; font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 14px; margin-bottom: 5px;">
        {selected_mode.upper()} STATUS
    </p>
    <p style="color: #6b7280; font-family: 'Inter', sans-serif; font-size: 12px; margin: 0; line-height: 1.4;">
        {mode_info['description']}
    </p>
    <p style="color: {mode_info['color']}; font-family: 'Inter', sans-serif; font-weight: bold; font-size: 11px; margin-top: 8px; opacity: 0.8;">
        SIGNAL THRESHOLD: {mode_info['threshold']:.4f}
    </p>
</div>
""", unsafe_allow_html=True)

if selected_mode != current_settings["active_mode"]:
    current_settings["active_mode"] = selected_mode
    save_settings(current_settings)
    st.sidebar.success(f"Deployed: {selected_mode} Protocol")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ COMMAND BAR")
cmd_input = st.sidebar.text_input("Enter command...", placeholder="CLOSE_ALL, PAUSE, RESUME")
if st.sidebar.button("Execute"):
    if cmd_input:
        from utils.database import save_command
        save_command(cmd_input.upper())
        st.sidebar.success(f"Sent: {cmd_input.upper()}")
    else:
        st.sidebar.error("Empty command.")

# ── Data Loading Functions ──
@st.cache_data(ttl=5)
def load_data(market_source):
    def get_csv_fallback():
        try:
            if os.path.exists(SIGNALS_CSV):
                signals = pd.read_csv(SIGNALS_CSV)
                rename_map = {"candle_close_utc": "timestamp", "close": "close_price"}
                signals = signals.rename(columns=rename_map)
                if "timestamp" not in signals.columns and "open_time" in signals.columns:
                    signals = signals.rename(columns={"open_time": "timestamp"})
            else:
                signals = pd.DataFrame(columns=["timestamp", "prediction", "close_price"])
            
            if os.path.exists(TRADES_CSV):
                trades = pd.read_csv(TRADES_CSV)
            else:
                trades = pd.DataFrame(columns=["timestamp", "symbol", "side", "price", "amount", "status"])
            
            for col in ["timestamp", "prediction", "close_price"]:
                if col not in signals.columns: signals[col] = np.nan
            
            return signals.sort_values('timestamp'), trades, None, 0.0, pd.DataFrame()
        except:
            return pd.DataFrame(columns=["timestamp", "prediction", "close_price"]), pd.DataFrame(), None, 0.0, pd.DataFrame()

    if market_source == "Historical (CSV)":
        return get_csv_fallback()

    balance = 0.0
    try:
        from utils.database import SessionLocal, SignalLog, TradeLog, PortfolioSnapshot
        from live.execution_engine import ExecutionEngine
        
        try:
            engine = ExecutionEngine(mock=MOCK_TRADING)
            balance = engine.get_balance()
        except:
            balance = 0.0

        session = SessionLocal()
        signals = pd.read_sql(session.query(SignalLog).order_by(SignalLog.timestamp.desc()).limit(200).statement, session.bind)
        trades = pd.read_sql(session.query(TradeLog).order_by(TradeLog.timestamp.desc()).limit(200).statement, session.bind)
        
        accuracy = None
        if not signals.empty and 'is_verified' in signals.columns:
            verified = signals[signals['is_verified'] == True]
            if not verified.empty:
                accuracy = verified['was_correct'].mean()
            
        portfolio = pd.read_sql(session.query(PortfolioSnapshot).order_by(PortfolioSnapshot.timestamp.desc()).limit(1000).statement, session.bind)
        
        session.close()
        return signals.sort_values('timestamp'), trades, accuracy, balance, portfolio.sort_values('timestamp')
    except:
        return get_csv_fallback()

@st.cache_data(ttl=1)
def load_market_data():
    from data.fetcher import fetch_latest_candles
    from config import TIMEFRAME
    try:
        df = fetch_latest_candles(SYMBOL, TIMEFRAME, limit=100)
        return df
    except:
        return pd.DataFrame()

# Initial Data Load (needed for sidebar sliders)
df_signals, df_trades, real_accuracy, wallet_balance, df_portfolio = load_data(source)
df_market = load_market_data()

# ── REPLAY MODE (Time Travel) ──
st.sidebar.markdown("---")
st.sidebar.subheader("🎞️ REPLAY MODE")
enable_replay = st.sidebar.toggle("Enable Replay Mode")
replay_index = 0
if enable_replay and not df_signals.empty:
    replay_index = st.sidebar.select_slider(
        "Historical State",
        options=list(range(len(df_signals))),
        value=len(df_signals) - 1,
        label_visibility="collapsed"
    )
    st.sidebar.info(f"Viewing: {df_signals.iloc[replay_index]['timestamp']}")

# Apply Replay Filter
if enable_replay and not df_signals.empty:
    cutoff_time = df_signals.iloc[replay_index]['timestamp']
    df_signals = df_signals.iloc[:replay_index + 1]
    # Filter trades and portfolio based on cutoff
    if not df_trades.empty:
        df_trades = df_trades[df_trades['timestamp'] <= cutoff_time]
    if not df_portfolio.empty:
        df_portfolio = df_portfolio[df_portfolio['timestamp'] <= cutoff_time]

def safe_float(val, default=0.0):
    try:
        import pandas as pd
        if val is None or pd.isna(val): return default
        return float(val)
    except:
        return default

def calculate_pnl_from_df(trades_df):
    r_pnl = 0.0
    o_positions = []
    if not trades_df.empty and 'status' in trades_df.columns:
        for _, trade in trades_df.iterrows():
            if trade['status'] in ['STOP LOSS', 'TAKE PROFIT', 'CLOSED']:
                entry = safe_float(trade.get('entry_price', trade.get('price')))
                exit_price = safe_float(trade.get('exit_price', trade.get('close_price', trade.get('price'))))
                amount = safe_float(trade.get('amount'))
                side = str(trade.get('side', '')).upper()
                side = str(trade.get('side', '')).upper()
                if side in ['LONG', 'BUY']:
                    pnl = (exit_price - entry) * amount
                else:  # SHORT
                    pnl = (entry - exit_price) * amount
                r_pnl += pnl
            elif trade['status'] == 'OPEN':
                o_positions.append(trade)
    return r_pnl, o_positions

realized_pnl, open_positions = calculate_pnl_from_df(df_trades)

# ── 💰 FINANCIAL PERFORMANCE SECTION ─────────────────────────────────────────
st.markdown("### 💰 FINANCIAL PERFORMANCE")

# Calculate metrics
total_trades = len(df_trades) if not df_trades.empty else 0
winning_trades = len(df_trades[df_trades['status'] == 'TAKE PROFIT']) if not df_trades.empty else 0
losing_trades = len(df_trades[df_trades['status'] == 'STOP LOSS']) if not df_trades.empty else 0

# Calculate unrealized PnL from open positions
unrealized_pnl = 0.0
if not df_market.empty:
    current_price = df_market['close'].iloc[-1]
    for pos in open_positions:
        entry = safe_float(pos.get('entry_price', pos.get('price')))
        amount = safe_float(pos.get('amount'))
        side = str(pos.get('side', '')).upper()
        if side in ['LONG', 'BUY']:
            unrealized_pnl += (current_price - entry) * amount
        elif side in ['SHORT', 'SELL']:
            unrealized_pnl += (entry - current_price) * amount

total_pnl = realized_pnl + unrealized_pnl

# Portfolio value
from config import MOCK_TRADING
if not MOCK_TRADING:
    current_balance = wallet_balance
    initial_balance = current_balance  # Quick fix to prevent the UI color logic crash
    pnl_percentage = 0.0  # Live percentage requires continuous deposit tracking
else:
    initial_balance = INITIAL_PAPER_BALANCE
    current_balance = initial_balance + total_pnl
    pnl_percentage = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0

# ── METRICS ROW ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    color = "#059669" if total_pnl >= 0 else "#DC2626"
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6B7280; font-family: 'Inter', sans-serif; margin: 0; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">TOTAL PnL</p>
        <p style="color: {color}; font-family: 'Montserrat', sans-serif; margin: 0; font-size: 24px; font-weight: 800;">
            {total_pnl:+.2f} <span style="font-size: 14px; opacity: 0.7;">USDT</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6B7280; font-family: 'Inter', sans-serif; margin: 0; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">RETURN %</p>
        <p style="color: {color}; font-family: 'Montserrat', sans-serif; margin: 0; font-size: 24px; font-weight: 800;">
            {pnl_percentage:+.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    color_val = "#059669" if current_balance >= initial_balance else "#DC2626"
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6B7280; font-family: 'Inter', sans-serif; margin: 0; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">PORTFOLIO VALUE</p>
        <p style="color: #111827; font-family: 'Montserrat', sans-serif; margin: 0; font-size: 24px; font-weight: 800;">
            ${current_balance:,.0f}
        </p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6B7280; font-family: 'Inter', sans-serif; margin: 0; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">WIN RATE</p>
        <p style="color: #2563EB; font-family: 'Montserrat', sans-serif; margin: 0; font-size: 24px; font-weight: 800;">
            {win_rate:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with c5:
    color_realized = "#059669" if realized_pnl >= 0 else "#DC2626"
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6B7280; font-family: 'Inter', sans-serif; margin: 0; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">REALIZED PnL</p>
        <p style="color: {color_realized}; font-family: 'Montserrat', sans-serif; margin: 0; font-size: 24px; font-weight: 800;">
            {realized_pnl:+.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)

with c6:
    color_unrealized = "#059669" if unrealized_pnl >= 0 else "#DC2626"
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6B7280; font-family: 'Inter', sans-serif; margin: 0; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">UNREALIZED PnL</p>
        <p style="color: {color_unrealized}; font-family: 'Montserrat', sans-serif; margin: 0; font-size: 24px; font-weight: 800;">
            {unrealized_pnl:+.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Secondary Metrics ──────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("Total Trades", total_trades)
with s2:
    st.metric("Winning Trades", winning_trades, delta=winning_trades-losing_trades, delta_color="normal")
with s3:
    st.metric("Losing Trades", losing_trades)
with s4:
    avg_win = abs(realized_pnl / winning_trades) if winning_trades > 0 else 0
    st.metric("Avg Win Size", f"${avg_win:.2f}")

# ── Live Market Chart ───────────────────────────────────────────────────────
st.subheader(f"📈 {SYMBOL} Live Market Data")
if not df_market.empty:
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_market.index,
        open=df_market['open'],
        high=df_market['high'],
        low=df_market['low'],
        close=df_market['close'],
        increasing_line_color='#10B981', 
        decreasing_line_color='#EF4444'
    )])
    fig_candle.update_layout(
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.5)"
    )
    st.plotly_chart(fig_candle, use_container_width=True)
else:
    st.warning("Could not fetch live market data. Check your internet connection.")

# ── Analytics Tabs ────────────────────────────────────────────────────────────
st.subheader("📊 Market Intelligence & Analytics")
m1, m2 = st.columns([2, 1])

with m1:
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Prediction History", "🎯 Precision Check", "💹 Equity Curve", "⚠️ Risk Analytics"])
    with tab1:
        if not df_signals.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df_signals['timestamp'], format='ISO8601'), 
                y=df_signals["prediction"],
                mode='lines',
                name='Model Prediction',
                line=dict(color='#3B82F6', width=2),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=20, b=20),
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'is_verified' in df_signals.columns:
            verified = df_signals[df_signals['is_verified'] == True]
            if not verified.empty:
                import plotly.express as px
                fig_scatter = px.scatter(
                    verified, x="prediction", y="actual_return",
                    color="was_correct",
                    color_discrete_map={True: "#10B981", False: "#EF4444"},
                    title="Prediction vs Reality"
                )
                fig_scatter.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Wait for predictions to be verified (takes 4h+)")
        else:
            st.warning("Enable database for precision tracking")

    with tab3:
        if not df_portfolio.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df_portfolio['timestamp']), 
                y=df_portfolio['equity'],
                mode='lines', 
                name='Equity (USDT)', 
                line=dict(color='#10B981', width=3),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))
            fig.update_layout(
                title="Portfolio Equity Curve",
                template="plotly_dark", 
                height=350, 
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="USDT"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No portfolio history yet")

    with tab4:
        if not df_portfolio.empty:
            returns = df_portfolio['equity'].pct_change().dropna()
            rolling_max = df_portfolio['equity'].cummax()
            drawdown = (df_portfolio['equity'] - rolling_max) / rolling_max
            max_dd = drawdown.min()
            sharpe = (returns.mean() / returns.std() * np.sqrt(365 * 24)) if not returns.empty else 0.0

            col1, col2, col3 = st.columns(3)
            col1.metric("Max Drawdown", f"{max_dd*100:.2f}%", delta_color="inverse")
            col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col3.metric("Win Rate", f"{(winning_trades/total_trades*100):.1f}%" if total_trades > 0 else "N/A")

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=pd.to_datetime(df_portfolio['timestamp']),
                y=drawdown * 100,
                fill='tozeroy',
                name='Drawdown %',
                line=dict(color='#EF4444')
            ))
            fig_dd.update_layout(title="Drawdown History (%)", template="plotly_dark", height=250)
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.info("Insufficient data for risk analytics")

with m2:
    import plotly.express as px
    heatmap_data = np.random.randn(5, 5)
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Asset", y="Asset", color="Corr"),
        x=['ETH', 'BTC', 'SOL', 'BNB', 'LINK'],
        y=['ETH', 'BTC', 'SOL', 'BNB', 'LINK'],
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    fig_heat.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Data Tables ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🛡️ Execution Log")
    if not df_trades.empty:
        # Reorder and format columns for better readability
        display_cols = ['timestamp', 'symbol', 'side', 'price', 'tp_sl', 'amount', 'status']
        available_cols = [c for c in display_cols if c in df_trades.columns]
        st.dataframe(df_trades[available_cols].tail(15), use_container_width=True)
    else:
        st.dataframe(df_trades.tail(15), use_container_width=True)

with col_right:
    st.subheader("📡 Signal Stream")
    st.dataframe(df_signals.tail(15), use_container_width=True)

if auto_refresh:
    time.sleep(5)
    st.rerun()
