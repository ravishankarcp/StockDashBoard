import os
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, send_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

MAX_THREADS = 4
LAST_RESULTS = pd.DataFrame()

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# =========================================================
# GET NIFTY 200 LIST
# =========================================================
def get_nifty200():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(StringIO(r.text))
        return [s + ".NS" for s in df["Symbol"].tolist()]
    except:
        return []

# =========================================================
# SAFE COLUMN EXTRACTION (Fix Yahoo MultiIndex)
# =========================================================
def extract_series(df, column):
    data = df[column]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data

# =========================================================
# TRADINGVIEW LINK
# =========================================================
def tradingview_link(symbol):
    return f"https://www.tradingview.com/chart/?symbol=NSE:{symbol.replace('.NS','')}"

# =========================================================
# SCAN STOCK
# =========================================================
def scan_stock(symbol, vol_mult, tp_pct, sl_pct,
               use_ema20, use_ema50,
               ema20_no_touch, ema50_no_touch):

    try:
        df = yf.download(symbol, period="2y", interval="1d",
                         auto_adjust=False, threads=False, progress=False)

        if df.empty or len(df) < 200:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        close = extract_series(df, "Close")
        volume = extract_series(df, "Volume")
        high = extract_series(df, "High")
        low = extract_series(df, "Low")

        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        avg_vol20 = volume.rolling(20).mean()

        last_signal = None

        for i in range(21, len(df)-1):

            if pd.isna(avg_vol20.iloc[i-1]):
                continue

            if volume.iloc[i] <= vol_mult * avg_vol20.iloc[i-1]:
                continue

            if use_ema20 and close.iloc[i] >= ema20.iloc[i]:
                continue

            if use_ema50 and close.iloc[i] >= ema50.iloc[i]:
                continue

            if ema20_no_touch and use_ema20:
                if low.iloc[i] <= ema20.iloc[i] <= high.iloc[i]:
                    continue

            if ema50_no_touch and use_ema50:
                if low.iloc[i] <= ema50.iloc[i] <= high.iloc[i]:
                    continue

            last_signal = i

        if last_signal is None:
            return None

        bottom_close = close.iloc[last_signal]
        bottom_low = low.iloc[last_signal]
        bottom_date = df.index[last_signal]

        entry_price = None
        exit_price = None
        entry_date = None
        exit_date = None
        status = "Waiting"
        return_pct = 0

        for j in range(last_signal+1, len(df)):
            if close.iloc[j] > bottom_close:

                entry_price = round(close.iloc[j],2)
                entry_date = df.index[j]

                stop_loss = entry_price * (1 - sl_pct/100)
                take_profit = entry_price * (1 + tp_pct/100)

                for k in range(j+1, len(df)):
                    if high.iloc[k] >= take_profit:
                        status = "WIN"
                        return_pct = tp_pct
                        exit_price = round(take_profit,2)
                        exit_date = df.index[k]
                        break

                    if low.iloc[k] <= stop_loss:
                        status = "LOSS"
                        return_pct = round(((stop_loss-entry_price)/entry_price)*100,2)
                        exit_price = round(stop_loss,2)
                        exit_date = df.index[k]
                        break

                if status not in ["WIN","LOSS"]:
                    status = "Active"
                    return_pct = round(((close.iloc[-1]-entry_price)/entry_price)*100,2)

                break

        return {
            "Stock": symbol,
            "Bottom Date": bottom_date.strftime("%Y-%m-%d"),
            "Entry Date": entry_date.strftime("%Y-%m-%d") if entry_date else "-",
            "Exit Date": exit_date.strftime("%Y-%m-%d") if exit_date else "-",
            "Entry Price": entry_price if entry_price else "-",
            "Exit Price": exit_price if exit_price else "-",
            "Status": status,
            "Return %": return_pct,
            "Chart": tradingview_link(symbol)
        }

    except:
        return None

## =========================================================
# TIME-BASED EQUITY + FULL METRICS
# =========================================================
def calculate_performance(df):

    if df.empty:
        return {}

    df_trades = df[df["Entry Date"] != "-"].copy()

    if df_trades.empty:
        return {}

    df_trades["Entry Date"] = pd.to_datetime(df_trades["Entry Date"])
    df_trades["Exit Date"] = pd.to_datetime(df_trades["Exit Date"], errors="coerce")

    # ---- BASIC METRICS ----
    wins = len(df[df["Status"]=="WIN"])
    losses = len(df[df["Status"]=="LOSS"])
    total = len(df)

    win_rate = round((wins/(wins+losses))*100,2) if (wins+losses)>0 else 0

    closed = df[df["Status"].isin(["WIN","LOSS"])]

    avg_win = closed[closed["Return %"]>0]["Return %"].mean()
    avg_loss = closed[closed["Return %"]<0]["Return %"].mean()

    avg_win = round(avg_win,2) if pd.notna(avg_win) else 0
    avg_loss = round(avg_loss,2) if pd.notna(avg_loss) else 0

    expectancy = round(
        ((win_rate/100)*avg_win) -
        ((1-win_rate/100)*abs(avg_loss)),2
    ) if (wins+losses)>0 else 0

    profit_factor = round(
        closed[closed["Return %"]>0]["Return %"].sum() /
        abs(closed[closed["Return %"]<0]["Return %"].sum()),
        2
    ) if losses>0 else "∞"

    # ---- TIME BASED EQUITY ----
    start_date = df_trades["Entry Date"].min()
    end_date = pd.Timestamp.today().normalize()
    date_range = pd.date_range(start_date, end_date, freq="D")

    initial_capital = 100000
    allocation = initial_capital / len(df_trades)

    equity_curve = []
    equity_dates = []

    for current_date in date_range:

        daily_equity = 0

        for _, trade in df_trades.iterrows():

            entry = trade["Entry Date"]
            exit_ = trade["Exit Date"]
            ret = trade["Return %"]

            if current_date < entry:
                continue

            if pd.isna(exit_) or current_date <= exit_:
                daily_equity += allocation * (1 + ret/100)
            else:
                daily_equity += allocation * (1 + ret/100)

        if daily_equity == 0:
            daily_equity = initial_capital

        equity_curve.append(daily_equity)
        equity_dates.append(current_date)

    final_capital = round(equity_curve[-1],2)

    # ---- MAX DRAWDOWN ----
    peak = equity_curve[0]
    max_dd = 0

    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val)/peak
        max_dd = max(max_dd, dd)

    max_dd = round(max_dd*100,2)

    # ---- SAVE CHART ----
    plt.figure(figsize=(10,5))
    plt.plot(equity_dates, equity_curve)
    plt.title("Equity Curve (Time-Based, Mark-to-Market)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/equity_curve.png")
    plt.close()

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "final_capital": final_capital
    }

# =========================================================
# MAIN ROUTE
# =========================================================
@app.route("/", methods=["GET"])
def home():

    global LAST_RESULTS

    vol = float(request.args.get("vol",1.8))
    tp = float(request.args.get("tp",20))
    sl = float(request.args.get("sl",5))

    use_ema20 = request.args.get("ema20","on") == "on"
    use_ema50 = request.args.get("ema50","on") == "on"
    ema20_no_touch = request.args.get("ema20touch","off") == "on"
    ema50_no_touch = request.args.get("ema50touch","on") == "on"

    symbols = get_nifty200()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(scan_stock, s, vol, tp, sl,
                                   use_ema20, use_ema50,
                                   ema20_no_touch, ema50_no_touch)
                   for s in symbols]

        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    df = pd.DataFrame(results)
    LAST_RESULTS = df

    if df.empty:
        return "No signals found."

    perf = calculate_performance(df)

    rows = ""
    for _, row in df.iterrows():
        rows += f"""
        <tr class="{row['Status']}">
            <td>{row['Stock']}</td>
            <td>{row['Bottom Date']}</td>
            <td>{row['Entry Date']}</td>
            <td>{row['Exit Date']}</td>
            <td>{row['Entry Price']}</td>
            <td>{row['Exit Price']}</td>
            <td>{row['Status']}</td>
            <td>{row['Return %']}</td>
            <td><a href="{row['Chart']}" target="_blank">Chart</a></td>
        </tr>
        """

    return f"""
    <html>
    <head>
    <style>
    body {{ font-family: Arial; padding:20px; }}
    table {{ border-collapse: collapse; width:100%; }}
    th, td {{ border:1px solid #ddd; padding:6px; text-align:center; }}
    th {{ background:black; color:white; }}
    .WIN {{ background:#d4edda; }}
    .LOSS {{ background:#f8d7da; }}
    .Active {{ background:#fff3cd; }}
    </style>
    </head>
    <body>

    <h2>Bottom Scanner</h2>

    <form method="get">
    Volume: <input name="vol" value="{vol}">
    TP %: <input name="tp" value="{tp}">
    SL %: <input name="sl" value="{sl}">
    <br><br>
    EMA20 <input type="checkbox" name="ema20" {"checked" if use_ema20 else ""}>
    EMA50 <input type="checkbox" name="ema50" {"checked" if use_ema50 else ""}>
    20 EMA No Touch <input type="checkbox" name="ema20touch" {"checked" if ema20_no_touch else ""}>
    50 EMA No Touch <input type="checkbox" name="ema50touch" {"checked" if ema50_no_touch else ""}>
    <br><br>
    <button type="submit">Run</button>
    </form>

    <p>
    Total: {perf['total']} |
    Wins: {perf['wins']} |
    Losses: {perf['losses']} |
    Win Rate: {perf['win_rate']}% <br><br>
    Avg Win: {perf['avg_win']}% |
    Avg Loss: {perf['avg_loss']}% |
    Expectancy: {perf['expectancy']}% <br><br>
    Profit Factor: {perf['profit_factor']} |
    Max DD: {perf['max_dd']}% <br><br>
    Final Portfolio (Start 100k): ₹{perf['final_capital']}
    </p>

    <h3>Equity Curve</h3>
    <img src="/static/equity_curve.png" width="700">

    <br><br>
    <a href="/download">Download CSV</a>

    <table>
    <tr>
        <th>Stock</th>
        <th>Bottom</th>
        <th>Entry</th>
        <th>Exit</th>
        <th>Entry Price</th>
        <th>Exit Price</th>
        <th>Status</th>
        <th>Return %</th>
        <th>Chart</th>
    </tr>
    {rows}
    </table>

    </body>
    </html>
    """

@app.route("/download")
def download():
    global LAST_RESULTS
    if LAST_RESULTS.empty:
        return "No data."
    LAST_RESULTS.to_csv("results.csv", index=False)
    return send_file("results.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)