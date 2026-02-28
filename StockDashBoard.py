import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, send_file
import os

app = Flask(__name__)

MAX_THREADS = 4
LAST_RESULTS = pd.DataFrame()  # for CSV export


# =========================================
# GET NIFTY 200
# =========================================
def get_nifty200():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(StringIO(r.text))
        return [s + ".NS" for s in df["Symbol"].tolist()]
    except Exception as e:
        print("NSE Error:", e)
        return []


# =========================================
# TRADINGVIEW LINK
# =========================================
def tradingview_link(symbol):
    tv_symbol = symbol.replace(".NS", "")
    return f"https://www.tradingview.com/chart/?symbol=NSE:{tv_symbol}"


# =========================================
# SAFE COLUMN EXTRACTION
# =========================================
def extract_series(df, column):
    data = df[column]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


# =========================================
# SCAN STOCK
# =========================================
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

        entry_date = None
        exit_date = None
        entry_price = None
        exit_price = None
        status = "Waiting"
        return_pct = 0

        for j in range(last_signal+1, len(df)):

            if close.iloc[j] > bottom_close:

                entry_date = df.index[j]
                entry_price = round(close.iloc[j], 2)

                stop_loss = bottom_low * (1 - sl_pct/100)
                take_profit = entry_price * (1 + tp_pct/100)

                for k in range(j+1, len(df)):
                    if high.iloc[k] >= take_profit:
                        status = "WIN"
                        return_pct = tp_pct
                        exit_date = df.index[k]
                        exit_price = round(take_profit, 2)
                        break

                    if low.iloc[k] <= stop_loss:
                        status = "LOSS"
                        return_pct = round(((stop_loss-entry_price)/entry_price)*100,2)
                        exit_date = df.index[k]
                        exit_price = round(stop_loss, 2)
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

    except Exception as e:
        print("Error:", symbol, e)
        return None


# =========================================
# MAIN ROUTE
# =========================================
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

    # Portfolio stats
    total = len(df)
    wins = len(df[df["Status"]=="WIN"])
    losses = len(df[df["Status"]=="LOSS"])
    win_rate = round((wins/(wins+losses))*100,2) if (wins+losses)>0 else 0

    capital = 100000
    for r in df[df["Status"].isin(["WIN","LOSS"])]["Return %"]:
        capital *= (1 + r/100)

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
            <td><a href="{row['Chart']}" target="_blank">Open Chart</a></td>
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
    Total Signals: {total} |
    Wins: {wins} |
    Losses: {losses} |
    Win Rate: {win_rate}% |
    Portfolio (Start 100k): ₹{round(capital,2)}
    </p>

    <a href="/download">Download CSV</a>

    <table>
    <tr>
        <th>Stock</th>
        <th>Bottom Date</th>
        <th>Entry Date</th>
        <th>Exit Date</th>
        <th>Entry</th>
        <th>Exit</th>
        <th>Status</th>
        <th>Return %</th>
        <th>Chart</th>
    </tr>
    {rows}
    </table>

    </body>
    </html>
    """


# =========================================
# CSV DOWNLOAD
# =========================================
@app.route("/download")
def download():
    global LAST_RESULTS
    if LAST_RESULTS.empty:
        return "No data to download."
    file_path = "results.csv"
    LAST_RESULTS.to_csv(file_path, index=False)
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)