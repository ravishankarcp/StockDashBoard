import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from io import StringIO
from datetime import datetime
import webbrowser
import matplotlib
matplotlib.use("Agg")  # 🔥 Fix threading warning
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# PARAMETERS
# ==============================
VOL_MULTIPLIER = 3   # realistic for large caps
TP_PERCENT = 0.20
SL_BUFFER = 0.05
MAX_THREADS = 5

CHART_FOLDER = "charts"
os.makedirs(CHART_FOLDER, exist_ok=True)

# ==============================
# GET NIFTY 200 LIST
# ==============================
def get_nifty200():
    url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    df = pd.read_csv(StringIO(r.text))
    return [symbol + ".NS" for symbol in df["Symbol"].tolist()]

# ==============================
# SAVE CHART IMAGE
# ==============================
def save_chart(symbol, df, bottom_date):
    plt.figure(figsize=(6,4))
    plt.plot(df.index, df["Close"], label="Close")
    plt.axvline(bottom_date, color="red", linestyle="--", label="Bottom")
    plt.title(symbol)
    plt.legend()
    filepath = f"{CHART_FOLDER}/{symbol}.png"
    plt.savefig(filepath)
    plt.close()
    return filepath

# ==============================
# SCAN SINGLE STOCK
# ==============================
def scan_stock(symbol):

    try:
        df = yf.download(
            symbol,
            period="2y",
            interval="1d",
            auto_adjust=False,
            threads=False,
            progress=False
        )

        if len(df) < 200:
            return None

        # 🔥 Fix MultiIndex columns issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["AVG_VOL20"] = df["Volume"].rolling(20).mean()

        last_signal_index = None

        for i in range(21, len(df)-1):

            avg_vol = df["AVG_VOL20"].iloc[i-1]
            if pd.isna(avg_vol):
                continue

            vol_condition = df["Volume"].iloc[i] > VOL_MULTIPLIER * avg_vol
            ema_condition = df["Close"].iloc[i] < df["EMA20"].iloc[i] and df["Close"].iloc[i] < df["EMA50"].iloc[i]

            if vol_condition and ema_condition:
                last_signal_index = i

        if last_signal_index is None:
            return None

        bottom_close = df["Close"].iloc[last_signal_index]
        bottom_low = df["Low"].iloc[last_signal_index]
        bottom_date = df.index[last_signal_index]
        current_close = df["Close"].iloc[-1]

        days_since_bottom = (datetime.now() - bottom_date).days
        distance_from_breakout = round(((current_close - bottom_close) / bottom_close) * 100, 2)

        status = "Waiting"
        entry_date = "-"
        exit_date = "-"
        return_pct = "-"

        # Breakout logic
        for j in range(last_signal_index+1, len(df)):
            if df["Close"].iloc[j] > bottom_close:

                entry_price = df["Close"].iloc[j]
                entry_date = df.index[j]

                stop_loss = bottom_low * (1 - SL_BUFFER)
                take_profit = entry_price * (1 + TP_PERCENT)

                for k in range(j+1, len(df)):

                    if df["High"].iloc[k] >= take_profit:
                        status = "WIN"
                        return_pct = 20
                        exit_date = df.index[k]
                        break

                    if df["Low"].iloc[k] <= stop_loss:
                        status = "LOSS"
                        return_pct = round(((stop_loss-entry_price)/entry_price)*100,2)
                        exit_date = df.index[k]
                        break

                if status not in ["WIN","LOSS"]:
                    status = "Active"
                    return_pct = round(((current_close-entry_price)/entry_price)*100,2)

                break

        chart_path = save_chart(symbol, df.tail(120), bottom_date)

        return {
            "Stock": symbol,
            "Bottom Date": bottom_date.date(),
            "Days Since Bottom": days_since_bottom,
            "Distance From Breakout %": distance_from_breakout,
            "Status": status,
            "Return %": return_pct,
            "Chart": chart_path
        }

    except Exception as e:
        return None

# ==============================
# RUN SCAN
# ==============================
def run():

    print("Fetching NIFTY 200 list...")
    symbols = get_nifty200()

    results = []

    print("Scanning stocks...")

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(scan_stock, s) for s in symbols]

        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    if not results:
        print("No signals found.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="Bottom Date", ascending=False)

    total = len(df)
    wins = len(df[df["Status"] == "WIN"])
    losses = len(df[df["Status"] == "LOSS"])
    win_rate = round((wins / (wins + losses)) * 100, 2) if (wins+losses)>0 else 0

    rows = ""
    for _, row in df.iterrows():
        rows += f"""
        <tr>
            <td>{row['Stock']}</td>
            <td>{row['Bottom Date']}</td>
            <td>{row['Days Since Bottom']}</td>
            <td>{row['Distance From Breakout %']}</td>
            <td>{row['Status']}</td>
            <td>{row['Return %']}</td>
            <td><img src="{row['Chart']}" width="200"></td>
        </tr>
        """

    html = f"""
    <html>
    <head>
    <title>NIFTY 200 Bottom Scanner</title>
    <style>
    body {{ font-family: Arial; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
    th {{ background-color: #222; color: white; }}
    </style>
    </head>
    <body>
    <h2>NIFTY 200 Bottom Scanner</h2>
    <p>Total Signals: {total} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate}%</p>
    <table>
    <tr>
        <th>Stock</th>
        <th>Bottom Date</th>
        <th>Days Since Bottom</th>
        <th>Distance From Breakout %</th>
        <th>Status</th>
        <th>Return %</th>
        <th>Chart</th>
    </tr>
    {rows}
    </table>
    </body>
    </html>
    """

    with open("report.html", "w") as f:
        f.write(html)

    print("Report generated.")
    webbrowser.open("report.html")

if __name__ == "__main__":
    run()