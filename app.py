import os
import pandas as pd
import yfinance as yf
import requests
import matplotlib
matplotlib.use("Agg")   # Important for Flask (no GUI)
import matplotlib.pyplot as plt
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, send_file, render_template



app = Flask(__name__)
MAX_THREADS = 6
LAST_RESULTS = pd.DataFrame()

def get_float_arg(name, default):
    value = request.args.get(name)
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def get_params():
    return {
        "vol": get_float_arg("vol", 1.8),
        "tp": get_float_arg("tp", 20.0),
        "sl": get_float_arg("sl", 5.0),
        "lock_pct": get_float_arg("lock_pct", 5.0),
        "rsi_limit": get_float_arg("rsi_value", 35.0)  # <-- ADD THIS
    }



print("Calling performance")

if not os.path.exists("static"):
    os.makedirs("static")

# =========================================================
# GET INDEX SYMBOL LIST
# =========================================================
def get_nifty200():
    url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    df = pd.read_csv(StringIO(r.text))
    return [s + ".NS" for s in df["Symbol"].tolist()]

def get_midcap():
    url = "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    df = pd.read_csv(StringIO(r.text))
    return [s + ".NS" for s in df["Symbol"].tolist()]

def get_smallcap():
    url = "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    df = pd.read_csv(StringIO(r.text))
    return [s + ".NS" for s in df["Symbol"].tolist()]

def get_sp500():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        r = requests.get(url, timeout=10)
        df = pd.read_csv(StringIO(r.text))
        symbols = df["Symbol"].tolist()

        # Convert BRK.B → BRK-B
        symbols = [s.replace(".", "-") for s in symbols]

        return symbols
    except:
        return []



# =========================================================
# SAFE COLUMN EXTRACTION
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

    if symbol.endswith(".NS"):
        tv_symbol = f"NSE:{symbol.replace('.NS','')}"
    else:
        tv_symbol = symbol   # TradingView auto-detects US ticker

    return f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

def scan_stock(symbol, vol_mult, tp_pct, sl_pct,
               use_ema20, use_ema50,
               ema20_no_touch, ema50_no_touch,
               use_rsi, rsi_limit,
               lock_profit, lock_pct):

    try:
        df = yf.download(symbol, period="2y", interval="1d",
                         auto_adjust=False, threads=False, progress=False)

        if df.empty or len(df) < 200:
            return []

        df = df.dropna()

        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, level=1, axis=1)

        close = df["Close"]
        open_ = df["Open"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()

        # Correct volume average (previous 20 days only)
        avg_vol20 = volume.rolling(20).mean().shift(1)

        # RSI (Wilder)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        trades = []
        i = 20

        while i < len(df) - 2:

            # ======================
            # SAFE VOLUME FILTER
            # ======================
            if pd.isna(avg_vol20.iloc[i]) or pd.isna(volume.iloc[i]):
                i += 1
                continue

            if volume.iloc[i] <= vol_mult * avg_vol20.iloc[i]:
                i += 1
                continue

            # ======================
            # EMA FILTERS
            # ======================
            if use_ema20 and close.iloc[i] >= ema20.iloc[i]:
                i += 1
                continue

            if use_ema50 and close.iloc[i] >= ema50.iloc[i]:
                i += 1
                continue

            if ema20_no_touch and use_ema20:
                if low.iloc[i] <= ema20.iloc[i] <= high.iloc[i]:
                    i += 1
                    continue

            if ema50_no_touch and use_ema50:
                if low.iloc[i] <= ema50.iloc[i] <= high.iloc[i]:
                    i += 1
                    continue

            # ======================
            # RSI FILTER (ENTRY DAY)
            # ======================
            entry_index = i + 1

            if entry_index >= len(df):
                i += 1
                continue

            if use_rsi:
                entry_rsi = rsi.iloc[entry_index]
                if pd.isna(entry_rsi) or entry_rsi < rsi_limit:
                    i += 1
                    continue

            # ======================
            # ENTRY
            # ======================
            entry_price = open_.iloc[entry_index]
            entry_date = df.index[entry_index]

            stop_loss = entry_price * (1 - sl_pct / 100)
            

            status = "Active"
            exit_price = None
            exit_date = None
            return_pct = 0

            k = entry_index + 1
            locked = False

            while k < len(df):

                # ======================
                # LOCK PROFIT LOGIC
                # ======================
                if lock_profit and not locked and high.iloc[k] >= entry_price * (1 + lock_pct / 100):
                    stop_loss = entry_price * (1 + lock_pct / 100)
                    locked = True

                # ======================
                # STOP LOSS (INITIAL OR LOCKED)
                # ======================
                if low.iloc[k] <= stop_loss:
                    exit_price = stop_loss
                    exit_date = df.index[k]

                    if locked:
                        status = "WIN"
                        return_pct = round(
                            ((exit_price - entry_price) / entry_price) * 100, 2
                        )
                    else:
                        status = "LOSS"
                        return_pct = -sl_pct

                    break

                k += 1

            if status == "Active":
                current_price = close.iloc[-1]
                return_pct = round(
                    ((current_price - entry_price) / entry_price) * 100, 2
                )

            trades.append({
                "Stock": symbol,
                "Bottom Date": df.index[i].strftime("%Y-%m-%d"),
                "Entry Date": entry_date.strftime("%Y-%m-%d"),
                "Entry Price": round(entry_price, 2),
                "Exit Date": exit_date.strftime("%Y-%m-%d") if exit_date else "-",
                "Exit Price": round(exit_price, 2) if exit_price else "-",
                "Status": status,
                "Return %": return_pct,
                "Chart": tradingview_link(symbol)
            })

            # Move pointer after exit
            i = k

        return trades

    except Exception as e:
        print("ERROR:", symbol, e)
        return []

# =========================================================
# PERFORMANCE
# =========================================================
def calculate_performance_NEW(df):

    base = {
        "total": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "expectancy": 0,
        "profit_factor": 0,
        "total_profit": 0,
        "total_loss": 0
    }

    if df.empty:
        return base

    closed = df[df["Status"].isin(["WIN", "LOSS"])].copy()

    if closed.empty:
        base["total"] = len(df)
        return base

    # Convert numeric
    closed["Entry Price"] = pd.to_numeric(closed["Entry Price"], errors="coerce")
    closed["Exit Price"] = pd.to_numeric(closed["Exit Price"], errors="coerce")
    closed["Return %"] = pd.to_numeric(closed["Return %"], errors="coerce")
    closed["Entry Date"] = pd.to_datetime(closed["Entry Date"])

    # Raw P/L
    closed["Raw PL"] = closed["Exit Price"] - closed["Entry Price"]

    profit_sum = closed.loc[closed["Raw PL"] > 0, "Raw PL"].sum()
    loss_sum = closed.loc[closed["Raw PL"] < 0, "Raw PL"].sum()

    wins = len(closed[closed["Status"] == "WIN"])
    losses = len(closed[closed["Status"] == "LOSS"])

    base["total"] = len(df)
    base["wins"] = wins
    base["losses"] = losses
    base["win_rate"] = round((wins / (wins + losses)) * 100, 2) if (wins + losses) > 0 else 0

    avg_win = closed.loc[closed["Return %"] > 0, "Return %"].mean()
    avg_loss = closed.loc[closed["Return %"] < 0, "Return %"].mean()

    base["avg_win"] = round(avg_win, 2) if pd.notna(avg_win) else 0
    base["avg_loss"] = round(avg_loss, 2) if pd.notna(avg_loss) else 0

    base["expectancy"] = round(
        (base["win_rate"] / 100) * base["avg_win"]
        - (1 - base["win_rate"] / 100) * abs(base["avg_loss"]),
        2
    )

    if loss_sum != 0:
        base["profit_factor"] = round(profit_sum / abs(loss_sum), 2)

    base["total_profit"] = round(float(profit_sum), 2)
    base["total_loss"] = round(float(loss_sum), 2)

    # ===============================
    # Chronological Equity Curve
    # ===============================
    closed = closed.sort_values("Entry Date")

    capital = 100000
    equity_values = []
    equity_dates = []

    for _, row in closed.iterrows():
        capital = capital * (1 + row["Return %"] / 100)
        equity_values.append(capital)
        equity_dates.append(row["Entry Date"])

    plt.figure(figsize=(10,5))
    plt.plot(equity_dates, equity_values)
    plt.title("Trade-by-Trade Equity Curve")
    plt.xlabel("Trade Entry Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/equity_curve.png")
    plt.close()

    return base

# =========================================================
# MAIN ROUTE
# =========================================================
@app.route("/", methods=["GET"])
def home():

    global LAST_RESULTS

    # First load (no params)
    if "vol" not in request.args:
        return render_template(
            "dashboard.html",
            perf=None,
            rows=[],
            message=None,
            vol=1.8,
            tp=20,
            sl=5,
            use_ema20=True,
            use_ema50=True,
            ema20_no_touch=False,
            ema50_no_touch=True,
            selected_index="nifty200",
            lock_profit=False,
            lock_pct=5
        )



    # -----------------------------
    # Load Parameters Safely
    # -----------------------------
    params = get_params()

    vol = params["vol"]
    tp = params["tp"]
    sl = params["sl"]
    lock_pct = params["lock_pct"]
    rsi_limit = params["rsi_limit"]
    use_rsi = "rsi" in request.args

    lock_profit = request.args.get("lock_profit") == "on"

    use_ema20 = request.args.get("ema20", "on") == "on"
    use_ema50 = request.args.get("ema50", "on") == "on"
    ema20_no_touch = request.args.get("ema20touch", "off") == "on"
    ema50_no_touch = request.args.get("ema50touch", "on") == "on"

    use_rsi = request.args.get("rsi", "off") == "on"

    selected_index = request.args.get("index", "nifty200")

    # -----------------------------
    # Select Symbols
    # -----------------------------
    if selected_index == "midcap":
        symbols = get_midcap()
    elif selected_index == "smallcap":
        symbols = get_smallcap()
    elif selected_index == "sp500":
        symbols = get_sp500()
    else:
        symbols = get_nifty200()

    # -----------------------------
    # Run Strategy
    # -----------------------------
    results = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [
            executor.submit(
                scan_stock,
                s,
                vol,
                tp,
                sl,
                use_ema20,
                use_ema50,
                ema20_no_touch,
                ema50_no_touch,
                use_rsi,
                rsi_limit,
                lock_profit,
                lock_pct
            )
            for s in symbols
        ]

        for f in as_completed(futures):
            r = f.result()
            if r:
                results.extend(r)

    df = pd.DataFrame(results)
    LAST_RESULTS = df

    # -----------------------------
    # If No Results
    # -----------------------------
    if df.empty:
        return render_template(
            "dashboard.html",
            perf=None,
            rows=[],
            message="No stocks found with current criteria.",
            vol=vol,
            tp=tp,
            sl=sl,
            use_ema20=use_ema20,
            use_ema50=use_ema50,
            ema20_no_touch=ema20_no_touch,
            ema50_no_touch=ema50_no_touch,
            use_rsi=use_rsi,
            rsi_limit=rsi_limit,
            selected_index=selected_index,
            lock_profit=lock_profit,
            lock_pct=lock_pct
        )

    # -----------------------------
    # Sort Data
    # -----------------------------
    df["Entry Date Sort"] = pd.to_datetime(df["Entry Date"], errors="coerce")
    df = df.sort_values(by="Entry Date Sort", ascending=False)
    df = df.drop(columns=["Entry Date Sort"])

    perf = calculate_performance_NEW(df)

    # -----------------------------
    # Render Final Dashboard
    # -----------------------------
    return render_template(
        "dashboard.html",
        perf=perf,
        rows=df.to_dict(orient="records"),
        vol=vol,
        tp=tp,
        sl=sl,
        use_ema20=use_ema20,
        use_ema50=use_ema50,
        ema20_no_touch=ema20_no_touch,
        ema50_no_touch=ema50_no_touch,
        use_rsi=use_rsi,
        rsi_limit=rsi_limit,
        selected_index=selected_index,
        lock_profit=lock_profit,
        lock_pct=lock_pct
    )
@app.route("/download")
def download():
    global LAST_RESULTS
    if LAST_RESULTS.empty:
        return "No data."
    LAST_RESULTS.to_csv("results.csv", index=False)
    return send_file("results.csv", as_attachment=True)

if __name__ == "__main__":
    app.run()