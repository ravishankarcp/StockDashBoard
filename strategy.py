import pandas as pd
import numpy as np

# ==========================
# PARAMETERS
# ==========================
VOL_MULTIPLIER = 5
TP_PERCENT = 0.20
SL_BUFFER = 0.05
LOOKBACK_DAYS = 252

# ==========================
# HELPER FUNCTIONS
# ==========================
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

# ==========================
# MAIN STRATEGY FUNCTION
# ==========================
def run_strategy(universe):

    results = []

    for stock in universe:

        data = get_daily_data(stock, LOOKBACK_DAYS)

        if data is None or len(data) < 60:
            continue

        data["EMA20"] = calculate_ema(data["Close"], 20)
        data["EMA50"] = calculate_ema(data["Close"], 50)
        data["AVG_VOL20"] = calculate_sma(data["Volume"], 20)
        data["PE_AVG_1Y"] = calculate_sma(data["PE"], 252)

        for t in range(20, len(data)-1):

            volume_condition = data.Volume.iloc[t] > VOL_MULTIPLIER * data.AVG_VOL20.iloc[t-1]
            ema_condition = data.Close.iloc[t] < data.EMA20.iloc[t] and data.Close.iloc[t] < data.EMA50.iloc[t]
            pe_condition = data.PE.iloc[t] < data.PE_AVG_1Y.iloc[t]

            if volume_condition and ema_condition and pe_condition:

                bottom_close = data.Close.iloc[t]
                bottom_low = data.Low.iloc[t]
                bottom_date = data.Date.iloc[t]

                # Look for entry
                for k in range(t+1, len(data)):

                    if data.Close.iloc[k] > bottom_close:

                        entry_price = data.Close.iloc[k]
                        entry_date = data.Date.iloc[k]

                        stop_loss = bottom_low * (1 - SL_BUFFER)
                        take_profit = entry_price * (1 + TP_PERCENT)

                        # Forward scan
                        for m in range(k+1, len(data)):

                            if data.High.iloc[m] >= take_profit:
                                results.append({
                                    "Stock": stock,
                                    "BottomDate": bottom_date,
                                    "EntryDate": entry_date,
                                    "ExitDate": data.Date.iloc[m],
                                    "Result": "WIN",
                                    "Return%": 20
                                })
                                break

                            if data.Low.iloc[m] <= stop_loss:
                                loss_pct = ((stop_loss - entry_price) / entry_price) * 100
                                results.append({
                                    "Stock": stock,
                                    "BottomDate": bottom_date,
                                    "EntryDate": entry_date,
                                    "ExitDate": data.Date.iloc[m],
                                    "Result": "LOSS",
                                    "Return%": round(loss_pct,2)
                                })
                                break

                        break

    return pd.DataFrame(results)