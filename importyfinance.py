import yfinance as yf

df = yf.download("RELIANCE.NS", period="2y", auto_adjust=False, threads=False)

print(df.head())
print("\nVolume sample:")
print(df["Volume"].head(30))
print("\nAny non-zero volume?", (df["Volume"] > 0).any())