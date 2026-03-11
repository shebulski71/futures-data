from query_futures_data import load_bars

df = load_bars(
    roots="ES",
    freq="5m",
    start="2026-01-01",
    end="2026-02-01",
    dataset="resampled",
)

print(df)