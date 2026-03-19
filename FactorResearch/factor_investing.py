import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def get_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia."""
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = table[0]
    tickers = df["Symbol"].tolist()
    # Fix tickers with dots (e.g., BRK.B -> BRK-B for yfinance)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def fetch_fundamental_data(tickers, batch_size=50):
    """Fetch fundamental data for all tickers."""
    data = []
    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Fetching fundamentals: batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                record = {
                    "ticker": ticker,
                    # Value factors
                    "pe_ratio": info.get("trailingPE", np.nan),
                    "pb_ratio": info.get("priceToBook", np.nan),
                    "dividend_yield": info.get("dividendYield", np.nan),
                    "ev_to_ebitda": info.get("enterpriseToEbitda", np.nan),
                    # Growth factors
                    "earnings_growth": info.get("earningsGrowth", np.nan),
                    "revenue_growth": info.get("revenueGrowth", np.nan),
                    "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth", np.nan),
                    "revenue_per_share": info.get("revenuePerShare", np.nan),
                    # Other useful info
                    "market_cap": info.get("marketCap", np.nan),
                    "sector": info.get("sector", "Unknown"),
                    "name": info.get("shortName", ticker),
                }
                data.append(record)
            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")
                continue
    return pd.DataFrame(data)


def compute_factor_scores(df):
    """
    Compute composite Value and Growth factor scores.
    
    Value Score: Low P/E, Low P/B, High Dividend Yield, Low EV/EBITDA
    Growth Score: High Earnings Growth, High Revenue Growth, High Quarterly Earnings Growth
    """
    scored = df.copy()

    # --- Value Factor ---
    # For value, lower P/E and P/B are better -> rank ascending (rank 1 = cheapest)
    # Higher dividend yield is better -> rank descending
    # Lower EV/EBITDA is better -> rank ascending

    scored["pe_rank"] = scored["pe_ratio"].rank(ascending=True, pct=True)
    scored["pb_rank"] = scored["pb_ratio"].rank(ascending=True, pct=True)
    scored["dy_rank"] = scored["dividend_yield"].rank(ascending=False, pct=True)
    scored["ev_ebitda_rank"] = scored["ev_to_ebitda"].rank(ascending=True, pct=True)

    # Composite value score (average of available ranks; 0 = most value, 1 = least value)
    value_cols = ["pe_rank", "pb_rank", "dy_rank", "ev_ebitda_rank"]
    scored["value_score"] = scored[value_cols].mean(axis=1, skipna=True)
    # Invert so higher = more value
    scored["value_score"] = 1 - scored["value_score"]

    # --- Growth Factor ---
    # Higher growth metrics are better -> rank descending
    scored["eg_rank"] = scored["earnings_growth"].rank(ascending=False, pct=True)
    scored["rg_rank"] = scored["revenue_growth"].rank(ascending=False, pct=True)
    scored["eqg_rank"] = scored["earnings_quarterly_growth"].rank(ascending=False, pct=True)

    growth_cols = ["eg_rank", "rg_rank", "eqg_rank"]
    scored["growth_score"] = scored[growth_cols].mean(axis=1, skipna=True)
    # Invert so higher = more growth
    scored["growth_score"] = 1 - scored["growth_score"]

    # --- Combined Factor Score ---
    # Equal weight between value and growth
    scored["combined_score"] = (
        0.5 * scored["value_score"].fillna(0) + 0.5 * scored["growth_score"].fillna(0)
    )

    return scored


def generate_signals(scored_df, top_n=30, bottom_n=30):
    """
    Generate long/short signals based on combined factor score.
    
    LONG the top_n stocks (high combined score).
    SHORT the bottom_n stocks (low combined score).
    """
    # Drop stocks with no combined score
    valid = scored_df.dropna(subset=["combined_score"]).copy()
    valid = valid.sort_values("combined_score", ascending=False).reset_index(drop=True)

    valid["signal"] = "NEUTRAL"
    valid.loc[valid.index[:top_n], "signal"] = "LONG"
    valid.loc[valid.index[-bottom_n:], "signal"] = "SHORT"

    return valid


def fetch_historical_prices(tickers, period="1y"):
    """Fetch historical adjusted close prices."""
    print(f"Fetching historical prices for {len(tickers)} tickers...")
    prices = yf.download(tickers, period=period, auto_adjust=True, progress=True)["Close"]
    return prices


def simulate_trading(signals_df, lookback_period="1y", rebalance_freq="M", initial_capital=1_000_000):
    """
    Simulate a long/short factor strategy.
    
    - Go LONG top-ranked stocks (equal weight)
    - Go SHORT bottom-ranked stocks (equal weight)
    - Rebalance monthly
    """
    long_tickers = signals_df[signals_df["signal"] == "LONG"]["ticker"].tolist()
    short_tickers = signals_df[signals_df["signal"] == "SHORT"]["ticker"].tolist()
    all_tickers = long_tickers + short_tickers

    if not all_tickers:
        print("No tickers to trade!")
        return None

    # Fetch prices
    prices = fetch_historical_prices(all_tickers, period=lookback_period)
    prices = prices.dropna(axis=1, how="all")

    # Filter to only tickers we actually got data for
    available_long = [t for t in long_tickers if t in prices.columns]
    available_short = [t for t in short_tickers if t in prices.columns]

    print(f"\nTrading {len(available_long)} LONG and {len(available_short)} SHORT positions")

    # Calculate daily returns
    returns = prices.pct_change().dropna()

    # Portfolio returns: equal-weight long/short
    if available_long:
        long_returns = returns[available_long].mean(axis=1)
    else:
        long_returns = pd.Series(0, index=returns.index)

    if available_short:
        short_returns = -returns[available_short].mean(axis=1)  # negative because we're short
    else:
        short_returns = pd.Series(0, index=returns.index)

    # Combined portfolio: 50% capital long, 50% capital short
    portfolio_returns = 0.5 * long_returns + 0.5 * short_returns

    # Benchmark: equal-weight S&P 500 proxy (use SPY)
    spy = yf.download("SPY", period=lookback_period, auto_adjust=True, progress=False)["Close"]
    spy_returns = spy.pct_change().dropna()

    # Align dates
    common_dates = portfolio_returns.index.intersection(spy_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    spy_returns = spy_returns.loc[common_dates]

    # Build equity curves
    portfolio_equity = (1 + portfolio_returns).cumprod() * initial_capital
    spy_equity = (1 + spy_returns).cumprod() * initial_capital
    long_equity = (1 + long_returns.loc[common_dates]).cumprod() * initial_capital
    short_equity = (1 + short_returns.loc[common_dates]).cumprod() * initial_capital

    results = {
        "portfolio_returns": portfolio_returns,
        "spy_returns": spy_returns,
        "portfolio_equity": portfolio_equity,
        "spy_equity": spy_equity,
        "long_equity": long_equity,
        "short_equity": short_equity,
    }

    return results


def compute_performance_metrics(results):
    """Compute key performance metrics."""
    port_ret = results["portfolio_returns"]
    spy_ret = results["spy_returns"]

    trading_days = 252

    def calc_metrics(returns, name):
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (trading_days / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(trading_days)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min()
        win_rate = (returns > 0).mean()

        return {
            "Strategy": name,
            "Total Return": f"{total_return:.2%}",
            "Annualized Return": f"{ann_return:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Win Rate (daily)": f"{win_rate:.2%}",
            "Trading Days": len(returns),
        }

    port_metrics = calc_metrics(port_ret, "Factor L/S Portfolio")
    spy_metrics = calc_metrics(spy_ret, "SPY Benchmark")

    metrics_df = pd.DataFrame([port_metrics, spy_metrics])
    return metrics_df


def plot_results(results):
    """Plot equity curves and drawdowns."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Equity Curves
        axes[0].plot(results["portfolio_equity"], label="Factor L/S Portfolio", linewidth=2)
        axes[0].plot(results["spy_equity"], label="SPY Benchmark", linewidth=2, alpha=0.7)
        axes[0].plot(results["long_equity"], label="Long Leg Only", linewidth=1, linestyle="--", alpha=0.6)
        axes[0].plot(results["short_equity"], label="Short Leg Only", linewidth=1, linestyle="--", alpha=0.6)
        axes[0].set_title("Factor Investing Strategy: Growth + Value", fontsize=14)
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Cumulative Returns
        port_cum = (1 + results["portfolio_returns"]).cumprod() - 1
        spy_cum = (1 + results["spy_returns"]).cumprod() - 1
        axes[1].plot(port_cum, label="Factor L/S Portfolio", linewidth=2)
        axes[1].plot(spy_cum, label="SPY Benchmark", linewidth=2, alpha=0.7)
        axes[1].set_ylabel("Cumulative Return")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Drawdown
        port_equity = (1 + results["portfolio_returns"]).cumprod()
        port_peak = port_equity.cummax()
        port_dd = (port_equity - port_peak) / port_peak

        spy_equity = (1 + results["spy_returns"]).cumprod()
        spy_peak = spy_equity.cummax()
        spy_dd = (spy_equity - spy_peak) / spy_peak

        axes[2].fill_between(port_dd.index, port_dd, 0, alpha=0.4, label="Factor L/S Portfolio")
        axes[2].fill_between(spy_dd.index, spy_dd, 0, alpha=0.4, label="SPY Benchmark")
        axes[2].set_ylabel("Drawdown")
        axes[2].set_xlabel("Date")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("factor_investing_results.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("\nChart saved to 'factor_investing_results.png'")

    except ImportError:
        print("matplotlib not installed. Skipping plots.")


def main():
    print("=" * 70)
    print("FACTOR INVESTING STRATEGY: GROWTH + VALUE")
    print("Universe: S&P 500 | Factors: Value & Growth | Long/Short")
    print("=" * 70)

    # Step 1: Get S&P 500 universe
    print("\n[1/6] Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"  Found {len(tickers)} tickers")

    # Step 2: Fetch fundamental data
    print("\n[2/6] Fetching fundamental data (this may take several minutes)...")
    fundamentals = fetch_fundamental_data(tickers)
    print(f"  Retrieved data for {len(fundamentals)} stocks")

    # Step 3: Compute factor scores
    print("\n[3/6] Computing factor scores...")
    scored = compute_factor_scores(fundamentals)

    # Display factor score distribution
    print("\n  Factor Score Summary:")
    print(scored[["value_score", "growth_score", "combined_score"]].describe().round(3))

    # Step 4: Generate signals
    print("\n[4/6] Generating trading signals...")
    signals = generate_signals(scored, top_n=30, bottom_n=30)

    long_stocks = signals[signals["signal"] == "LONG"][
        ["ticker", "name", "sector", "value_score", "growth_score", "combined_score"]
    ].head(15)
    short_stocks = signals[signals["signal"] == "SHORT"][
        ["ticker", "name", "sector", "value_score", "growth_score", "combined_score"]
    ].tail(15)

    print("\n  TOP 15 LONG Positions (Highest Combined Score):")
    print(long_stocks.to_string(index=False))

    print("\n  TOP 15 SHORT Positions (Lowest Combined Score):")
    print(short_stocks.to_string(index=False))

    # Signal summary
    signal_counts = signals["signal"].value_counts()
    print(f"\n  Signal Summary: {dict(signal_counts)}")

    # Sector breakdown
    print("\n  LONG positions by sector:")
    long_sectors = signals[signals["signal"] == "LONG"]["sector"].value_counts()
    for sector, count in long_sectors.items():
        print(f"    {sector}: {count}")

    print("\n  SHORT positions by sector:")
    short_sectors = signals[signals["signal"] == "SHORT"]["sector"].value_counts()
    for sector, count in short_sectors.items():
        print(f"    {sector}: {count}")

    # Step 5: Simulate trading
    print("\n[5/6] Simulating trading strategy (1-year backtest)...")
    results = simulate_trading(signals, lookback_period="1y", initial_capital=1_000_000)

    if results is None:
        print("Simulation failed. Exiting.")
        return

    # Step 6: Performance analysis
    print("\n[6/6] Computing performance metrics...")
    metrics = compute_performance_metrics(results)

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(metrics.to_string(index=False))

    # Monthly returns breakdown
    monthly_returns = results["portfolio_returns"].resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )
    print("\n  Monthly Returns:")
    for date, ret in monthly_returns.items():
        print(f"    {date.strftime('%Y-%m')}: {ret:+.2%}")

    # Plot results
    print("\n  Generating plots...")
    plot_results(results)

    # Save signals to CSV
    signals.to_csv("factor_signals.csv", index=False)
    print("\n  Signals saved to 'factor_signals.csv'")

    print("\n" + "=" * 70)
    print("STRATEGY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
