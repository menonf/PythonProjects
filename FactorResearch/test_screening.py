"""Long/short stock screen construction, enrichment and comparison (pure pandas)."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from analytics.factors.transforms import combine
from analytics.portfolio import quantile_weights
from analytics.screening import (
    add_quality_flags,
    build_screen,
    enrich_screen,
    export_screen,
    factor_score_rows,
    screen_changes,
    sector_exposure,
    summarize_screen,
    watchlist,
)
from tests.unit_tests.synthetic import synthetic_fundamentals, synthetic_security_master, synthetic_shares_outstanding


def _components(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-06-02", periods=25)
    cols = list(range(1001, 1001 + n))
    mom = pd.DataFrame(rng.normal(0.1, 0.3, size=(25, n)), index=dates, columns=cols)
    mlv = pd.DataFrame(rng.normal(0.0, 0.05, size=(25, n)), index=dates, columns=cols)
    return {"momentum": mom, "ml_value": mlv}, dates


def test_build_screen_sides_weights_and_ranks() -> None:
    comps, dates = _components()
    as_of = dates[-1]
    screen = build_screen(comps, as_of, quantile=0.1)
    assert screen.index.name == "security_id"
    assert len(screen) == 60
    assert (screen["side"] == "LONG").sum() == 6 and (screen["side"] == "SHORT").sum() == 6
    assert np.isclose(screen.loc[screen["side"] == "LONG", "weight"].sum(), 1.0)
    assert np.isclose(screen.loc[screen["side"] == "SHORT", "weight"].sum(), -1.0)
    assert list(screen["rank"]) == list(range(1, 61))
    assert screen["composite"].is_monotonic_decreasing
    assert screen.iloc[0]["side"] == "LONG" and screen.iloc[-1]["side"] == "SHORT"
    assert np.isclose(screen["percentile"].iloc[0], 1.0)
    # component scores are cross-sectional z-scores and the composite is their weighted mean
    assert np.isclose(screen["momentum_score"].mean(), 0, atol=1e-9) and np.isclose(screen["momentum_score"].std(ddof=0), 1, atol=1e-6)
    np.testing.assert_allclose(screen["composite"], 0.5 * screen["momentum_score"] + 0.5 * screen["ml_value_score"], atol=1e-9)
    assert screen.attrs["as_of"] == as_of and screen.attrs["quantile"] == 0.1


def test_build_screen_matches_backtest_functions() -> None:
    """The screen on a date must equal the book the backtest machinery produces for that date."""
    comps, dates = _components()
    as_of = dates[10]
    weights = {"momentum": 0.7, "ml_value": 0.3}
    screen = build_screen(comps, as_of, quantile=0.2, weights=weights)
    composite = combine(comps, weights).loc[as_of]
    book = quantile_weights(combine(comps, weights), 0.2).loc[as_of]
    np.testing.assert_allclose(screen["composite"].reindex(composite.index), composite, atol=1e-12)
    np.testing.assert_allclose(screen["weight"].reindex(book.index), book, atol=1e-12)


def test_eligibility_and_missing_signals() -> None:
    comps, dates = _components(n=40)
    as_of = dates[-1]
    comps["ml_value"].loc[as_of, [1001, 1002]] = np.nan
    members = [c for c in comps["momentum"].columns if c not in (1003, 1004)] + [9999]  # 9999 has no data at all
    strict = build_screen(comps, as_of, quantile=0.1, eligible=members, require_all=True)
    assert set(strict.index) == set(members) - {1001, 1002, 1003, 1004, 9999}
    loose = build_screen(comps, as_of, quantile=0.1, eligible=members, require_all=False)
    assert {1001, 1002}.issubset(loose.index) and 9999 not in loose.index
    assert loose.loc[1001, "n_signals"] == 1 and loose.loc[1005, "n_signals"] == 2
    assert np.isnan(loose.loc[1001, "ml_value_score"]) and np.isclose(loose.loc[1001, "composite"], 0.5 * loose.loc[1001, "momentum_score"])


def test_build_screen_rank_method_and_min_names() -> None:
    comps, dates = _components(n=10)
    ranked = build_screen(comps, dates[0], quantile=0.2, method="rank", min_names=5)
    assert (ranked["side"] == "LONG").sum() == 2
    assert np.isclose(ranked["momentum_score"].mean(), 0, atol=1e-12)
    too_small = build_screen(comps, dates[0], quantile=0.2, min_names=50)
    assert (too_small["side"] == "").all() and (too_small["weight"] == 0).all()
    with pytest.raises(KeyError):
        build_screen(comps, "1999-01-01")
    with pytest.raises(ValueError):
        build_screen({}, dates[0])


def test_enrich_and_flags() -> None:
    comps, dates = _components(n=30)
    as_of = dates[-1]
    screen = build_screen(comps, as_of, quantile=0.1)
    ids = list(screen.index)
    master = synthetic_security_master(ids)
    tickers = master.set_index("security_id")["vendor_ticker"]
    prices = pd.Series(100.0, index=ids)
    shares = synthetic_shares_outstanding(ids)
    ratios = synthetic_fundamentals(ids)
    fdates = pd.Series(pd.Timestamp("2025-03-31"), index=ids)
    fdates.loc[ids[0]] = pd.NaT
    fdates.loc[ids[1]] = pd.Timestamp("2024-06-30")
    rich = enrich_screen(screen, as_of, security_master=master, tickers=tickers, prices=prices, ratios=ratios, shares_outstanding=shares, fundamentals_date=fdates)
    assert list(rich.columns[:4]) == ["ticker", "name", "sector", "industry"]
    assert rich.loc[ids[2], "ticker"] == f"SYN{ids[2]}.O"
    assert np.isclose(rich.loc[ids[2], "market_cap"], 100.0 * shares.loc[ids[2]])
    assert "P/E" in rich.columns and "EV/EBIT" in rich.columns
    assert rich.loc[ids[2], "fundamentals_age_days"] == (as_of - pd.Timestamp("2025-03-31")).days
    flagged = add_quality_flags(rich, stale_days=200)
    assert flagged.loc[ids[0], "flags"] == "no fundamentals"
    assert flagged.loc[ids[1], "flags"].startswith("stale fundamentals")
    assert flagged.loc[ids[2], "flags"] == ""
    summary = summarize_screen(flagged)
    assert summary["n_eligible"] == 30 and summary["n_long"] == 3 and summary["n_short"] == 3
    assert np.isclose(summary["gross_exposure"], 2.0) and np.isclose(summary["net_exposure"], 0.0)
    assert "n_selected_with_flags" in summary


def test_watchlist_and_sector_exposure() -> None:
    comps, dates = _components(n=40)
    screen = build_screen(comps, dates[-1], quantile=0.1)
    watch = watchlist(screen, band=0.05)
    assert len(watch) == 4 and set(watch["watch"]) == {"NEAR LONG", "NEAR SHORT"}
    assert (watch["side"] == "").all()
    best_unselected = screen[screen["side"] == ""]["composite"].max()
    assert best_unselected in set(watch.loc[watch["watch"] == "NEAR LONG", "composite"])
    master = synthetic_security_master(list(screen.index))
    rich = enrich_screen(screen, security_master=master)
    expo = sector_exposure(rich)
    assert np.isclose(expo["long_weight"].sum(), 1.0) and np.isclose(expo["short_weight"].sum(), -1.0)
    assert expo["n_eligible"].sum() == 40
    with pytest.raises(KeyError):
        sector_exposure(screen)


def test_screen_changes() -> None:
    comps, dates = _components(n=30)
    prev = build_screen(comps, dates[0], quantile=0.1)
    cur = build_screen(comps, dates[-1], quantile=0.1)
    changes = screen_changes(cur, prev)
    entered_long = {i for i in cur.index if cur.loc[i, "side"] == "LONG" and prev.loc[i, "side"] == ""}
    assert set(changes.index[changes["change"] == "ENTER LONG"]) == entered_long
    exits = {i for i in prev.index if prev.loc[i, "side"] != "" and cur.loc[i, "side"] == ""}
    assert set(changes.index[changes["change"].str.startswith("EXIT")]) == exits
    # a name that left the eligible universe is an exit
    dropped = cur.drop(index=prev.index[prev["side"] == "LONG"][:1])
    changes2 = screen_changes(dropped, prev)
    assert "EXIT LONG" in set(changes2["change"])
    # flips are labelled
    flipped = cur.copy()
    short_name = prev.index[prev["side"] == "SHORT"][0]
    flipped.loc[short_name, "side"] = "LONG"
    assert "FLIP SHORT->LONG" in set(screen_changes(flipped, prev)["change"])


def test_factor_score_rows_and_export() -> None:
    comps, dates = _components(n=12)
    screen = build_screen(comps, dates[-1], quantile=0.25)
    rows = factor_score_rows(screen, "SP500", "refinitiv")
    assert set(rows["factor_name"]) == {"momentum", "ml_value", "composite"}
    assert len(rows) == 36 and rows["as_of_date"].iloc[0] == dates[-1].strftime("%Y-%m-%d")
    assert rows["rank_pct"].between(0, 1).all()
    with tempfile.TemporaryDirectory() as tmp:
        written = export_screen(screen, tmp, dates[-1], label="unit", excel=False)
        assert written == [os.path.join(tmp, f"unit_{dates[-1]:%Y-%m-%d}.csv")]
        back = pd.read_csv(written[0], index_col=0)
        assert len(back) == 12 and "composite" in back.columns
