"""Turn factor signals into a dated long/short stock screen.

A *screen* is the strategy's book on one date: every eligible security with its
raw signals, their cross-sectional scores, the blended composite, rank, side
(``LONG`` / ``SHORT`` / blank) and target weight - enriched with identifiers,
prices, ratios and data-quality flags so it can be read and acted on.

    screen = build_screen({"momentum": mom, "ml_value": mlv}, as_of, quantile=0.1, eligible=members)
    screen = enrich_screen(screen, as_of, security_master=sm, tickers=ric, prices=px, ratios=panel)
    screen = add_quality_flags(screen, stale_days=200)
    longs, shorts = screen[screen["side"] == "LONG"], screen[screen["side"] == "SHORT"]
    changes = screen_changes(screen, previous_screen)

The composite and the weights are produced by the same functions the backtest
uses (:func:`analytics.factors.transforms.combine`,
:func:`analytics.portfolio.long_short.quantile_weights`), so the screen on a
rebalance date is exactly the book the backtest would have held.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from analytics.factors.transforms import combine, rank_pct, zscore
from analytics.portfolio.long_short import quantile_weights

LONG, SHORT, NONE = "LONG", "SHORT", ""

#: Ratios shown on the enriched screen by default.
DEFAULT_RATIO_COLUMNS: tuple[str, ...] = ("P/E", "P/B", "EV/EBIT", "RoE", "Debt/Equity")

#: Identity columns placed first when present.
_ID_COLUMNS = ("ticker", "name", "sector", "industry")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def build_screen(
    components: dict[str, DataFrame],
    as_of: Any,
    quantile: float = 0.1,
    weights: Optional[dict[str, float]] = None,
    method: str = "zscore",
    eligible: Optional[Iterable[Any]] = None,
    require_all: bool = True,
    min_names: int = 20,
    max_weight: Optional[float] = None,
) -> DataFrame:
    """Score every eligible security on ``as_of`` and pick the long and short legs.

    Args:
        components: ``name -> date x security_id`` raw signal frames (each must
            contain a row for ``as_of``).
        as_of: the screen date.
        quantile: fraction of eligible names in each leg (0.1 = top/bottom decile).
        weights: ``name -> weight`` for the composite (default equal; normalised).
        method: ``"zscore"`` or ``"rank"`` cross-sectional transform before blending.
        eligible: securities allowed on this date (e.g. point-in-time index
            members). ``None`` = every security present in the components.
        require_all: drop names missing any component (default) instead of
            scoring the missing component as neutral.
        min_names: below this many eligible names no legs are formed.
        max_weight: optional per-name cap (excess redistributed pro-rata).

    Returns:
        DataFrame indexed by ``security_id``, sorted best to worst, with columns
        ``rank, percentile, side, weight, composite, <name>_score..., <name>_raw..., n_signals``.
        ``percentile`` is the share of eligible names scoring at or below the row.
    """
    if not components:
        raise ValueError("build_screen needs at least one component signal")
    as_of = pd.Timestamp(as_of)
    names = list(components)

    raw: dict[str, Series] = {}
    for name, frame in components.items():
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if as_of not in idx:
            raise KeyError(f"component '{name}' has no row for {as_of.date()}")
        raw[name] = Series(frame.iloc[idx.get_loc(as_of)].to_numpy(dtype=float), index=frame.columns)
    raw_df = DataFrame(raw)
    raw_df.index.name = "security_id"
    if eligible is not None:
        raw_df = raw_df.reindex(pd.Index(sorted(set(eligible)), name="security_id"))

    present = raw_df.notna()
    keep = present.all(axis=1) if require_all else present.any(axis=1)
    raw_df = raw_df[keep]
    n_signals = present[keep].sum(axis=1)

    score_cols = [f"{n}_score" for n in names]
    raw_cols = [f"{n}_raw" for n in names]
    out = DataFrame(index=raw_df.index)
    if raw_df.empty:
        for col in ["rank", "percentile", "side", "weight", "composite", *score_cols, *raw_cols, "n_signals"]:
            out[col] = Series(dtype=object if col == "side" else float)
        out.attrs.update({"as_of": as_of, "quantile": quantile, "method": method, "components": names})
        return out

    for name in names:
        series = raw_df[name]
        if method == "zscore":
            out[f"{name}_score"] = zscore(series)
        elif method == "rank":
            ranks = rank_pct(series)
            out[f"{name}_score"] = ranks - ranks.mean()
        else:
            raise ValueError(f"unknown method '{method}'")

    one_row = {name: DataFrame([raw_df[name].to_numpy()], index=[as_of], columns=raw_df.index) for name in names}
    composite = combine(one_row, weights, method=method, missing=0.0).iloc[0]
    composite.name = as_of
    target = quantile_weights(composite.to_frame().T, quantile, min_names=min_names, max_weight=max_weight).iloc[0]

    out["composite"] = composite
    out["weight"] = target
    out["side"] = np.where(target > 0, LONG, np.where(target < 0, SHORT, NONE))
    out["rank"] = composite.rank(ascending=False, method="first").astype(int)
    out["percentile"] = rank_pct(composite)
    for name in names:
        out[f"{name}_raw"] = raw_df[name]
    out["n_signals"] = n_signals.astype(int)

    out = out[["rank", "percentile", "side", "weight", "composite", *score_cols, *raw_cols, "n_signals"]].sort_values("rank")
    out.attrs.update({"as_of": as_of, "quantile": quantile, "method": method, "components": names})
    return out


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_screen(
    screen: DataFrame,
    as_of: Optional[Any] = None,
    security_master: Optional[DataFrame] = None,
    tickers: Optional[Series] = None,
    prices: Optional[Series] = None,
    ratios: Optional[DataFrame] = None,
    ratio_columns: Sequence[str] = DEFAULT_RATIO_COLUMNS,
    shares_outstanding: Optional[Series] = None,
    fundamentals_date: Optional[Series] = None,
) -> DataFrame:
    """Attach identifiers, prices, market cap, ratios and fundamentals freshness.

    Args:
        screen: output of :func:`build_screen` (index = ``security_id``).
        as_of: screen date; enables ``fundamentals_age_days``.
        security_master: frame with ``security_id`` and any of ``name``, ``sector``, ``industry``.
        tickers: ``security_id -> ticker`` (e.g. Refinitiv RIC from the vendor xref).
        prices: ``security_id -> price`` on ``as_of``.
        ratios: point-in-time ratio panel (index = ``security_id``) - the columns in
            ``ratio_columns`` are copied across.
        shares_outstanding: ``security_id -> shares``; ``market_cap = price x shares``.
        fundamentals_date: ``security_id -> latest fiscal period end`` visible on ``as_of``.
    """
    out = screen.copy()
    added: list[str] = []
    if tickers is not None:
        out["ticker"] = Series(tickers).reindex(out.index)
    if security_master is not None:
        sm = security_master.drop_duplicates("security_id").set_index("security_id")
        for col in ("name", "sector", "industry"):
            if col in sm.columns:
                out[col] = sm[col].reindex(out.index)
    if prices is not None:
        out["price"] = Series(prices).reindex(out.index).astype(float)
        added.append("price")
    if shares_outstanding is not None and prices is not None:
        out["market_cap"] = out["price"] * Series(shares_outstanding).reindex(out.index).astype(float)
        added.append("market_cap")
    if ratios is not None:
        for col in ratio_columns:
            if col in ratios.columns:
                out[col] = ratios[col].reindex(out.index).astype(float)
                added.append(col)
    if fundamentals_date is not None:
        out["fundamentals_date"] = pd.to_datetime(Series(fundamentals_date).reindex(out.index))
        added.append("fundamentals_date")
        if as_of is not None:
            out["fundamentals_age_days"] = (pd.Timestamp(as_of) - out["fundamentals_date"]).dt.days
            added.append("fundamentals_age_days")

    id_cols = [c for c in _ID_COLUMNS if c in out.columns]
    core = [c for c in screen.columns if c in out.columns]
    ordered = id_cols + core + [c for c in added if c not in core]
    out = out[ordered + [c for c in out.columns if c not in ordered]]
    out.attrs.update(screen.attrs)
    return out


def add_quality_flags(screen: DataFrame, stale_days: int = 200) -> DataFrame:
    """Add a ``flags`` column describing data problems per name.

    Flags: ``no fundamentals`` (no visible fiscal period), ``stale fundamentals``
    (latest period end older than ``stale_days``), ``no <signal>`` for a missing
    raw component, ``no market cap``.
    """
    out = screen.copy()
    flags: dict[Any, list[str]] = {idx: [] for idx in out.index}
    if "fundamentals_date" in out.columns:
        for idx, val in out["fundamentals_date"].items():
            if pd.isna(val):
                flags[idx].append("no fundamentals")
        if "fundamentals_age_days" in out.columns:
            for idx, age in out["fundamentals_age_days"].items():
                if pd.notna(age) and age > stale_days:
                    flags[idx].append(f"stale fundamentals ({int(age)}d)")
    for col in [c for c in out.columns if c.endswith("_raw")]:
        for idx in out.index[out[col].isna()]:
            flags[idx].append(f"no {col[:-4]}")
    if "market_cap" in out.columns:
        for idx in out.index[out["market_cap"].isna()]:
            flags[idx].append("no market cap")
    out["flags"] = ["; ".join(flags[idx]) for idx in out.index]
    out.attrs.update(screen.attrs)
    return out


# ---------------------------------------------------------------------------
# Reading the screen
# ---------------------------------------------------------------------------


def summarize_screen(screen: DataFrame) -> dict[str, Any]:
    """Counts and exposures of the book described by ``screen``."""
    w = screen["weight"].astype(float) if "weight" in screen.columns else Series(dtype=float)
    summary: dict[str, Any] = {
        "as_of": screen.attrs.get("as_of"),
        "n_eligible": int(len(screen)),
        "n_long": int((screen["side"] == LONG).sum()) if "side" in screen.columns else 0,
        "n_short": int((screen["side"] == SHORT).sum()) if "side" in screen.columns else 0,
        "long_exposure": float(w[w > 0].sum()),
        "short_exposure": float(w[w < 0].sum()),
        "gross_exposure": float(w.abs().sum()),
        "net_exposure": float(w.sum()),
        "max_abs_weight": float(w.abs().max()) if len(w) else 0.0,
    }
    if "flags" in screen.columns:
        selected = screen[screen["side"] != NONE]
        summary["n_selected_with_flags"] = int((selected["flags"] != "").sum())
    return summary


def watchlist(screen: DataFrame, band: float = 0.05) -> DataFrame:
    """Unselected names closest to each leg's cut-off.

    ``band`` is a fraction of the eligible universe: with 500 names and
    ``band=0.05`` the 25 best unselected names are ``NEAR LONG`` and the 25
    worst are ``NEAR SHORT``.
    """
    if screen.empty:
        return screen.iloc[0:0].assign(watch=Series(dtype=object))
    k = max(1, math.ceil(band * len(screen)))
    unselected = screen[screen["side"] == NONE]
    near_long = unselected.nlargest(k, "composite").assign(watch="NEAR LONG")
    near_short = unselected.nsmallest(k, "composite").assign(watch="NEAR SHORT")
    return pd.concat([near_long, near_short])


def sector_exposure(screen: DataFrame, by: str = "sector") -> DataFrame:
    """Long, short and net weight plus name counts per ``by`` group (default sector)."""
    if by not in screen.columns:
        raise KeyError(f"screen has no '{by}' column - enrich it with a security master first")
    groups = screen[by].fillna("Unknown")
    g = screen.groupby(groups)
    out = DataFrame(
        {
            "n_eligible": g.size(),
            "n_long": g["side"].apply(lambda s: int((s == LONG).sum())),
            "n_short": g["side"].apply(lambda s: int((s == SHORT).sum())),
            "long_weight": g["weight"].apply(lambda w: float(w[w > 0].sum())),
            "short_weight": g["weight"].apply(lambda w: float(w[w < 0].sum())),
        }
    )
    out["net_weight"] = out["long_weight"] + out["short_weight"]
    out.index.name = by
    return out.sort_values("net_weight", ascending=False)


def screen_changes(current: DataFrame, previous: DataFrame, id_columns: Sequence[str] = ("ticker", "name")) -> DataFrame:
    """Names whose side changed between ``previous`` and ``current``.

    ``change`` is ``ENTER LONG`` / ``ENTER SHORT`` (new position), ``EXIT LONG`` /
    ``EXIT SHORT`` (closed) or ``FLIP LONG->SHORT`` / ``FLIP SHORT->LONG``.
    Names that dropped out of the eligible universe count as exits.
    """
    ids = current.index.union(previous.index)
    cur = current["side"].reindex(ids).fillna(NONE) if "side" in current.columns else Series(NONE, index=ids)
    prev = previous["side"].reindex(ids).fillna(NONE) if "side" in previous.columns else Series(NONE, index=ids)
    changed = cur != prev
    out = DataFrame({"previous_side": prev[changed], "current_side": cur[changed]})

    def _label(p: str, c: str) -> str:
        if p == NONE:
            return f"ENTER {c}"
        if c == NONE:
            return f"EXIT {p}"
        return f"FLIP {p}->{c}"

    out["change"] = [_label(p, c) for p, c in zip(out["previous_side"], out["current_side"])]
    for col in id_columns:
        if col in current.columns or col in previous.columns:
            cur_col = current[col].reindex(out.index) if col in current.columns else Series(index=out.index, dtype=object)
            prev_col = previous[col].reindex(out.index) if col in previous.columns else Series(index=out.index, dtype=object)
            out[col] = cur_col.where(cur_col.notna(), prev_col)
    if "composite" in current.columns:
        out["current_composite"] = current["composite"].reindex(out.index)
        out["current_weight"] = current["weight"].reindex(out.index)
    order = ["ENTER LONG", "ENTER SHORT", "FLIP SHORT->LONG", "FLIP LONG->SHORT", "EXIT LONG", "EXIT SHORT"]
    out["_order"] = out["change"].map({k: i for i, k in enumerate(order)}).fillna(len(order))
    out = out.sort_values(["_order", "current_composite"] if "current_composite" in out.columns else ["_order"], ascending=[True, False] if "current_composite" in out.columns else [True])
    return out.drop(columns="_order")


def factor_score_rows(screen: DataFrame, universe: str, source_vendor: str) -> DataFrame:
    """Reshape a screen into ``analytics.factor_scores`` rows (one per security per signal + composite)."""
    as_of = pd.Timestamp(screen.attrs.get("as_of")).strftime("%Y-%m-%d")
    rows = []
    signals = {c[:-4]: c for c in screen.columns if c.endswith("_raw")}
    signals["composite"] = "composite"
    for factor, col in signals.items():
        values = screen[col].astype(float)
        ranks = rank_pct(values)
        for sec, val in values.items():
            if pd.isna(val):
                continue
            rows.append({"as_of_date": as_of, "security_id": int(sec), "factor_name": factor, "factor_value": float(val), "rank_pct": float(ranks.loc[sec]), "universe": universe, "source_vendor": source_vendor})
    return DataFrame(rows)


def export_screen(screen: DataFrame, directory: str, as_of: Any, label: str = "screen", excel: bool = True) -> list[str]:
    """Write the screen to ``<directory>/<label>_<as_of>.csv`` (and ``.xlsx`` when an Excel engine is installed)."""
    os.makedirs(directory, exist_ok=True)
    stem = os.path.join(directory, f"{label}_{pd.Timestamp(as_of):%Y-%m-%d}")
    written = [stem + ".csv"]
    screen.to_csv(written[0])
    if excel:
        try:
            screen.to_excel(stem + ".xlsx")
            written.append(stem + ".xlsx")
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
    return written
