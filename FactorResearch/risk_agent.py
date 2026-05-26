"""Risk commentary report generator.

Reads portfolio data from Aladdin Explore Excel exports, aggregates metrics
by dimension, sends context to a local LLM for interpretation, and produces
PDF reports with tables and AI-generated commentary.
"""

import argparse
import json
import logging
import platform
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
import pandas as pd
import yaml
from fpdf import FPDF

# ═══════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# Paths
DATA_DIR = Path("explore_output/two_week_changes")
OUTPUT_DIR = Path("exports")
PORTFOLIOS = ["BCGHYBU", "BCEHYBF", "BCUSHYB"]

# LLM
INTERPRETER_MODEL = "deepseek-r1:8b"
INTERPRETER_OPTIONS = {"temperature": 0}
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = 2.0  # seconds, doubles each retry
MAX_LLM_WORKERS = 3  # concurrent LLM calls per dimension

# Dimension cardinality (high → top/bottom 5 groups only; low → all groups)
HIGH_CARDINALITY_DIMS = ["Ticker", "Industry", "Country Of Risk"]
LOW_CARDINALITY_DIMS = ["Sleeve", "Rating_l1"]
ALL_DIMS = LOW_CARDINALITY_DIMS + HIGH_CARDINALITY_DIMS

# Multi-dimension: top N combos to keep (limits combinatorial explosion)
MULTI_DIM_TOP_N = 15

# Ranking metrics (used to select top/bottom groups and drill-down positions)
RANKING_METRICS: list[str] = [
    "Active Duration Times Spread Contribution Change in 2 weeks",
]

# PDF styling constants
PDF_HEADER_BG = (41, 65, 122)
PDF_HEADER_FG = (255, 255, 255)
PDF_ALT_ROW_BG = (235, 238, 245)
PDF_SEPARATOR_COLOR = (180, 180, 180)

# ═══════════════════════════════════════════════════════════════════════
# Column Definitions
# ═══════════════════════════════════════════════════════════════════════

CHANGE_SUFFIX = "Change in 2 weeks"

IDENTIFIERS = [
    "Security", "Ticker", "Sleeve", "Rating_l1", "Rating_l2",
    "Industry", "Country Of Risk", "Isin",
]

SCOPES = ["Portfolio", "Benchmark", "Active"]

METRICS = [
    "Market Value Percent",
    "Duration", "Duration Contribution",
    "Spread", "Spread Contribution",
    "Spread Duration", "Spread Duration Contribution",
    "Duration Times Spread", "Duration Times Spread Contribution",
    "Risk Contribution",
]

METRIC_GROUPS = [
    ["Market Value Percent"],
    ["Duration", "Duration Contribution"],
    ["Spread", "Spread Contribution"],
    ["Spread Duration", "Spread Duration Contribution"],
    ["Duration Times Spread", "Duration Times Spread Contribution"],
    ["Risk Contribution"],
]

SECURITY_METRICS = ["Duration", "Spread", "Spread Duration", "Duration Times Spread"]

SECURITY_CONTRIBUTION_MAP = {
    "Duration": "Portfolio Duration Contribution",
    "Spread": "Portfolio Spread Contribution",
    "Spread Duration": "Portfolio Spread Duration Contribution",
    "Duration Times Spread": "Portfolio Duration Times Spread Contribution",
}

# ── Derived column lists (computed once at import time) ──────────────


def _expand_metric_across_scopes(metric: str) -> list[str]:
    """Return [Scope Metric, Scope Metric Change, …] for every scope."""
    return [
        f"{scope} {metric}{suffix}"
        for scope in SCOPES
        for suffix in ("", f" {CHANGE_SUFFIX}")
    ]


SELECTED_COLUMNS = IDENTIFIERS + [
    col for metric in METRICS for col in _expand_metric_across_scopes(metric)
]

MV_PERCENT_COLS = [
    f"{scope} Market Value Percent{suffix}"
    for scope in SCOPES
    for suffix in ("", f" {CHANGE_SUFFIX}")
]


def _build_display_order() -> list[str]:
    """Build column ordering that groups related metrics together."""
    cols = list(IDENTIFIERS)
    for group in METRIC_GROUPS:
        for scope in SCOPES + ["Security"]:
            for metric in group:
                cols.append(f"{scope} {metric}")
                cols.append(f"{scope} {metric} {CHANGE_SUFFIX}")
    return cols


DISPLAY_ORDER = _build_display_order()

PORTFOLIO_SUMMARY_COLUMNS = [
    "Security",
    "Portfolio Market Value Percent",
    "Benchmark Market Value Percent",
    *[
        col
        for metric in ["Duration", "Spread", "Spread Duration",
                        "Duration Times Spread", "Risk Contribution"]
        for scope in SCOPES
        for col in (f"{scope} {metric}", f"{scope} {metric} {CHANGE_SUFFIX}")
    ],
]

# ── PDF table column specs: (full_name, short_header, format) ────────

PDF_TABLE_COLS = [
    ("Portfolio Market Value Percent", "Port\nMV%", ".2f"),
    ("Portfolio Market Value Percent Change in 2 weeks", "Port MV%\nChg 2w", ".2f"),
    ("Benchmark Market Value Percent", "BM\nMV%", ".2f"),
    ("Benchmark Market Value Percent Change in 2 weeks", "BM MV%\nChg 2w", ".2f"),
    ("Active Market Value Percent", "Active\nMV%", ".2f"),
    ("Active Market Value Percent Change in 2 weeks", "Active MV%\nChg 2w", ".2f"),
    ("Portfolio Duration Times Spread Contribution", "Port DTS\nContrib", ".4f"),
    ("Portfolio Duration Times Spread Contribution Change in 2 weeks", "Port DTS\nChg 2w", ".4f"),
    ("Benchmark Duration Times Spread Contribution", "BM DTS\nContrib", ".4f"),
    ("Benchmark Duration Times Spread Contribution Change in 2 weeks", "BM DTS\nChg 2w", ".4f"),
    ("Active Duration Times Spread Contribution", "Active DTS\nContrib", ".4f"),
    ("Active Duration Times Spread Contribution Change in 2 weeks", "Active DTS\nChg 2w", ".4f"),
]

PDF_PORTFOLIO_COLS = [
    ("Portfolio Duration", "Port\nDur", ".2f"),
    ("Portfolio Duration Change in 2 weeks", "Port Dur\nChg 2w", ".4f"),
    ("Active Duration", "Active\nDur", ".2f"),
    ("Active Duration Change in 2 weeks", "Active Dur\nChg 2w", ".4f"),
    ("Portfolio Spread", "Port\nSprd", ".1f"),
    ("Portfolio Spread Change in 2 weeks", "Port Sprd\nChg 2w", ".1f"),
    ("Active Spread", "Active\nSprd", ".1f"),
    ("Active Spread Change in 2 weeks", "Active Sprd\nChg 2w", ".1f"),
    ("Portfolio Spread Duration", "Port\nSprd Dur", ".2f"),
    ("Portfolio Spread Duration Change in 2 weeks", "Port SpDur\nChg 2w", ".4f"),
    ("Active Spread Duration", "Active\nSprd Dur", ".2f"),
    ("Active Spread Duration Change in 2 weeks", "Active SpDur\nChg 2w", ".4f"),
    ("Portfolio Duration Times Spread", "Port\nDTS", ".1f"),
    ("Portfolio Duration Times Spread Change in 2 weeks", "Port DTS\nChg 2w", ".2f"),
    ("Active Duration Times Spread", "Active\nDTS", ".1f"),
    ("Active Duration Times Spread Change in 2 weeks", "Active DTS\nChg 2w", ".2f"),
]

# ═══════════════════════════════════════════════════════════════════════
# File I/O Helpers
# ═══════════════════════════════════════════════════════════════════════


def _data_path(portfolio: str) -> Path:
    return DATA_DIR / f"All_Data(Ex-Cash & Derivatives)_{portfolio}.xlsx"


def _output_path(portfolio: str) -> Path:
    return OUTPUT_DIR / f"analysis_results_{portfolio}.txt"


# ═══════════════════════════════════════════════════════════════════════
# DataFrame Utilities
# ═══════════════════════════════════════════════════════════════════════


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to group related metrics together."""
    ordered = [c for c in DISPLAY_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in set(ordered)]
    return df[ordered + remaining]


def _scale_mv_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Market Value Percent columns from fraction to percentage."""
    for col in MV_PERCENT_COLS:
        if col in df.columns:
            df[col] = df[col] * 100
    return df


def _derive_security_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create Security-level columns by falling back from Portfolio to Benchmark."""
    for metric in SECURITY_METRICS:
        for suffix in ("", f" {CHANGE_SUFFIX}"):
            port_col = f"Portfolio {metric}{suffix}"
            bench_col = f"Benchmark {metric}{suffix}"
            sec_col = f"Security {metric}{suffix}"
            if port_col in df.columns and bench_col in df.columns:
                df[sec_col] = df[port_col].fillna(df[bench_col])
    return df


def _to_yaml(df: pd.DataFrame) -> str:
    """Convert DataFrame to YAML (via JSON) for LLM-readable context."""
    return yaml.dump(
        json.loads(df.to_json(orient="records")),
        sort_keys=False,
        allow_unicode=True,
    )


def _top_bottom(
    df: pd.DataFrame, cols: str | list[str], n: int, direction: str,
) -> pd.DataFrame:
    """Return top or bottom *n* rows ranked by *cols*.

    Supports a single column name or a list for multi-metric ranking.
    """
    if isinstance(cols, str):
        cols = [cols]
    usable = [c for c in cols if c in df.columns]
    if not usable:
        return df.head(n)
    ascending = direction == "bottom"
    return df.sort_values(usable, ascending=ascending).head(n)


# ═══════════════════════════════════════════════════════════════════════
# Data Loading & Preparation
# ═══════════════════════════════════════════════════════════════════════


def _read_metadata(path: Path) -> pd.DataFrame:
    """Read portfolio metadata (cells C3:G4) from the Excel file."""
    return pd.read_excel(path, header=None, skiprows=2, nrows=2, usecols="C:G")


def _read_raw_data(path: Path) -> pd.DataFrame:
    """Read the Excel export and assign canonical column names."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_excel(path, skiprows=7, skipfooter=3)
    df.columns = SELECTED_COLUMNS

    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def _prepare_portfolio_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the first row (portfolio totals) with selected columns."""
    totals = df.iloc[:1][PORTFOLIO_SUMMARY_COLUMNS].copy()
    totals = totals.where(pd.notna(totals), None)
    return _scale_mv_percent(totals)


def _prepare_positional_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract security-level rows with derived metrics and percentage scaling."""
    positions = df.iloc[1:].copy()
    _derive_security_metrics(positions)
    _scale_mv_percent(positions)
    positions = positions.where(pd.notna(positions), None)
    positions = _reorder_columns(positions)

    sort_col = "Portfolio Market Value Percent"
    if sort_col in positions.columns:
        positions = positions.sort_values(sort_col, ascending=False).reset_index(drop=True)

    return positions


# ═══════════════════════════════════════════════════════════════════════
# Grouped Aggregation
# ═══════════════════════════════════════════════════════════════════════


def _weighted_avg(
    df: pd.DataFrame, value_cols: list[str], weight_col: str, group_cols: list[str],
) -> pd.DataFrame | None:
    """Compute grouped weighted averages for *value_cols* using *weight_col*."""
    if not value_cols or weight_col not in df.columns:
        return None

    work = df[group_cols + [weight_col] + value_cols].copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")

    weighted_col_names = []
    for col in value_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        wv_name = f"_wv_{col}"
        work[wv_name] = work[col] * work[weight_col]
        weighted_col_names.append(wv_name)

    agg = {weight_col: "sum", **{wv: "sum" for wv in weighted_col_names}}
    result = work.groupby(group_cols, as_index=False).agg(agg)

    for col, wv_name in zip(value_cols, weighted_col_names):
        result[col] = result[wv_name] / result[weight_col].replace(0, pd.NA)

    return result[group_cols + value_cols]


def _weighted_avg_with_change(
    df: pd.DataFrame,
    level_cols: list[str],
    weight_col: str,
    weight_change_col: str,
    group_cols: list[str],
) -> pd.DataFrame | None:
    """Compute current weighted average and derive 2-week changes.

    Change = WA_current - WA_prior, where prior values are reconstructed
    by subtracting the 2-week deltas from current values and weights.
    """
    if not level_cols or weight_col not in df.columns:
        return None

    current = _weighted_avg(df, level_cols, weight_col, group_cols)
    if current is None:
        return None
    if weight_change_col not in df.columns:
        return current

    # Map each level column to its change column (if available)
    change_map: dict[str, str] = {}
    for level_col in level_cols:
        change_col = f"{level_col} {CHANGE_SUFFIX}"
        if change_col in df.columns:
            change_map[level_col] = change_col

    # Reconstruct prior-period values: prior = current - change
    needed = [weight_col, weight_change_col] + level_cols + list(change_map.values())
    work = df[group_cols + needed].copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work[weight_change_col] = pd.to_numeric(work[weight_change_col], errors="coerce").fillna(0)
    work["_prior_weight"] = work[weight_col] - work[weight_change_col]

    prior_col_names = []
    for level_col in level_cols:
        work[level_col] = pd.to_numeric(work[level_col], errors="coerce")
        prior_name = f"_prior_{level_col}"
        if level_col in change_map:
            work[change_map[level_col]] = (
                pd.to_numeric(work[change_map[level_col]], errors="coerce").fillna(0)
            )
            work[prior_name] = work[level_col] - work[change_map[level_col]]
        else:
            work[prior_name] = work[level_col]
        prior_col_names.append(prior_name)

    # Weighted average of prior values using prior weights
    for prior_name in prior_col_names:
        work[f"_wv_{prior_name}"] = work[prior_name] * work["_prior_weight"]

    agg = {"_prior_weight": "sum", **{f"_wv_{p}": "sum" for p in prior_col_names}}
    prior_grouped = work.groupby(group_cols, as_index=False).agg(agg)
    for prior_name in prior_col_names:
        prior_grouped[prior_name] = (
            prior_grouped[f"_wv_{prior_name}"]
            / prior_grouped["_prior_weight"].replace(0, pd.NA)
        )

    # Merge and compute: change = current - prior
    result = current.merge(
        prior_grouped[group_cols + prior_col_names], on=group_cols, how="left",
    )
    change_result_cols = []
    for level_col, prior_name in zip(level_cols, prior_col_names):
        change_col = f"{level_col} {CHANGE_SUFFIX}"
        result[change_col] = result[level_col] - result[prior_name]
        change_result_cols.append(change_col)

    return result[group_cols + level_cols + change_result_cols]


def _prepare_grouped_data(positions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate positional data by *group_cols*.

    Steps:
      1. Sum additive columns (contributions, MV%)
      2. Weighted-average standalone metrics (Duration, Spread, etc.)
      3. Derive Active = Portfolio - Benchmark
      4. Compute Security-level metrics from contribution/weight ratios
      5. Compute Security-level 2-week changes
    """
    port_wt = "Portfolio Market Value Percent"
    port_wt_chg = f"{port_wt} {CHANGE_SUFFIX}"
    bench_wt = "Benchmark Market Value Percent"
    bench_wt_chg = f"{bench_wt} {CHANGE_SUFFIX}"

    # Classify columns into additive (sum) vs standalone (weighted average)
    additive_cols = [
        c for c in positions.columns
        if ("Contribution" in c or "Market Value Percent" in c)
        and c not in group_cols
    ]
    standalone_cols = [
        c for c in positions.columns
        if c.startswith(("Portfolio", "Benchmark", "Active"))
        and c not in additive_cols
        and c not in group_cols
    ]

    # Step 1: Sum additive columns
    agg_dict = {c: "sum" for c in additive_cols if c in positions.columns}
    grouped = positions.groupby(group_cols, as_index=False).agg(agg_dict)

    # Step 2: Weighted-average standalone metrics by scope
    port_levels = [c for c in standalone_cols if c.startswith("Portfolio") and CHANGE_SUFFIX not in c]
    bench_levels = [c for c in standalone_cols if c.startswith("Benchmark") and CHANGE_SUFFIX not in c]
    active_levels = [c for c in standalone_cols if c.startswith("Active") and CHANGE_SUFFIX not in c]

    for scope_levels, wt, wt_chg in [
        (port_levels, port_wt, port_wt_chg),
        (bench_levels, bench_wt, bench_wt_chg),
    ]:
        wa = _weighted_avg_with_change(positions, scope_levels, wt, wt_chg, group_cols)
        if wa is not None:
            grouped = grouped.merge(wa, on=group_cols, how="left")

    # Step 3: Active = Portfolio - Benchmark
    for active_col in active_levels:
        metric_suffix = active_col[len("Active "):]
        port_col = f"Portfolio {metric_suffix}"
        bench_col = f"Benchmark {metric_suffix}"
        if port_col in grouped.columns and bench_col in grouped.columns:
            grouped[active_col] = grouped[port_col] - grouped[bench_col]
        port_chg = f"{port_col} {CHANGE_SUFFIX}"
        bench_chg = f"{bench_col} {CHANGE_SUFFIX}"
        active_chg = f"{active_col} {CHANGE_SUFFIX}"
        if port_chg in grouped.columns and bench_chg in grouped.columns:
            grouped[active_chg] = grouped[port_chg] - grouped[bench_chg]

    # Step 4: Security-level metrics = Contribution / Weight
    for metric, contrib_col in SECURITY_CONTRIBUTION_MAP.items():
        if contrib_col in grouped.columns and port_wt in grouped.columns:
            grouped[f"Security {metric}"] = (
                grouped[contrib_col] / grouped[port_wt].replace(0, pd.NA)
            )

    # Step 5: Security-level changes = current_ratio - prior_ratio
    for metric, contrib_col in SECURITY_CONTRIBUTION_MAP.items():
        contrib_chg = f"{contrib_col} {CHANGE_SUFFIX}"
        required = [contrib_col, contrib_chg, port_wt, port_wt_chg]
        if all(c in grouped.columns for c in required):
            prior_contrib = grouped[contrib_col] - grouped[contrib_chg]
            prior_weight = grouped[port_wt] - grouped[port_wt_chg]
            grouped[f"Security {metric} {CHANGE_SUFFIX}"] = (
                grouped[contrib_col] / grouped[port_wt].replace(0, pd.NA)
                - prior_contrib / prior_weight.replace(0, pd.NA)
            )

    grouped = _reorder_columns(grouped)

    # Prefix weighted-average columns to distinguish from summed ones
    rename_map = {}
    for col_list in (port_levels, bench_levels, active_levels):
        for col in col_list:
            for variant in (col, f"{col} {CHANGE_SUFFIX}"):
                if variant in grouped.columns:
                    rename_map[variant] = f"Weighted Average {variant}"
    grouped = grouped.rename(columns=rename_map)

    if port_wt in grouped.columns:
        grouped = grouped.sort_values(port_wt, ascending=False).reset_index(drop=True)

    return grouped


# ═══════════════════════════════════════════════════════════════════════
# Drill-Down
# ═══════════════════════════════════════════════════════════════════════


def _drill_down(
    positional_data: pd.DataFrame, group_col: str, group_value: str,
) -> pd.DataFrame:
    """Return security-level rows for a specific group value.

    Keeps Security, Ticker, Isin + the group column as identifiers,
    plus all contribution and Market Value Percent columns.
    """
    filtered = positional_data[positional_data[group_col] == group_value].copy()
    keep_ids = {"Security", "Ticker", "Isin", group_col}

    drop_cols = [
        c for c in filtered.columns
        if c.startswith(("Portfolio ", "Benchmark ", "Active "))
        and "Contribution" not in c
        and "Market Value Percent" not in c
    ]
    drop_cols += [c for c in IDENTIFIERS if c not in keep_ids and c in filtered.columns]
    return filtered.drop(columns=drop_cols)


def _drill_down_multi(
    positional_data: pd.DataFrame,
    group_cols: list[str],
    group_values: dict[str, str],
) -> pd.DataFrame:
    """Return security-level rows matching all group column values.

    Like _drill_down but filters on multiple columns simultaneously.
    """
    mask = pd.Series(True, index=positional_data.index)
    for col, val in group_values.items():
        mask &= positional_data[col] == val
    filtered = positional_data[mask].copy()

    keep_ids = {"Security", "Ticker", "Isin"} | set(group_cols)
    drop_cols = [
        c for c in filtered.columns
        if c.startswith(("Portfolio ", "Benchmark ", "Active "))
        and "Contribution" not in c
        and "Market Value Percent" not in c
    ]
    drop_cols += [c for c in IDENTIFIERS if c not in keep_ids and c in filtered.columns]
    return filtered.drop(columns=drop_cols)


# ═══════════════════════════════════════════════════════════════════════
# LLM Prompts & Interpretation
# ═══════════════════════════════════════════════════════════════════════

INTERPRETATION_PROMPT = """<think>

</think>

<task>
You are a senior market/credit risk analyst. Interpret the analysis results below and provide clear, insightful commentary to the question. The data is in YAML format.

<question>
{question}
</question>

<portfolio_context>
{portfolio_context}
</portfolio_context>

<grouped_analysis>
{grouped_context}
</grouped_analysis>

<drill_down_positions>
{drill_down_context}
</drill_down_positions>

<requirements>
- Do not write any code.
- Keep the commentary under 300 words.
- Structure the response with bullet points for key findings.
- Any column suffixed with "Change in 2 weeks" represents the two-week delta, i.e., the net change in that metric relative to its value two weeks ago.
- If (Portfolio Market Value Percent)==(Portfolio Market Value Percent Change in 2 weeks) then a new position has been added to the portfolio in the last 2 weeks.
- If (Benchmark Market Value Percent)==(Benchmark Market Value Percent Change in 2 weeks) then a new position has been added to the benchmark in the last 2 weeks.
- If (Portfolio Market Value Percent) is 0 or None and (Portfolio Market Value Percent Change in 2 weeks) is negative, the position has been removed from the portfolio in the last 2 weeks.
- If (Benchmark Market Value Percent) is 0 or None and (Benchmark Market Value Percent Change in 2 weeks) is negative, the position has been removed from the benchmark in the last 2 weeks.
- For Duration, Spread, Spread Duration, and Duration Times Spread: increases in 2-week changes indicate worsening risk; decreases indicate improving risk.
- For Contribution columns (e.g., DTS Contribution): a positive change means increased credit risk exposure (deteriorating credit quality); a negative change means reduced credit risk exposure (improving credit quality). Do NOT invert this — positive DTS Contribution Change = worsening, negative = improving.
- Ignore changes in Duration, Spread, Spread Duration, and Duration Times Spread for positions that were newly added or removed in the last 2 weeks; note the newly added or removed positions only if they contribute meaningfully, otherwise ignore.
- Use the grouped analysis for high-level themes and the drill-down positions for security-level color.
</requirements>

<metric_glossary>
- Duration: Measures a bond's price sensitivity to interest rate changes (in years). Higher duration = greater interest rate risk.
- Spread: The yield premium over a risk-free benchmark (in basis points). Reflects credit risk; wider spread = higher perceived credit risk.
- Spread Duration: Duration x sensitivity to spread changes. Measures how much a bond's price moves for a 1bp change in credit spread.
- Duration Times Spread (DTS): Duration x Spread. A composite measure of credit risk exposure that captures both spread level and spread sensitivity. Higher DTS = greater credit risk contribution.
- Contribution columns: The position's weighted contribution to the portfolio/benchmark/active total for that metric (= position weight x metric value).
- Market Value Percent: The position's weight in the portfolio or benchmark as a percentage of total market value.
- Active: Portfolio value minus Benchmark value. Positive = overweight or higher metric vs benchmark.
- Security-level metrics: The bond's own Duration, Spread, Spread Duration, or DTS (falling back from Portfolio to Benchmark value when the position is benchmark-only).
</metric_glossary>

<output_format>
Start with a bold header "**Key Findings:**" on its own line.
Then list findings as bullet points using "- " prefix.
Do not use asterisks (*) for bullets.
</output_format>"""

SUMMARY_PROMPT = """<think>

</think>

<task>
You are a senior market/credit risk analyst. Summarise the analysis commentaries below
into a single concise executive summary for portfolio '{portfolio_name}'.

<commentaries>
{commentaries}
</commentaries>

<requirements>
- Do not write any code.
- Keep the summary under 600 words.
- Focus only on the most important themes: key risk drivers, notable position changes,
  and the overall direction of active credit risk (DTS) relative to the benchmark.
- Structure with a short opening sentence, then bullet points for key themes.
- Use plain language; avoid repeating raw numbers unless they are critical.
</requirements>

<output_format>
Start with a bold header "**Executive Summary:**" on its own line.
Then a one-sentence overview.
Then list key themes as bullet points using "- " prefix.
Do not use asterisks (*) for bullets.
</output_format>"""


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _call_llm(prompt: str, model: str = INTERPRETER_MODEL) -> str:
    """Send a prompt to the local Ollama LLM and return the response text.

    Retries up to LLM_MAX_RETRIES times with exponential backoff on failure.
    """
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=INTERPRETER_OPTIONS,
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            wait = LLM_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, LLM_MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} attempts") from last_exc


def interpret_result(
    question: str,
    portfolio_context: str,
    grouped_context: str,
    drill_down_context: str,
    *,
    model: str = INTERPRETER_MODEL,
) -> str:
    """Send portfolio, grouped, and drill-down context to the LLM for commentary."""
    prompt = INTERPRETATION_PROMPT.format(
        question=question,
        portfolio_context=portfolio_context,
        grouped_context=grouped_context,
        drill_down_context=drill_down_context,
    )
    return _call_llm(prompt, model)


def _generate_portfolio_summary(
    portfolio_name: str,
    sections: list[dict],
    *,
    model: str = INTERPRETER_MODEL,
) -> str:
    """Generate an executive summary from all section commentaries."""
    commentaries = "\n\n".join(
        f"[{s['dimension']} / {s['group_value']}]\n{_strip_think_tags(s['commentary'])}"
        for s in sections
    )
    prompt = SUMMARY_PROMPT.format(
        portfolio_name=portfolio_name,
        commentaries=commentaries,
    )
    return _call_llm(prompt, model)


# ═══════════════════════════════════════════════════════════════════════
# Dimension Processing (Analysis Pipeline)
# ═══════════════════════════════════════════════════════════════════════


def _determine_direction(portfolio_data: pd.DataFrame) -> str:
    """Return 'bottom' if the first ranking metric is negative, else 'top'."""
    metric = RANKING_METRICS[0] if RANKING_METRICS else None
    if metric is None or metric not in portfolio_data.columns:
        return "top"
    val = portfolio_data[metric].iloc[0]
    return "bottom" if (val is not None and val < 0) else "top"


def _ranking_label() -> str:
    """Return a human-readable label for the current ranking metrics."""
    return ", ".join(RANKING_METRICS)


def _build_question(dim: str, group_value: str, direction: str, portfolio_name: str) -> str:
    """Build the interpretation question for a specific group."""
    rank_label = _ranking_label()
    return (
        f"For portfolio '{portfolio_name}', analyse the {dim} group '{group_value}'. "
        f"The grouped data shows the {direction} 5 contributing {dim} groups "
        f"ranked by [{rank_label}]. "
        f"The drill-down shows the {direction} 5 positions within '{group_value}'. "
        "Provide insights on why this group and its key positions contributed to "
        f"[{rank_label}]. "
        "Consider Active Market Value Percent and its change in 2 weeks."
    )


def _process_dimension(
    dim: str,
    positional_data: pd.DataFrame,
    portfolio_yaml: str,
    direction: str,
    portfolio_name: str,
) -> tuple[list[str], list[dict]]:
    """Process one dimension: group, select top/bottom, drill down, interpret.

    Returns:
        text_parts: Plain-text output for each group.
        structured_parts: Dicts with dimension, group_value, commentary, and table data.
    """
    grouped = _prepare_grouped_data(positional_data, [dim])

    # Drop derived columns not needed for LLM context
    drop_cols = [
        c for c in grouped.columns
        if c.startswith("Weighted Average ") or c.startswith("Security ")
    ]
    grouped = grouped.drop(columns=drop_cols).reset_index(drop=True)

    # High-cardinality dims: top/bottom 5; low-cardinality: all groups
    if dim in HIGH_CARDINALITY_DIMS:
        selected = _top_bottom(grouped, RANKING_METRICS, 5, direction)
    else:
        selected = grouped

    group_values = selected[dim].dropna().tolist()
    selected_yaml = _to_yaml(selected)
    grouped_table = _extract_table_data(selected, dim)
    logger.info("Selected %d groups for %s: %s", len(group_values), dim, group_values)

    text_parts: list[str] = []
    structured_parts: list[dict] = []

    def _interpret_group(group_value: str) -> dict:
        """Prepare data and call LLM for a single group value."""
        drilldown = _drill_down(positional_data, dim, group_value).reset_index(drop=True)
        drilldown_sample = _top_bottom(drilldown, RANKING_METRICS, 5, direction)
        drilldown_table = _extract_table_data(drilldown_sample, "Security")
        drilldown_yaml = _to_yaml(drilldown_sample)
        question = _build_question(dim, group_value, direction, portfolio_name)

        try:
            t0 = time.time()
            commentary = interpret_result(
                question, portfolio_yaml, selected_yaml, drilldown_yaml,
            )
            logger.info("  %s = %s interpreted in %.2fs", dim, group_value, time.time() - t0)
        except Exception as exc:
            logger.error("  Interpretation failed for %s = %s: %s", dim, group_value, exc)
            commentary = f"ERROR: {exc}"

        return {
            "group_value": group_value,
            "commentary": commentary,
            "drilldown_table": drilldown_table,
        }

    # Run LLM interpretations concurrently
    results_by_group: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
        futures = {
            executor.submit(_interpret_group, gv): gv for gv in group_values
        }
        for future in as_completed(futures):
            result = future.result()
            results_by_group[result["group_value"]] = result

    # Reassemble in original order
    for i, group_value in enumerate(group_values, 1):
        result = results_by_group[group_value]
        commentary = result["commentary"]

        logger.info("  [%d/%d] %s = %s done", i, len(group_values), dim, group_value)
        print(f"\n--- Commentary for {dim} = {group_value} ---")
        print(commentary)

        header = f"\n{'=' * 80}\nAnalysis for {dim} = {group_value}\n{'=' * 80}\n\n"
        text_parts.append(f"{header}COMMENTARY:\n{commentary}\n")
        structured_parts.append({
            "dimension": dim,
            "group_value": group_value,
            "commentary": commentary,
            "grouped_table": grouped_table,
            "drilldown_table": result["drilldown_table"],
        })

    return text_parts, structured_parts


def _process_multi_dimension(
    dims: list[str],
    positional_data: pd.DataFrame,
    portfolio_yaml: str,
    direction: str,
    portfolio_name: str,
) -> tuple[list[str], list[dict]]:
    """Process multiple dimensions simultaneously.

    Groups data by all *dims* columns at once, producing one row per unique
    combination (e.g., Sleeve x Rating_l1 x Industry).  Selects the top/bottom
    MULTI_DIM_TOP_N combos by DTS Contribution Change, then interprets each.

    Returns:
        text_parts: Plain-text output for each combo.
        structured_parts: Dicts with dimension, group_value, commentary, and table data.
    """
    dim_label = " x ".join(dims)
    grouped = _prepare_grouped_data(positional_data, dims)

    # Drop derived columns not needed for LLM context
    drop_cols = [
        c for c in grouped.columns
        if c.startswith("Weighted Average ") or c.startswith("Security ")
    ]
    grouped = grouped.drop(columns=drop_cols).reset_index(drop=True)

    # Build a combined label column for display
    grouped["_combo_label"] = grouped[dims].apply(
        lambda row: " | ".join(f"{d}={row[d]}" for d in dims), axis=1,
    )

    # Select top/bottom N combos
    selected = _top_bottom(grouped, RANKING_METRICS, MULTI_DIM_TOP_N, direction)

    # Build list of combo dicts (column -> value) in original order
    combos: list[dict[str, str]] = []
    for _, row in selected.iterrows():
        combo = {d: row[d] for d in dims}
        if combo not in combos:
            combos.append(combo)

    selected_yaml = _to_yaml(selected.drop(columns=["_combo_label"]))
    grouped_table = _extract_table_data(selected, "_combo_label")
    logger.info(
        "Selected %d combos for [%s]", len(combos), dim_label,
    )

    text_parts: list[str] = []
    structured_parts: list[dict] = []

    def _interpret_combo(combo: dict[str, str]) -> dict:
        combo_label = " | ".join(f"{d}={v}" for d, v in combo.items())
        drilldown = _drill_down_multi(
            positional_data, dims, combo,
        ).reset_index(drop=True)
        drilldown_sample = _top_bottom(drilldown, RANKING_METRICS, 5, direction)
        drilldown_table = _extract_table_data(drilldown_sample, "Security")
        drilldown_yaml = _to_yaml(drilldown_sample)

        rank_label = _ranking_label()
        question = (
            f"For portfolio '{portfolio_name}', analyse the multi-dimension group "
            f"[{combo_label}]. "
            f"The grouped data shows the {direction} {MULTI_DIM_TOP_N} contributing "
            f"combinations of {dim_label} ranked by [{rank_label}]. "
            f"The drill-down shows the {direction} 5 positions within [{combo_label}]. "
            "Provide insights on why this combination and its key positions "
            f"contributed to [{rank_label}]. "
            "Consider Active Market Value Percent and its change in 2 weeks."
        )

        try:
            t0 = time.time()
            commentary = interpret_result(
                question, portfolio_yaml, selected_yaml, drilldown_yaml,
            )
            logger.info(
                "  [%s] interpreted in %.2fs", combo_label, time.time() - t0,
            )
        except Exception as exc:
            logger.error("  Interpretation failed for [%s]: %s", combo_label, exc)
            commentary = f"ERROR: {exc}"

        return {
            "combo_label": combo_label,
            "commentary": commentary,
            "drilldown_table": drilldown_table,
        }

    # Run LLM interpretations concurrently
    results_by_label: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
        futures = {
            executor.submit(_interpret_combo, combo): combo for combo in combos
        }
        for future in as_completed(futures):
            result = future.result()
            results_by_label[result["combo_label"]] = result

    # Reassemble in original order
    for i, combo in enumerate(combos, 1):
        combo_label = " | ".join(f"{d}={v}" for d, v in combo.items())
        result = results_by_label[combo_label]
        commentary = result["commentary"]

        logger.info("  [%d/%d] %s done", i, len(combos), combo_label)
        print(f"\n--- Commentary for [{combo_label}] ---")
        print(commentary)

        header = (
            f"\n{'=' * 80}\nAnalysis for [{combo_label}]\n{'=' * 80}\n\n"
        )
        text_parts.append(f"{header}COMMENTARY:\n{commentary}\n")
        structured_parts.append({
            "dimension": dim_label,
            "group_value": combo_label,
            "commentary": commentary,
            "grouped_table": grouped_table,
            "drilldown_table": result["drilldown_table"],
        })

    return text_parts, structured_parts


# ═══════════════════════════════════════════════════════════════════════
# PDF Rendering
# ═══════════════════════════════════════════════════════════════════════


def _extract_table_data(
    df: pd.DataFrame,
    name_col: str,
    max_name_len: int = 30,
    cols: list | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Extract headers and formatted row data from a DataFrame for PDF tables."""
    col_spec = cols if cols is not None else PDF_TABLE_COLS
    headers = [name_col] + [short for _, short, _ in col_spec]
    rows = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "") or "")
        if len(name) > max_name_len:
            name = name[: max_name_len - 3] + "..."
        cells = [name]
        for full_col, _, fmt in col_spec:
            val = row.get(full_col, None)
            if val is not None and pd.notna(val):
                try:
                    cells.append(f"{float(val):{fmt}}")
                except (ValueError, TypeError):
                    cells.append(str(val))
            else:
                cells.append("-")
        rows.append(cells)
    return headers, rows


def _pdf_render_table(
    pdf: FPDF,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    name_width: int = 38,
) -> None:
    """Render a data table with header bar and alternating row colors."""
    if not rows:
        return

    n_data = len(headers) - 1
    avail = pdf.w - pdf.l_margin - pdf.r_margin - name_width
    col_w = avail / n_data if n_data else avail

    max_lines = max(h.count("\n") + 1 for h in headers)
    header_h = max(6, max_lines * 4 + 2)

    # Ensure space for title + header + at least one data row
    min_space = header_h + 5 + (7 if title else 0)
    if pdf.get_y() + min_space > pdf.h - pdf.b_margin:
        pdf.add_page()

    if title:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def draw_header():
        y_start = pdf.get_y()
        prev_margin = pdf.b_margin
        pdf.set_auto_page_break(auto=False)
        pdf.set_font("Helvetica", "B", 6)
        pdf.set_fill_color(*PDF_HEADER_BG)
        pdf.set_text_color(*PDF_HEADER_FG)

        x = pdf.l_margin
        pdf.rect(x, y_start, name_width, header_h, style="DF")
        pdf.set_xy(x + 1, y_start + 1)
        pdf.multi_cell(name_width - 2, 4, headers[0], align="L")
        x += name_width

        for h in headers[1:]:
            pdf.rect(x, y_start, col_w, header_h, style="DF")
            pdf.set_xy(x, y_start + 1)
            pdf.multi_cell(col_w, 4, h, align="C")
            x += col_w

        pdf.set_xy(pdf.l_margin, y_start + header_h)
        pdf.set_text_color(0, 0, 0)
        pdf.set_auto_page_break(auto=True, margin=prev_margin)

    draw_header()
    pdf.set_font("Helvetica", "", 7)

    for i, row in enumerate(rows):
        if pdf.get_y() + 5 > pdf.h - pdf.b_margin:
            pdf.add_page()
            draw_header()
            pdf.set_font("Helvetica", "", 7)
        fill = i % 2 == 1
        if fill:
            pdf.set_fill_color(*PDF_ALT_ROW_BG)
        pdf.cell(name_width, 5, f" {row[0]}", border=1, fill=fill)
        for val in row[1:]:
            pdf.cell(col_w, 5, val, border=1, fill=fill, align="R")
        pdf.ln()
    pdf.ln(4)


def _pdf_render_metadata_table(pdf: FPDF, metadata: pd.DataFrame) -> None:
    """Render portfolio metadata (from Excel C3:G4) as a centered table."""
    headers = [str(v) for v in metadata.iloc[0].tolist()]
    values = [str(v) for v in metadata.iloc[1].tolist()]
    total_width = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = total_width / len(headers)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*PDF_HEADER_BG)
    pdf.set_text_color(*PDF_HEADER_FG)
    for h in headers:
        pdf.cell(col_w, 6, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(*PDF_ALT_ROW_BG)
    for v in values:
        pdf.cell(col_w, 6, v, border=1, fill=True, align="C")
    pdf.ln()


def _pdf_render_info_box(pdf: FPDF, title: str, items: dict[str, str]) -> None:
    """Render a key-value info box."""
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*PDF_HEADER_BG)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    for key, val in items.items():
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(40, 5, f"  {key}:")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5, val, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


_UNICODE_REPLACEMENTS = {
    "\u20ac": "EUR",  # €
    "\u2013": "-",    # en dash
    "\u2014": "--",   # em dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # ellipsis
    "\u00a3": "GBP",  # £
    "\u00a5": "JPY",  # ¥
}


def _sanitize_for_pdf(text: str) -> str:
    """Replace Unicode characters unsupported by core PDF fonts."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text


def _pdf_render_commentary(pdf: FPDF, commentary: str) -> None:
    """Render LLM commentary with bullet points and markdown bold formatting."""
    commentary = _sanitize_for_pdf(commentary)
    pdf.set_font("Helvetica", "", 10)

    for line in commentary.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue

        # Bullet points
        if line.startswith(("- ", "* ", "\u2022 ")):
            body = line[2:]
            colon_pos = body.find(":")
            pdf.set_x(pdf.l_margin + 6)
            pdf.set_font("Helvetica", "", 10)
            pdf.write(5, "\xb7 ")
            if 0 < colon_pos < 60:
                pdf.set_font("Helvetica", "B", 10)
                pdf.write(5, body[: colon_pos + 1])
                pdf.set_font("Helvetica", "", 10)
                pdf.write(5, body[colon_pos + 1 :])
                pdf.ln(5)
            else:
                pdf.multi_cell(0, 5, body)

        # Markdown headings
        elif line.startswith("#"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, line.lstrip("# ").strip())
            pdf.set_font("Helvetica", "", 10)

        # Inline **bold** text
        else:
            bold_parts = re.split(r"(\*\*.*?\*\*)", line)
            if len(bold_parts) > 1:
                for part in bold_parts:
                    if part.startswith("**") and part.endswith("**"):
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.write(5, part[2:-2])
                        pdf.set_font("Helvetica", "", 10)
                    else:
                        pdf.write(5, part)
                pdf.ln(5)
            else:
                pdf.multi_cell(0, 5, line)


def _pdf_render_section(pdf: FPDF, section: dict) -> None:
    """Render one dimension group: sub-header, drill-down table, commentary."""
    commentary = _strip_think_tags(section["commentary"])

    # Group sub-header
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*PDF_HEADER_BG)
    pdf.cell(0, 8, section["group_value"], new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.set_draw_color(*PDF_SEPARATOR_COLOR)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)

    # Security-level drill-down table
    dd_headers, dd_rows = section["drilldown_table"]
    _pdf_render_table(
        pdf,
        "Top/Bottom Positions based on Active DTS Contrib Change in 2 weeks",
        dd_headers, dd_rows, name_width=42,
    )

    _pdf_render_commentary(pdf, commentary)
    pdf.ln(6)


# ── Full PDF Reports ────────────────────────────────────────────────


def _generate_detail_pdf(
    portfolio_sections: list[tuple[str, pd.DataFrame, pd.DataFrame, list[dict]]],
    output_path: Path,
) -> None:
    """Generate the detailed PDF report with tables and commentary per dimension."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    for portfolio, metadata, port_summary, sections in portfolio_sections:
        # Title page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 28)
        pdf.ln(20)
        pdf.cell(0, 15, "Risk Commentary Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(8)

        _pdf_render_metadata_table(pdf, metadata)
        pdf.ln(6)

        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 12, f"Portfolio: {portfolio}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(
            0, 10,
            f"Generated: {time.strftime('%d %B %Y')}",
            new_x="LMARGIN", new_y="NEXT", align="C",
        )

        # Portfolio summary table
        pdf.ln(12)
        p_headers, p_rows = _extract_table_data(
            port_summary, "Security", cols=PDF_PORTFOLIO_COLS,
        )
        _pdf_render_table(pdf, "Portfolio Summary", p_headers, p_rows, name_width=30)

        # Content pages by dimension
        current_dim = None
        for section in sections:
            dim = section["dimension"]
            if dim != current_dim:
                pdf.add_page()
                current_dim = dim

                # Dimension header bar
                pdf.set_fill_color(*PDF_HEADER_BG)
                pdf.set_text_color(*PDF_HEADER_FG)
                pdf.set_font("Helvetica", "B", 18)
                pdf.cell(0, 14, f"  {dim}", new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.ln(4)
                pdf.set_text_color(0, 0, 0)

                g_headers, g_rows = section["grouped_table"]
                _pdf_render_table(pdf, "Dimension Summary", g_headers, g_rows, name_width=40)

            _pdf_render_section(pdf, section)

    pdf.output(str(output_path))
    logger.info("PDF saved to %s", output_path)


def _generate_summary_pdf(
    portfolio_summaries: list[tuple[str, pd.DataFrame, pd.DataFrame, str]],
    output_path: Path,
) -> None:
    """Generate a summary PDF with one page per portfolio plus model/hardware info."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # Info page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 12, "All Portfolio Summary", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    _pdf_render_info_box(pdf, "Data Source", {
        "Source": (
            "BlackRock Aladdin Explore "
            "(Workspace - https://barings.blackrock.com/apps/explore/?workspace=53605)"
        ),
        "Disclaimer": (
            "This analysis was automatically generated using local LLM models via Ollama"
        ),
    })
    pdf.ln(4)
    _pdf_render_info_box(pdf, "Software Used", _get_model_info())
    pdf.ln(4)
    _pdf_render_info_box(pdf, "Hardware Used", _get_hardware_info())

    # Per-portfolio pages
    for portfolio, metadata, port_summary, summary_text in portfolio_summaries:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 12, f"Portfolio: {portfolio}", new_x="LMARGIN", new_y="NEXT", align="L")
        pdf.ln(4)

        _pdf_render_metadata_table(pdf, metadata)
        pdf.ln(4)

        p_headers, p_rows = _extract_table_data(
            port_summary, "Security", cols=PDF_PORTFOLIO_COLS,
        )
        _pdf_render_table(pdf, "Portfolio Summary", p_headers, p_rows, name_width=30)
        pdf.ln(4)

        _pdf_render_commentary(pdf, _strip_think_tags(summary_text))

    pdf.output(str(output_path))
    logger.info("Summary PDF saved to %s", output_path)


# ═══════════════════════════════════════════════════════════════════════
# System & Model Metadata
# ═══════════════════════════════════════════════════════════════════════


def _get_hardware_info() -> dict[str, str]:
    """Collect CPU, RAM, and GPU details from the local machine."""
    info: dict[str, str] = {"CPU": platform.processor() or "Unknown"}

    try:
        import psutil
        info["RAM"] = f"{psutil.virtual_memory().total / (1024 ** 3):.1f} GB"
    except ImportError:
        info["RAM"] = "Unknown (psutil not installed)"

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                gpus.append(
                    f"{parts[0]} ({parts[1]} MB)" if len(parts) == 2 else line.strip()
                )
            info["GPU"] = "; ".join(gpus)
        else:
            info["GPU"] = "No NVIDIA GPU detected"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["GPU"] = "No NVIDIA GPU detected"

    return info


def _get_model_info(model: str = INTERPRETER_MODEL) -> dict[str, str]:
    """Collect LLM model metadata from Ollama."""
    info: dict[str, str] = {
        "LLM Model": model,
        "Runtime": "Ollama (local)",
        "Temperature": str(INTERPRETER_OPTIONS.get("temperature", "N/A")),
    }
    try:
        model_details = ollama.show(model)
        model_info = model_details.get("model_info", {})
        for key, val in model_info.items():
            if "context_length" in key:
                info["Context Length"] = str(val)
                break
        details = model_details.get("details", {})
        if details.get("parameter_size"):
            info["Parameter Size"] = details["parameter_size"]
        if details.get("quantization_level"):
            info["Quantization"] = details["quantization_level"]
    except Exception:
        info["Context Length"] = "Unknown"
    return info


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Orchestration
# ═══════════════════════════════════════════════════════════════════════


def run_portfolio(
    portfolio: str,
    dims: list[str] | None = None,
    multi_dims: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Run the full analysis pipeline for a single portfolio.

    Args:
        portfolio: Portfolio identifier.
        dims: Single dimensions to process individually. Defaults to ALL_DIMS.
        multi_dims: If provided, also run a multi-dimension cross-analysis
                    grouping by all listed columns simultaneously.

    Returns (metadata, port_summary, sections) for PDF generation.
    """
    if dims is None:
        dims = ALL_DIMS
    data_path = _data_path(portfolio)

    logger.info("Loading data for %s...", portfolio)
    t0 = time.time()
    metadata = _read_metadata(data_path)
    raw_df = _read_raw_data(data_path)
    portfolio_totals = _prepare_portfolio_totals(raw_df)
    port_summary = _scale_mv_percent(raw_df.iloc[:1].copy())
    positional_data = _prepare_positional_data(raw_df)
    del raw_df
    logger.info("Data loaded in %.2f seconds (%d positions)", time.time() - t0, len(positional_data))

    portfolio_yaml = _to_yaml(portfolio_totals)
    direction = _determine_direction(portfolio_totals)

    output_parts: list[str] = []
    all_sections: list[dict] = []

    # Single-dimension analyses
    for dim in dims:
        logger.info("=" * 60)
        logger.info("Processing dimension: %s", dim)
        logger.info("=" * 60)
        text_parts, structured_parts = _process_dimension(
            dim, positional_data, portfolio_yaml, direction, portfolio,
        )
        output_parts.extend(text_parts)
        all_sections.extend(structured_parts)

    # Multi-dimension cross-analysis
    if multi_dims and len(multi_dims) >= 2:
        dim_label = " x ".join(multi_dims)
        logger.info("=" * 60)
        logger.info("Processing multi-dimension: %s", dim_label)
        logger.info("=" * 60)
        text_parts, structured_parts = _process_multi_dimension(
            multi_dims, positional_data, portfolio_yaml, direction, portfolio,
        )
        output_parts.extend(text_parts)
        all_sections.extend(structured_parts)

    # Save text output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _output_path(portfolio).write_text("".join(output_parts), encoding="utf-8")
    logger.info("Results saved to %s", _output_path(portfolio))

    return metadata, port_summary, all_sections


DIMS_CHOICES = {
    "all": ALL_DIMS,
    "high": HIGH_CARDINALITY_DIMS,
    "low": LOW_CARDINALITY_DIMS,
}


def main() -> None:
    """Entry point: process all portfolios and generate PDF reports."""
    parser = argparse.ArgumentParser(description="Risk commentary report generator.")
    parser.add_argument(
        "--dims",
        choices=DIMS_CHOICES,
        default="all",
        help="Which dimension set to run: all, high, or low (default: all)",
    )
    parser.add_argument(
        "--portfolios",
        nargs="+",
        default=PORTFOLIOS,
        help=f"Portfolio(s) to process (default: {PORTFOLIOS})",
    )
    parser.add_argument(
        "--model",
        default=INTERPRETER_MODEL,
        help=f"Ollama model to use (default: {INTERPRETER_MODEL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_LLM_WORKERS,
        help=f"Max concurrent LLM calls per dimension (default: {MAX_LLM_WORKERS})",
    )
    parser.add_argument(
        "--multi-dims",
        nargs="+",
        default=None,
        metavar="DIM",
        help=(
            "Run an additional multi-dimension cross-analysis grouping by all "
            "listed columns. E.g. --multi-dims Sleeve Rating_l1 Industry"
        ),
    )
    parser.add_argument(
        "--multi-top-n",
        type=int,
        default=MULTI_DIM_TOP_N,
        help=f"Max combos to analyse in multi-dim mode (default: {MULTI_DIM_TOP_N})",
    )
    parser.add_argument(
        "--rank-by",
        nargs="+",
        default=None,
        metavar="METRIC",
        help=(
            "Ranking metric(s) for selecting top/bottom groups. "
            "Default: 'Active Duration Times Spread Contribution Change in 2 weeks'. "
            "E.g. --rank-by 'Active Risk Contribution Change in 2 weeks'"
        ),
    )
    args = parser.parse_args()
    selected_dims = DIMS_CHOICES[args.dims]

    # Apply runtime overrides
    global INTERPRETER_MODEL, MAX_LLM_WORKERS, MULTI_DIM_TOP_N, RANKING_METRICS  # noqa: PLW0603
    INTERPRETER_MODEL = args.model
    MAX_LLM_WORKERS = args.workers
    MULTI_DIM_TOP_N = args.multi_top_n
    if args.rank_by:
        RANKING_METRICS = args.rank_by

    total_start = time.time()

    # Pre-warm interpreter model
    logger.info("Pre-warming model %s...", INTERPRETER_MODEL)
    try:
        ollama.chat(
            model=INTERPRETER_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            keep_alive="30m",
        )
    except Exception:
        pass
    logger.info("Model ready.")

    # Process each portfolio
    all_portfolio_sections: list[tuple[str, pd.DataFrame, pd.DataFrame, list[dict]]] = []
    for portfolio in args.portfolios:
        logger.info("#" * 60)
        logger.info("PORTFOLIO: %s", portfolio)
        logger.info("#" * 60)
        try:
            metadata, port_summary, sections = run_portfolio(
                portfolio, dims=selected_dims, multi_dims=args.multi_dims,
            )
            all_portfolio_sections.append((portfolio, metadata, port_summary, sections))
        except FileNotFoundError as exc:
            logger.error("Skipping %s: %s", portfolio, exc)

    if not all_portfolio_sections:
        logger.warning("No portfolios processed successfully.")
        return

    # Generate detailed PDF
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _generate_detail_pdf(all_portfolio_sections, OUTPUT_DIR / "analysis_results.pdf")

    # Generate executive summary PDF
    summary_data: list[tuple[str, pd.DataFrame, pd.DataFrame, str]] = []
    for portfolio, metadata, port_summary, sections in all_portfolio_sections:
        logger.info("Generating executive summary for %s...", portfolio)
        t0 = time.time()
        summary_text = _generate_portfolio_summary(portfolio, sections)
        logger.info("Summary generated in %.2f seconds", time.time() - t0)
        summary_data.append((portfolio, metadata, port_summary, summary_text))

    _generate_summary_pdf(summary_data, OUTPUT_DIR / "All Portfolio Summary.pdf")

    logger.info("Total elapsed: %.2f seconds", time.time() - total_start)


if __name__ == "__main__":
    main()
    # python risk_agent_data.py              # runs all dims (default)
    # python risk_agent_data.py --dims low   # only Sleeve, Rating_l1
    # python risk_agent_data.py --dims high  # only Ticker, Industry, Country Of Risk
    # python risk_agent_data.py --dims low --portfolios BCGHYBU
    # python risk_agent_data.py --model llama3:8b --workers 5
    # python risk_agent_data.py --portfolios BCGHYBU BCEHYBF --dims high
    #
    # Multi-dimension examples:
    # python risk_agent_data.py --multi-dims Sleeve Rating_l1
    # python risk_agent_data.py --multi-dims Sleeve Rating_l1 Industry
    # python risk_agent_data.py --multi-dims Sleeve Rating_l1 Industry --multi-top-n 20
    # python risk_agent_data.py --dims low --multi-dims Sleeve Rating_l1 Industry
    #
    # Custom ranking metric(s):
    # python risk_agent_data.py --rank-by "Active Risk Contribution Change in 2 weeks"
    # python risk_agent_data.py --rank-by "Active Duration Times Spread Contribution Change in 2 weeks" "Active Risk Contribution Change in 2 weeks"
