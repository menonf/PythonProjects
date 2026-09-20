"""DB-free modelling table: point-in-time features joined to completed forward returns."""

import numpy as np
import pandas as pd

from analytics.factors.ml_training import build_modelling_table_from_panel
from data_engineering.fundamentals import RATIO_COLUMNS, InMemoryFundamentalsProvider
from tests.unit_tests.synthetic import synthetic_fundamentals_history, synthetic_prices


def test_modelling_table_shape_and_no_lookahead() -> None:
    prices = synthetic_prices(n_securities=20, n_days=520, seed=3)  # ~2 years
    quarter_ends = pd.date_range(prices.index[0] - pd.offsets.QuarterEnd(1), prices.index[-1], freq="QE")
    history = synthetic_fundamentals_history(list(prices.columns), quarter_ends)
    provider = InMemoryFundamentalsProvider(history, availability_lag_days=45)

    table = build_modelling_table_from_panel(prices, provider, None, str(prices.index[0].date()), str(prices.index[-1].date()), forward_months=3, demean_target=True)
    assert set(["security_id", "snapshot_date", "y"] + RATIO_COLUMNS).issubset(table.columns)
    assert table["snapshot_date"].isin(prices.index).all()
    # labels need a completed 3-month forward window: no snapshot in the last ~3 months
    assert table["snapshot_date"].max() <= prices.index[-1] - pd.DateOffset(months=3) + pd.Timedelta(days=7)
    # demeaned target averages to ~0 per snapshot
    assert np.allclose(table.groupby("snapshot_date")["y"].mean(), 0, atol=1e-12)
    # features on a snapshot come from the latest quarter published >= 45 days before it
    snap = table["snapshot_date"].iloc[-1]
    visible = history[history["effective_date"] + pd.Timedelta(days=45) <= snap]
    latest = visible.sort_values("effective_date").groupby("security_id").tail(1).set_index("security_id")
    row = table[table["snapshot_date"] == snap].set_index("security_id")
    common = row.index.intersection(latest.index)
    np.testing.assert_allclose(row.loc[common, "P/B"], latest.loc[common, "P/B"])
    assert not table[RATIO_COLUMNS].isna().any().any()  # median imputed
