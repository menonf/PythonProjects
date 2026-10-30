# Databricks notebook source
# MAGIC %md # GEMS factor reconstruction from Quantlake
# MAGIC
# MAGIC Rebuilds the GEMS stock-level factors and composites **directly from Quantlake tables**, replicating the production
# MAGIC `PSA_Crisp_Metrics` code as coded, without the `barings_quant` data library.
# MAGIC
# MAGIC **Scope:** 126 of the 129 factors in the GEMS Factor Construction Library.
# MAGIC Not built: `beta`, `eps_volatility` and `sales_volatility`. Production fills these with placeholders (the local share price,
# MAGIC a copy of NTM EPS and a copy of NTM sales), so there is no real definition to rebuild.
# MAGIC
# MAGIC **Sources**
# MAGIC - Worldscope fundamentals: `marketdata.financials_metrics` (LSEG Worldscope via SAGE), field numbers via `settings.marketdata_metric_vendor_mappings`
# MAGIC - I/B/E/S: `marketdata.analyst_estimates_summary`, `marketdata.analyst_estimates_actuals_surprises`
# MAGIC - Prices: `marketdata.security_prices` (Worldscope weekly/monthly close, listing currency); FX: `<sage>.market.vfx_rate_fact` (WM11_SPOT_RATE)
# MAGIC - Universe: `investment.universe_securities` (qwd_global, id 163); limits: `settings.metric_rules`
# MAGIC
# MAGIC **Data repairs applied (differ from production on purpose)**
# MAGIC 1. Worldscope rows with no `statement_date` are dropped. These are old periods stamped with the load date (seen for TSMC on 26-May-2025 and 12-Jun-2026). Rows with fiscal_period `current` are kept.
# MAGIC 2. Annual rows dated off the company's fiscal year-end are dropped (e.g. TSMC's annual row on 31-Mar-2024).
# MAGIC 3. One value per company, field and period; the latest insert wins.
# MAGIC
# MAGIC **Known differences from production (cannot be replicated)**
# MAGIC - Missing values are not filled from the ultimate parent company (production's `map_parent_data`); no parent mapping is available.
# MAGIC - Selection rules and staleness windows of `barings_quant` (`Financials.get`, `Estimates.get`, `get_all_latest_prices`) are approximated (see loader constants).
# MAGIC - NTM EPS 1M/3M changes are recomputed from estimates rather than read from stored CRISP values.
# MAGIC
# MAGIC Run all cells top to bottom. Section 8 compares results with the values production stored for the same date.

# COMMAND ----------

# MAGIC %md ## 0. Parameters

# COMMAND ----------

dbutils.widgets.text("run_date", "2026-09-30", "Calculation date (YYYY-MM-DD)")
dbutils.widgets.text("quantlake_catalog", spark.conf.get("catalog.quantlake", "quantlake_pr"), "Quantlake catalog")
dbutils.widgets.text("sage_catalog", spark.conf.get("catalog.sage", "sage_pr"), "SAGE catalog (for FX)")
dbutils.widgets.dropdown("point_in_time", "false", ["false", "true"], "Only data inserted by run date (backtests)")
dbutils.widgets.dropdown("fix_qgarp_revisions", "false", ["false", "true"], "Fix QGARP revisions direction")
dbutils.widgets.text("output_table", "", "Optional output table (blank = do not write)")

import datetime as dt
RUN_DATE = dt.date.fromisoformat(dbutils.widgets.get("run_date"))
QL_CATALOG = dbutils.widgets.get("quantlake_catalog")
SAGE_CATALOG = dbutils.widgets.get("sage_catalog")
POINT_IN_TIME = dbutils.widgets.get("point_in_time") == "true"
FIX_QGARP = dbutils.widgets.get("fix_qgarp_revisions") == "true"
OUTPUT_TABLE = dbutils.widgets.get("output_table").strip()
print(RUN_DATE, QL_CATALOG, SAGE_CATALOG, POINT_IN_TIME, FIX_QGARP, OUTPUT_TABLE or "(no write)")

# COMMAND ----------

# MAGIC %md ## 1. Data loaders
# MAGIC Replace the `barings_quant` calls with direct queries. Point-in-time Worldscope values, I/B/E/S snapshots, prices with USD conversion, universe and identifiers.

# COMMAND ----------

# DBTITLE 1,Data loaders
# Data-access layer: replaces barings_quant calls with direct Quantlake / SAGE queries.
# Runs on Databricks (needs `spark`). Everything returned is pandas.
#
# Data rules applied (from the TSMC investigation and the ingestion repo):
#  * Worldscope rows with a NULL statement_date are history rows stamped with the load date
#    (real period = statement_year + value_sequence). They are EXCLUDED, except fiscal_period='current'
#    snapshot rows whose period is the load date by definition.
#  * Annual rows whose month differs from the company's usual fiscal-year-end month are EXCLUDED
#    (e.g. TSMC annual rows dated 31-Mar with values 3-5x the quarterly ones).
#  * One value per company / field / period: latest inserted_date wins.
#  * Optional point-in-time mode: only rows inserted on or before the calculation date.

import datetime as _dt
import pandas as pd
import numpy as np

STALENESS_DAYS = 550          # max age of a Worldscope value (period end) relative to the as-of date
FX_LOOKBACK_DAYS = 7          # max age of an FX rate
EST_LOOKBACK_DAYS = 14        # max age of an I/B/E/S snapshot
PRICE_LOOKBACK_DAYS = 30      # max age of the 'latest' price

FP_PRIORITY = "case lower(f.fiscal_period) when 'q' then 1 when 'current' then 2 when 's_a' then 3 when 'a' then 4 else 5 end"

INDUSTRY_CODES = {1: "INDUSTRIAL", 2: "UTILITY", 3: "TRANSPORTATION", 4: "BANK", 5: "INSURANCE", 6: "OTHER FINANCIAL"}
STRING_FIELDS = {"general_industry_classification", "primary_issue_identifier"}


def _d(x):
    return pd.Timestamp(x).date()


def _sqlist(values):
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def normalise_industry(v):
    """Worldscope WC06010 -> production label ('BANK', 'INSURANCE', ...). Handles codes or text."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper()
    try:
        return INDUSTRY_CODES.get(int(float(s)))
    except ValueError:
        pass
    if "BANK" in s or "SAVINGS" in s:
        return "BANK"
    if "INSUR" in s:
        return "INSURANCE"
    if "OTHER FIN" in s or s == "FINANCIAL":
        return "OTHER FINANCIAL"
    if "UTIL" in s:
        return "UTILITY"
    if "TRANSPORT" in s:
        return "TRANSPORTATION"
    if "INDUSTR" in s:
        return "INDUSTRIAL"
    return s


class QuantlakeLoader:
    def __init__(self, spark, ql_catalog="quantlake_pr", sage_catalog="sage_pr",
                 universe_id=163, universe_name="qwd_global", point_in_time=False):
        self.spark = spark
        self.QL = ql_catalog
        self.SAGE = sage_catalog
        self.universe_id = universe_id
        self.universe_name = universe_name
        self.point_in_time = point_in_time
        self._fx_cache = None
        self._vendors = spark.sql(f"select vendor_id, lower(vendor_name) vendor_name, lower(sub_vendor_name) sub_vendor_name "
                                  f"from {self.QL}.marketdata.vendors").toPandas()
        self.v_ws = self._vendor("refinitiv", "worldscope")
        self.v_ibes = self._vendor("refinitiv", "ibes")
        try:
            self.v_px = self._vendor(None, "psa_worldscope")
        except KeyError:
            self.v_px = 6  # hard-coded in ingestion (fundamentals_data.py get_worldscope_prices)
        self._register_metric_ids()

    # ------------------------------------------------------------------ setup
    def _vendor(self, name, sub):
        v = self._vendors
        m = v[(v.sub_vendor_name == sub) & ((v.vendor_name == name) if name else True)]
        if m.empty:
            raise KeyError(f"vendor {name}/{sub} not found")
        return int(m.vendor_id.iloc[0])

    def _register_metric_ids(self):
        # Vendor-specific lookup avoids the name clashes with computed factors
        # (current_ratio, market_capitalization, capital_adequacy_* exist as both).
        self.spark.sql(f"""
            select d.marketdata_metric_id, lower(d.barings_metric_name) as name, m.vendor_metric_name,
                   row_number() over (partition by lower(d.barings_metric_name) order by lpad(m.vendor_metric_name, 8, '0')) as id_priority
            from {self.QL}.settings.marketdata_metric_vendor_mappings m
            join {self.QL}.settings.marketdata_metrics_definitions d on d.marketdata_metric_id = m.marketdata_metric_id
            where m.vendor_id = {self.v_ws} and coalesce(m.is_active, true)
        """).createOrReplaceTempView("gems_ws_metric_ids")
        self.spark.sql(f"""
            select d.marketdata_metric_id, lower(m.vendor_metric_name) as measure
            from {self.QL}.settings.marketdata_metric_vendor_mappings m
            join {self.QL}.settings.marketdata_metrics_definitions d on d.marketdata_metric_id = m.marketdata_metric_id
            where m.vendor_id = {self.v_ibes} and coalesce(m.is_active, true)
              and lower(d.barings_metric_name) <> 'pharmaceutical_sales'   -- 'sal' is mapped twice
        """).createOrReplaceTempView("gems_ibes_metric_ids")

    # ------------------------------------------------------------------ universe & ids
    def build_universe(self, run_date):
        run_date = _d(run_date)
        QL = self.QL
        univ = self.spark.sql(f"""
            with u as (
                select security_id, datadate
                from {QL}.investment.universe_securities
                where (universe_id = {self.universe_id}
                       or universe_id in (select universe_id from {QL}.investment.universe
                                          where universe = '{self.universe_name}'))
                  and datadate <= '{run_date}'
            )
            select distinct security_id from u where datadate = (select max(datadate) from u)
        """)
        univ.createOrReplaceTempView("gems_univ")

        ids = self.spark.sql(f"""
            with ws_map as (
                select svm.security_id, coalesce(svm.vendor_code, evm.vendor_code) as worldscope_id, 1 as priority
                from {QL}.marketdata.securities_vendor_mappings svm
                left join {QL}.marketdata.entities_securities es on es.security_id = svm.security_id
                left join {QL}.marketdata.entities_vendor_mappings evm
                       on es.entity_id = evm.entity_id and es.entity_id != 1 and evm.vendor_id = {self.v_ws}
                where svm.vendor_id = {self.v_ws}
                union
                select es.security_id, coalesce(evm.vendor_code, svm.vendor_code) as worldscope_id, 2 as priority
                from {QL}.marketdata.entities_vendor_mappings evm
                join {QL}.marketdata.entities_securities es on es.entity_id = evm.entity_id and es.entity_id != 1
                left join {QL}.marketdata.securities_vendor_mappings svm
                       on es.security_id = svm.security_id and svm.vendor_id = {self.v_ws}
                where evm.vendor_id = {self.v_ws}
            ),
            ws as (
                select security_id, worldscope_id, count(*) over (partition by security_id) as n_ws_codes
                from ws_map where worldscope_id is not null
                qualify row_number() over (partition by security_id order by priority, worldscope_id) = 1
            ),
            ib as (
                select security_id, vendor_code as ibes_id
                from {QL}.marketdata.securities_vendor_mappings
                where vendor_id = {self.v_ibes} and vendor_code is not null
                qualify row_number() over (partition by security_id order by vendor_code) = 1
            ),
            msm as (
                select m.vendor_security_id, lower(d.barings_metric_name) as name, m.metric_value_str
                from {QL}.marketdata.marketdata_security_metrics m
                join {QL}.settings.marketdata_metrics_definitions d on d.marketdata_metric_id = m.marketdata_metric_id
                where lower(d.barings_metric_name) in ('worldscope_currency', 'ibes_currency', 'pricing_currency')
                qualify row_number() over (partition by m.vendor_security_id, d.barings_metric_name order by m.datadate desc) = 1
            )
            select u.security_id, s.security_name, s.sedol, s.isin,
                   ws.worldscope_id, ws.n_ws_codes, ib.ibes_id,
                   upper(wc.metric_value_str) as worldscope_currency,
                   upper(ic.metric_value_str) as ibes_currency,
                   upper(pc.metric_value_str) as pricing_currency
            from gems_univ u
            left join {QL}.marketdata.securities s on s.security_id = u.security_id
            left join ws on ws.security_id = u.security_id
            left join ib on ib.security_id = u.security_id
            left join msm wc on wc.name = 'worldscope_currency' and wc.vendor_security_id = ws.worldscope_id
            left join msm ic on ic.name = 'ibes_currency' and ic.vendor_security_id = ib.ibes_id
            left join msm pc on pc.name = 'pricing_currency' and pc.vendor_security_id = cast(u.security_id as string)
        """).toPandas()
        ids["security_id"] = ids["security_id"].astype(int)
        id_rows = ids[["security_id", "worldscope_id", "ibes_id"]].astype(object).where(
            ids[["security_id", "worldscope_id", "ibes_id"]].notna(), None).values.tolist()
        self.spark.createDataFrame(id_rows, "security_id long, worldscope_id string, ibes_id string") \
            .createOrReplaceTempView("gems_ids")

        # latest price on/before run date (production: get_all_latest_prices)
        px = self.get_prices(run_date - _dt.timedelta(days=PRICE_LOOKBACK_DAYS), run_date, ids=ids)
        px = px.sort_values("date").groupby("security_id").tail(1)
        px = px.rename(columns={"price_local": "price", "date": "price_date"})
        univ = ids.merge(px[["security_id", "price", "price_usd", "price_date"]], on="security_id", how="left")
        univ["price_fx_rate"] = univ["price_usd"] / univ["price"]
        univ["worldscope_fx_rate"] = self.fx_to_usd(univ["worldscope_currency"], univ["price_date"])
        univ["ibes_fx_rate"] = self.fx_to_usd(univ["ibes_currency"], univ["price_date"])

        ind = self.get_ws(["general_industry_classification"], run_date, max_age_days=None)
        ind["general_industry_classification"] = ind["general_industry_classification"].map(normalise_industry)
        univ = univ.merge(ind, on="worldscope_id", how="left")
        self.univ = univ
        self._prepare_caches(run_date)
        return univ

    def _prepare_caches(self, run_date):
        """Materialise the universe's slice of the big tables once (all later queries read these)."""
        run_date = _d(run_date)
        fin = self.spark.sql(f"""
            select f.* from {self.QL}.marketdata.financials_metrics f
            join (select distinct worldscope_id from gems_ids where worldscope_id is not null) u on u.worldscope_id = f.vendor_entity_id
            join (select distinct marketdata_metric_id from gems_ws_metric_ids) mi on mi.marketdata_metric_id = f.marketdata_metric_id
            where f.vendor_id = {self.v_ws} and coalesce(f.statement_date, f.datadate) >= date_sub('{run_date}', 7 * 366)
        """).cache()
        fin.createOrReplaceTempView("gems_fin")
        est = self.spark.sql(f"""
            select e.* from {self.QL}.marketdata.analyst_estimates_summary e
            join (select distinct ibes_id from gems_ids where ibes_id is not null) u on u.ibes_id = e.vendor_entity_code
            join (select distinct marketdata_metric_id from gems_ibes_metric_ids) mi on mi.marketdata_metric_id = e.marketdata_metric_id
            where e.as_of_date >= date_sub('{run_date}', 230) and e.as_of_date <= '{run_date}'
        """).cache()
        est.createOrReplaceTempView("gems_est")
        self._ws_cache = {}
        self._cache_date = run_date

    # ------------------------------------------------------------------ FX
    def _fx(self):
        if self._fx_cache is None:
            self._fx_cache = self.spark.sql(f"""
                select upper(fx_base_currency) base, upper(fx_foreign_currency) foreign, cast(fx_date as date) fx_date,
                       cast(fx_base_rate as double) rate
                from {self.SAGE}.market.vfx_rate_fact
                where fx_rate_source = 'WM11_SPOT_RATE'
                  and (upper(fx_foreign_currency) = 'USD' or upper(fx_base_currency) = 'USD')
                  and fx_date >= '2015-01-01'
            """).toPandas()
            f = self._fx_cache
            to_usd = f[f.foreign == "USD"][["base", "fx_date", "rate"]].rename(columns={"base": "ccy"})
            inv = f[(f.base == "USD") & (f.foreign != "USD")][["foreign", "fx_date", "rate"]].rename(columns={"foreign": "ccy"})
            inv["rate"] = 1.0 / inv["rate"]
            fx = pd.concat([to_usd.assign(p=1), inv.assign(p=2)]).sort_values(["ccy", "fx_date", "p"])
            fx = fx.drop_duplicates(["ccy", "fx_date"]).drop(columns="p")
            fx["fx_date"] = pd.to_datetime(fx["fx_date"]).astype("datetime64[ns]")
            self._fx_cache = fx.sort_values("fx_date")
        return self._fx_cache

    def fx_to_usd(self, ccy, dates):
        """Multiplier converting `ccy` into USD at each date (latest WM11 rate within FX_LOOKBACK_DAYS)."""
        q = pd.DataFrame({"ccy": pd.Series(ccy).astype("object").values,
                          "d": pd.to_datetime(pd.Series(dates).values).astype("datetime64[ns]"),
                          "i": range(len(ccy))})
        out = pd.Series(np.nan, index=q.i.values)
        out[q.ccy.values == "USD"] = 1.0
        fx = self._fx()
        qq = q[(q.ccy != "USD") & q.ccy.notna() & q.d.notna()].sort_values("d")
        if not qq.empty:
            m = pd.merge_asof(qq, fx, left_on="d", right_on="fx_date", left_by="ccy", right_by="ccy",
                              direction="backward", tolerance=pd.Timedelta(days=FX_LOOKBACK_DAYS))
            out[m.i.values] = m.rate.values
        return out.values

    # ------------------------------------------------------------------ prices
    def get_prices(self, start, end, ids=None):
        """All stored closes (Worldscope weekly + monthly, listing currency) with USD conversion."""
        ids = self.univ if ids is None else ids
        px = self.spark.sql(f"""
            select p.vendor_security_id as security_id, cast(p.datadate as date) as date, cast(p.bid_close as double) as price_local
            from {self.QL}.marketdata.security_prices p
            join gems_univ u on cast(p.vendor_security_id as string) = cast(u.security_id as string)
            where p.vendor_id = {self.v_px} and p.datadate between '{_d(start)}' and '{_d(end)}' and p.bid_close is not null
        """).toPandas()
        px["security_id"] = px["security_id"].astype(int)
        px["date"] = pd.to_datetime(px["date"])
        px = px.merge(ids[["security_id", "pricing_currency"]], on="security_id", how="left")
        px["price_usd"] = px["price_local"] * self.fx_to_usd(px["pricing_currency"], px["date"])
        return px[["security_id", "date", "price_local", "price_usd"]].sort_values(["security_id", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------ Worldscope
    def _fin_source(self):
        return "gems_fin" if getattr(self, "_ws_cache", None) is not None else f"{self.QL}.marketdata.financials_metrics"

    def get_ws(self, metrics, asof, annual=False, max_age_days=STALENESS_DAYS):
        asof = _d(asof)
        metrics = [m.lower() for m in metrics]
        key = (tuple(metrics), asof, annual, max_age_days)
        cache = getattr(self, "_ws_cache", None)
        if cache is not None and key in cache:
            return cache[key].copy()
        out = self._get_ws(metrics, asof, annual, max_age_days)
        if cache is not None:
            cache[key] = out
        return out.copy()

    def _get_ws(self, metrics, asof, annual, max_age_days):
        pit = f"and f.inserted_date < date_add('{asof}', 1)" if self.point_in_time else ""
        age = f"and period_end >= date_sub('{asof}', {int(max_age_days)})" if max_age_days else ""
        ann = "and lower(f.fiscal_period) = 'a'" if annual else ""
        sdf = self.spark.sql(f"""
            with raw as (
                select f.vendor_entity_id as worldscope_id, mi.name, mi.id_priority, lower(f.fiscal_period) fiscal_period,
                       coalesce(f.statement_date, case when lower(f.fiscal_period) = 'current' then f.datadate end) as period_end,
                       f.inserted_date, f.metric_value_num, f.metric_value_str, {FP_PRIORITY} as fp_priority
                from {self._fin_source()} f
                join gems_ws_metric_ids mi on mi.marketdata_metric_id = f.marketdata_metric_id
                join (select distinct worldscope_id from gems_ids where worldscope_id is not null) u
                     on u.worldscope_id = f.vendor_entity_id
                where f.vendor_id = {self.v_ws} and mi.name in ({_sqlist(metrics)}) {pit} {ann}
            ),
            fye_counts as (
                select worldscope_id, month(period_end) fye_month, count(*) n, max(period_end) last_pe
                from raw where fiscal_period = 'a' and period_end is not null
                group by worldscope_id, month(period_end)
            ),
            fye as (   -- usual fiscal-year-end month per company, from all its annual rows
                select worldscope_id, fye_month from fye_counts
                qualify row_number() over (partition by worldscope_id order by n desc, last_pe desc) = 1
            ),
            ok as (
                select r.* from raw r left join fye on fye.worldscope_id = r.worldscope_id
                where r.period_end is not null and r.period_end <= '{asof}' {age}
                  and not (r.fiscal_period = 'a' and fye.fye_month is not null and month(r.period_end) <> fye.fye_month)
            )
            select * from ok
        """)
        if annual:
            sdf = sdf.selectExpr("*", "max(period_end) over (partition by worldscope_id) as max_pe").where("period_end = max_pe")
            order = "inserted_date desc, id_priority"
        else:
            order = "period_end desc, fp_priority, inserted_date desc, id_priority"
        sdf.createOrReplaceTempView("gems_ws_tmp")
        best = self.spark.sql(f"""
            select worldscope_id, name, period_end, metric_value_num, metric_value_str
            from gems_ws_tmp
            qualify row_number() over (partition by worldscope_id, name order by {order}) = 1
        """).toPandas()
        if best.empty:
            out = pd.DataFrame(columns=["worldscope_id"] + metrics + (["period_end"] if annual else []))
            return out
        str_val = best["metric_value_str"].astype("object").where(best["metric_value_str"].notna(),
                                                                   best["metric_value_num"].astype("object"))
        best["value"] = np.where(best["name"].isin(STRING_FIELDS), str_val, best["metric_value_num"].astype("object"))
        out = best.pivot(index="worldscope_id", columns="name", values="value").reset_index()
        for m in metrics:
            if m not in out.columns:
                out[m] = np.nan
            elif m not in STRING_FIELDS:
                out[m] = pd.to_numeric(out[m], errors="coerce")
        if annual:
            pe = best.groupby("worldscope_id")["period_end"].max().rename("period_end").reset_index()
            pe["period_end"] = pd.to_datetime(pe["period_end"])
            out = out.merge(pe, on="worldscope_id", how="left")
        out.columns.name = None
        return out[["worldscope_id"] + metrics + (["period_end"] if annual else [])]

    # ------------------------------------------------------------------ I/B/E/S
    EST_COLS = ["number_of_current_estimates", "mean_current", "median_current", "standard_deviation_current",
                "mean_three_months_ago", "number_of_estimates_raised_over_four_weeks",
                "number_of_estimates_lowered_over_four_week"]

    def _est_source(self, asof):
        cached = getattr(self, "_ws_cache", None) is not None and 0 <= (self._cache_date - asof).days <= 200
        return "gems_est" if cached else f"{self.QL}.marketdata.analyst_estimates_summary"

    def get_estimates(self, measure, asof, periods=("FY1", "FY2", "FY3")):
        asof = _d(asof)
        cols = ", ".join(f"cast(e.{c} as double) {c}" for c in self.EST_COLS)
        pdf = self.spark.sql(f"""
            with e as (
                select e.vendor_entity_code as ibes_id, e.forecast_period, cast(e.fiscal_period_year as int) fiscal_period_year,
                       cast(e.fiscal_period_month as int) fiscal_period_month, cast(e.as_of_date as date) as_of_date, {cols},
                       e.inserted_date
                from {self._est_source(asof)} e
                join gems_ibes_metric_ids mi on mi.marketdata_metric_id = e.marketdata_metric_id and mi.measure = '{measure.lower()}'
                join (select distinct ibes_id from gems_ids where ibes_id is not null) u on u.ibes_id = e.vendor_entity_code
                where e.forecast_period in ({_sqlist(periods)})
                  and e.as_of_date <= '{asof}' and e.as_of_date >= date_sub('{asof}', {EST_LOOKBACK_DAYS})
            )
            select * from e
            qualify as_of_date = max(as_of_date) over (partition by ibes_id)
                and row_number() over (partition by ibes_id, forecast_period order by inserted_date desc) = 1
        """).toPandas()
        pdf["as_of_date"] = pd.to_datetime(pdf["as_of_date"])
        return pdf.drop(columns=["inserted_date"])

    def get_actuals(self, measure, asof):
        asof = _d(asof)
        pit = f"and a.as_of_date <= '{asof}'"
        pdf = self.spark.sql(f"""
            select a.vendor_entity_code as ibes_id, cast(a.fiscal_period_year as int) fiscal_period_year,
                   cast(a.fiscal_period_month as int) fiscal_period_month, cast(a.actual_value as double) actual_value
            from {self.QL}.marketdata.analyst_estimates_actuals_surprises a
            join gems_ibes_metric_ids mi on mi.marketdata_metric_id = a.marketdata_metric_id and mi.measure = '{measure.lower()}'
            join (select distinct ibes_id from gems_ids where ibes_id is not null) u on u.ibes_id = a.vendor_entity_code
            where a.forecast_period = 'FY0' and lower(a.fiscal_period) in ({_sqlist(self.annual_ibes_fiscal_periods)})
              and coalesce(a.restated, false) = false {pit}
            qualify row_number() over (partition by a.vendor_entity_code
                                       order by a.fiscal_period_year desc, a.fiscal_period_month desc, a.as_of_date desc, a.inserted_date desc) = 1
        """).toPandas()
        return pdf

    annual_ibes_fiscal_periods = ("a", "annual", "y", "fy")

    # ------------------------------------------------------------------ settings
    def get_rules(self):
        r = self.spark.sql(f"""
            select lower(d.barings_metric_name) barings_metric_name, r.rule_type, r.rule_value
            from {self.QL}.settings.metric_rules r
            join {self.QL}.settings.marketdata_metrics_definitions d on d.marketdata_metric_id = r.marketdata_metric_id
            where d.metric_type = 'computed' and coalesce(r.is_active, true)
              and r.rule_type in ('lower_limit', 'upper_limit', 'exclude_financials', 'invert_factors',
                                  'use_logs', 'valuation_multiples', 'use_std_dev_limit')
        """).toPandas()
        r["rule_value"] = pd.to_numeric(r["rule_value"], errors="coerce")
        w = r.pivot_table(index="barings_metric_name", columns="rule_type", values="rule_value", aggfunc="max").reset_index()
        for c in ["lower_limit", "upper_limit", "exclude_financials"]:
            if c not in w.columns:
                w[c] = np.nan
        w.columns.name = None
        return w

    def get_production_values(self, run_date, metric_names):
        """Stored production raw / clean values for reconciliation (reporting.qwd_position_level)."""
        return self.spark.sql(f"""
            select quantlake_security_id as security_id, metric_name,
                   avg(raw_value) prod_raw_value, avg(clean_value) prod_clean_value, avg(zscore_value) prod_zscore_value
            from {self.QL}.reporting.qwd_position_level
            where datadate = '{_d(run_date)}' and metric_name in ({_sqlist(metric_names)})
            group by quantlake_security_id, metric_name
        """).toPandas()

# COMMAND ----------

# MAGIC %md ## 2. Worldscope factors
# MAGIC Replicates `worldscope_raw_current_metrics`, `worldscope_raw_annuals_metrics`, `worldscope_raw_over_time_metrics`,
# MAGIC `worldscope_raw_over_time_ebit` and `worldscope_raw_operating_leverage` (`psa_crisp_metrics/investments/metrics.py` 538-4221).

# COMMAND ----------

"""Worldscope-based GEMS factors rebuilt in pure pandas.

Replicates (as coded, quirks included) these production classes from
psa_crisp_metrics/investments/metrics.py:
  worldscope_raw_current_metrics      (538-886)   -> compute_ws_current
  worldscope_raw_annuals_metrics      (888-1470)  -> compute_ws_annuals
  worldscope_raw_over_time_ebit       (1934-2540) -> compute_ws_over_time_ebit
  worldscope_raw_operating_leverage   (2542-2706) -> compute_operating_leverage
  worldscope_raw_over_time_metrics    (2708-4221) -> compute_ws_over_time

Data access is replaced by the coordinator's `get_ws` callable and `univ` frame (see INTERFACE.md):
  Financials.get(datadate=D)                    -> get_ws(metrics, D)
  Financials.get(datadate=D, fiscal_period='a') -> get_ws(metrics, D, annual=True) ('period_end' == datadate_annual)
  base_qwd.create_initial_datastore             -> univ (price, price_usd, price_fx_rate, price_date, worldscope_fx_rate)

Null semantics: the Spark-built classes (current, over_time, over_time_ebit, operating_leverage) use Spark
arithmetic where x/0 is NULL (non-ANSI) -> `_sdiv`. The annuals class computes in pandas (x/0 = +-inf, 0/0 = NaN)
-> plain pandas division.
"""

import calendar
from datetime import date, datetime
from functools import reduce

import numpy as np
import pandas as pd

try:  # production uses dateutil.relativedelta(years=n, months=m, day=31)
    from dateutil.relativedelta import relativedelta as _relativedelta
except ImportError:  # pragma: no cover
    _relativedelta = None


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _ws_to_date(d) -> date:
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    return d


def month_end_back(run_date, years: int = 0, months: int = 0) -> date:
    """Equivalent of `datadate - relativedelta(years=years, months=months, day=31)`:
    shift back by years/months, then clip to the last day of the resulting month."""
    d = _ws_to_date(run_date)
    if _relativedelta is not None:
        return d - _relativedelta(years=years, months=months, day=31)
    total = d.year * 12 + (d.month - 1) - years * 12 - months
    y, m = divmod(total, 12)
    m += 1
    return date(y, m, calendar.monthrange(y, m)[1])


def _sdiv(a, b):
    """Spark (non-ANSI) division: NULL when the divisor is 0 or NULL."""
    a = pd.to_numeric(a, errors="coerce").astype(float) if isinstance(a, pd.Series) else a
    b = pd.to_numeric(b, errors="coerce").astype(float) if isinstance(b, pd.Series) else b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.where(b != 0) if isinstance(out, pd.Series) else out


def _base(univ: pd.DataFrame) -> pd.DataFrame:
    """Per-security frame with a clean string worldscope key and float price/fx columns."""
    cols = ["security_id", "worldscope_id", "price", "price_usd", "price_fx_rate", "price_date",
            "worldscope_fx_rate"]
    df = univ[[c for c in cols if c in univ.columns]].copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df["worldscope_id"] = pd.Series(
        [None if pd.isna(v) else str(v) for v in df["worldscope_id"].tolist()], index=df.index, dtype=object)
    for c in ["price", "price_usd", "price_fx_rate", "worldscope_fx_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    return df.reset_index(drop=True)


def _ws(get_ws, metrics, asof, annual=False, suffix=""):
    """Call get_ws and return one row per worldscope_id with (suffixed) float metric columns."""
    raw = get_ws(list(metrics), _ws_to_date(asof), annual=annual)
    keep = ["worldscope_id", *metrics] + (["period_end"] if annual else [])
    df = raw.reindex(columns=keep).copy() if raw is not None else pd.DataFrame(columns=keep)
    df = df[df["worldscope_id"].notna()]
    df["worldscope_id"] = df["worldscope_id"].map(str).astype(object)
    df = df.drop_duplicates(subset=["worldscope_id"], keep="first")
    for m in metrics:
        if m == "general_industry_classification":
            df[m] = df[m].astype(object)
        else:
            df[m] = pd.to_numeric(df[m], errors="coerce").astype(float)
    if annual:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df = df.rename(columns={"period_end": "datadate_annual"})
    if suffix:
        df = df.rename(columns={c: c + suffix for c in df.columns if c != "worldscope_id"})
    return df


def _attach(base: pd.DataFrame, frames) -> pd.DataFrame:
    out = base
    for fr in frames:
        out = out.merge(fr, on="worldscope_id", how="left")
    return out


def _final_ebit(df: pd.DataFrame, suffix: str = "") -> pd.Series:
    """final_trailing_twelve_months_ebit = coalesce(TTM EBIT, TTM pretax income)."""
    e = df["trailing_twelve_months_ebit" + suffix]
    return e.where(e.notna(), df["trailing_twelve_months_pretax_income" + suffix])


def _growth_piecewise(c: pd.Series, p: pd.Series, n=None) -> pd.Series:
    """Production nested f.when chains (c = current, p = past).
    n=None -> eps_growth_1_year_hist variant (plain (c-p)/|p|, no CAGR, no sign tweak when both < 0).
    n=3/5  -> *_hist_cagr variant."""
    c = pd.to_numeric(c, errors="coerce").astype(float)
    p = pd.to_numeric(p, errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (c - p) / p.abs()
        if n is None:
            g = r
        else:
            g = (r + 1) ** (1.0 / n) - 1
            g_neg = -(((r.abs() + 1) ** (1.0 / n)) - 1)
    conds = [
        (p == 0) & (c == 0),
        (p == 0) & (c < 0),
        (p == 0) & (c > 0),
        (p > 0) & (c == 0),
        (p > 0) & (c > 0),
        (p > 0) & (c < 0),
        (p < 0) & (c == 0),
        (p < 0) & (c > 0),
    ]
    choices = [0.0, -1.0, 1.0, -1.0, g, -1.0, 1.0, g]
    if n is None:
        conds += [(p < 0) & (c < 0), p == c]
        choices += [g, 0.0]
    else:
        conds += [(p < 0) & (c < 0) & (c < p), (p < 0) & (c < 0) & (c > p), p == c]
        choices += [g_neg, g, 0.0]
    choices = [np.broadcast_to(np.asarray(x, dtype=float), c.shape) for x in choices]
    return pd.Series(np.select([x.to_numpy(dtype=bool) for x in conds], choices, default=np.nan), index=c.index)


def _neg_pop_std(cols) -> pd.Series:
    """-sqrt(sum((x_i - mean)^2) / N) with NULL if any input is NULL (Spark column arithmetic)."""
    mean = reduce(lambda a, b: a + b, cols) / len(cols)
    var = reduce(lambda a, b: a + b, [(x - mean) ** 2 for x in cols]) / len(cols)
    return -np.sqrt(var)


def _finalize_avg(df: pd.DataFrame, univ_base: pd.DataFrame, factors, run_date) -> pd.DataFrame:
    """Production tail of the Spark classes: keep rows with EntityId not null and
    coalesce(price_datadate, financials datadate) >= last month-end, select distinct
    (EntityId, DataDate, factors...), then avg per (EntityId, DataDate) (avg ignores NULLs).
    Per INTERFACE: averaged per worldscope_id and broadcast back to every security of that entity."""
    last_month_date = pd.Timestamp(month_end_back(run_date, months=1))
    # Deviation (data access): get_ws exposes no financial record date, so securities without a price
    # date cannot use production's coalesce fallback and are excluded from the entity average.
    keep = df["worldscope_id"].notna() & df["price_date"].notna() & (df["price_date"] >= last_month_date)
    sub = df.loc[keep, ["worldscope_id", "price_date", *factors]].copy()
    for c in factors:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").astype(float)
    sub = sub.drop_duplicates()
    agg = sub.groupby("worldscope_id", sort=False)[list(factors)].mean().reset_index()
    out = univ_base[["security_id", "worldscope_id"]].merge(agg, on="worldscope_id", how="left")
    return out[["security_id", *factors]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# 1. worldscope_raw_current_metrics
# --------------------------------------------------------------------------------------------------
CURRENT_FACTORS = [
    "asset_turnover", "book_yield", "capex_to_depreciation", "capital_adequacy_ratio_tier_1_capital",
    "capital_adequacy_ratio_total_capital", "cash_flow_conversion", "cash_flow_yield", "current_ratio",
    "debt_to_assets", "debt_to_equity", "dividend_growth_hist", "dividend_yield", "earnings_yield",
    "ebit_margin", "ebit_yield", "ebitda_to_ev", "ebitda_to_price", "interest_coverage", "loans_to_deposits",
    "market_capitalization", "net_income_margin", "operating_margin", "payout_ratio", "quick_ratio",
    "sales_growth_hist", "sales_to_ev", "sales_yield", "short_term_debt_to_total_assets",
    "short_term_debt_to_total_debt",
]

_CURRENT_METRICS = [
    "total_assets", "book_value_per_outstanding_share_fiscal_year_end", "capital_adequacy_ratio_tier_1_capital",
    "capital_adequacy_ratio_total_capital", "common_equity", "common_shares_outstanding",
    "common_shares_used_to_calculate_earnings_per_share", "current_assets_total", "current_liabilities_total",
    "current_ratio", "deposits_total", "dividends_per_share_1_yr_annual_growth", "enterprise_value",
    "market_capitalization", "net_sales_revenues_1_yr_annual_growth", "loans_net", "inventories_total",
    "short_term_debt_current_portion_of_long_term_debt", "total_asset_turnover", "total_debt",
    "total_shareholders_equity", "trailing_twelve_months_earnings_per_share", "trailing_twelve_months_net_profit",
    "trailing_twelve_months_dividends_per_share", "trailing_twelve_months_net_cash_flow_operating_activities",
    "trailing_twelve_months_depreciation_depletion_and_amortization", "trailing_twelve_months_capital_expenditures",
    "trailing_twelve_months_ebit", "trailing_twelve_months_ebitda", "trailing_twelve_months_pretax_income",
    "trailing_twelve_months_net_sales_revenue", "trailing_twelve_months_net_margin",
    "trailing_twelve_months_interest_expense", "trailing_12_months_operating_profit_margin",
]


def compute_ws_current(univ: pd.DataFrame, get_ws, run_date) -> pd.DataFrame:
    """Production: worldscope_raw_current_metrics, metrics.py lines 538-886 (29 outputs).

    Single Financials.get at run_date (non-annual). EBITDA falls back to pretax income + D&A, EBIT to pretax
    income. Quirk kept: `market_capitalization` is overwritten with price_usd * shares BEFORE sales_to_ev is
    computed, so sales_to_ev's denominator is EV*fx - (price_usd*shares)*fx + price_usd*shares, while
    ebitda_to_ev (computed earlier) uses the Worldscope market cap.
    """
    base = _base(univ)
    d = _attach(base, [_ws(get_ws, _CURRENT_METRICS, run_date)])
    fx, px = d["worldscope_fx_rate"], d["price_usd"]
    sh = d["common_shares_outstanding"]

    ebitda = d["trailing_twelve_months_ebitda"]
    final_ebitda = ebitda.where(
        ebitda.notna(),
        d["trailing_twelve_months_pretax_income"] + d["trailing_twelve_months_depreciation_depletion_and_amortization"],
    )
    final_ebit = _final_ebit(d)
    sales = d["trailing_twelve_months_net_sales_revenue"]
    eps = d["trailing_twelve_months_earnings_per_share"]
    dps = d["trailing_twelve_months_dividends_per_share"]
    cfo = d["trailing_twelve_months_net_cash_flow_operating_activities"]

    o = pd.DataFrame({"security_id": d["security_id"], "worldscope_id": d["worldscope_id"],
                      "price_date": d["price_date"]})
    o["sales_yield"] = _sdiv(sales * fx, px * sh)
    o["earnings_yield"] = _sdiv(eps * fx, px)
    o["book_yield"] = _sdiv(d["book_value_per_outstanding_share_fiscal_year_end"] * fx, px)
    o["dividend_yield"] = _sdiv(dps * fx, px)
    o["cash_flow_yield"] = _sdiv(_sdiv(cfo * fx, d["common_shares_used_to_calculate_earnings_per_share"]), px)
    ws_mcap = d["market_capitalization"]
    o["ebitda_to_ev"] = _sdiv(final_ebitda * fx, d["enterprise_value"] * fx - ws_mcap * fx + sh * px)
    o["debt_to_equity"] = _sdiv(d["total_debt"], d["total_shareholders_equity"])
    o["sales_growth_hist"] = d["net_sales_revenues_1_yr_annual_growth"] / 100
    mcap = px * sh  # overwrites the Worldscope market_capitalization column from here on (as coded)
    o["market_capitalization"] = mcap
    o["ebit_margin"] = _sdiv(final_ebit, sales)
    o["net_income_margin"] = d["trailing_twelve_months_net_margin"] / 100
    o["payout_ratio"] = _sdiv(dps, eps)
    o["sales_to_ev"] = _sdiv(sales * fx, d["enterprise_value"] * fx - mcap * fx + px * sh)
    o["asset_turnover"] = d["total_asset_turnover"]
    o["capex_to_depreciation"] = _sdiv(d["trailing_twelve_months_capital_expenditures"],
                                       d["trailing_twelve_months_depreciation_depletion_and_amortization"])
    o["cash_flow_conversion"] = _sdiv(cfo, d["common_equity"])
    o["current_ratio"] = d["current_ratio"]
    o["debt_to_assets"] = _sdiv(d["total_debt"], d["total_assets"])
    o["ebitda_to_price"] = _sdiv(final_ebitda * fx, px * sh)
    o["ebit_yield"] = _sdiv(final_ebit * fx, px * sh)
    o["dividend_growth_hist"] = d["dividends_per_share_1_yr_annual_growth"] / 100
    o["interest_coverage"] = _sdiv(final_ebit, d["trailing_twelve_months_interest_expense"])
    o["operating_margin"] = d["trailing_12_months_operating_profit_margin"] / 100
    o["quick_ratio"] = _sdiv(d["current_assets_total"] - d["inventories_total"], d["current_liabilities_total"])
    o["short_term_debt_to_total_assets"] = _sdiv(d["short_term_debt_current_portion_of_long_term_debt"],
                                                 d["total_assets"])
    o["short_term_debt_to_total_debt"] = _sdiv(d["short_term_debt_current_portion_of_long_term_debt"],
                                               d["total_debt"])
    o["capital_adequacy_ratio_tier_1_capital"] = d["capital_adequacy_ratio_tier_1_capital"] / 100
    o["capital_adequacy_ratio_total_capital"] = d["capital_adequacy_ratio_total_capital"] / 100
    o["loans_to_deposits"] = _sdiv(-d["loans_net"], d["deposits_total"])
    return _finalize_avg(o, base, CURRENT_FACTORS, run_date)


# --------------------------------------------------------------------------------------------------
# 2. worldscope_raw_annuals_metrics
# --------------------------------------------------------------------------------------------------
ANNUALS_FACTORS = [
    "cash_flow_growth_1_year_change", "cash_flow_growth_3_year_change", "free_cash_flow_yield",
    "free_cash_flow_conversion", "cash_flow_from_financing_to_total_assets",
    "extraordinary_credit_to_operating_income", "extraordinary_charge_to_operating_income",
    "unusual_items_to_operating_income", "operating_cash_flow_growth_1_year_change",
    "operating_cash_flow_growth_3_year_change", "shareholder_yield_growth_hist",
    "shareholder_yield_growth_3_year_hist", "net_income_to_risk_weighted_assets", "net_interest_income_to_net_loan",
    "interest_income_to_interest_expense", "non_interest_income_to_non_interest_expense",
    "loan_loss_provisions_to_net_loans", "non_performing_loans_to_equity_and_loan_loss_provision",
]

_ANN_CURRENT_ANNUAL = [
    "net_income_starting_line", "net_income_used_to_calculate_diluted_earnings_per_share", "net_cash_flow_financing",
    "extraordinary_credit_pretax", "extraordinary_charge_pretax", "net_cash_flow_operating_activities",
    "cash_dividends_paid_total", "net_proceeds_from_sale_issue_of_common_preferred",
    "common_preferred_redeemed_retired_converted_etc", "increase_decrease_in_short_term_borrowings",
    "long_term_borrowings", "reduction_in_long_term_debt", "discontinued_operations", "net_interest_income",
    "interest_income_total", "interest_expense_total", "non_interest_income", "non_interest_expense",
    "provision_for_loan_losses",
]
_ANN_CURRENT = [
    "total_assets", "trailing_twelve_months_depreciation_depletion_and_amortization",
    "trailing_twelve_months_capital_expenditures", "common_shares_outstanding", "common_equity",
    "trailing_12_months_operating_profit", "risk_weighted_assets", "loans_net", "non_performing_loans",
    "general_industry_classification",
]
_ANN_HIST_ANNUAL = [
    "net_income_starting_line", "net_cash_flow_operating_activities", "cash_dividends_paid_total",
    "net_proceeds_from_sale_issue_of_common_preferred", "common_preferred_redeemed_retired_converted_etc",
    "increase_decrease_in_short_term_borrowings", "long_term_borrowings", "reduction_in_long_term_debt",
]


def _shareholder_yield(d: pd.DataFrame, sfx: str, is_bank: pd.Series) -> pd.Series:
    """(div - (issue - redeem) - [non-bank: ST borrow chg + LT borrowings - LT debt reduction]) / |TA|,
    every flow fillna(0) (metrics.py 1286-1418)."""
    def z(c):
        return d[c + sfx].fillna(0)

    debt = np.where(is_bank, 0.0,
                    z("increase_decrease_in_short_term_borrowings") + z("long_term_borrowings")
                    - z("reduction_in_long_term_debt"))
    num = (z("cash_dividends_paid_total")
           - (z("net_proceeds_from_sale_issue_of_common_preferred")
              - z("common_preferred_redeemed_retired_converted_etc"))
           - debt)
    return num / d["total_assets" + sfx].abs()


def compute_ws_annuals(univ: pd.DataFrame, get_ws, run_date) -> pd.DataFrame:
    """Production: worldscope_raw_annuals_metrics, metrics.py lines 888-1470 (18 outputs).

    Pulls (as coded):
      annual (fiscal_period='a', 'period_end' = datadate_annual) at run_date: 19 flow items;
      non-annual at run_date: total_assets, TTM D&A, TTM capex, shares, common_equity, TTM operating profit,
        risk_weighted_assets, loans_net, non_performing_loans, general_industry_classification;
      annual at last_year_date / three_year_hist_date (= run_date - relativedelta(years=n, day=31)): 8 flow items;
      non-annual total_assets at the same dates.
    Staleness guards: 1-year changes NULL if datadate_annual == datadate_annual_last_year; 3-year changes NULL if
    datadate_annual - 700 days < datadate_annual_three_year_hist; missing datadate_annual and
    datadate_annual_three_year_hist are filled with 1950-01-01 (datadate_annual_last_year is not filled).
    Zero total_assets (all dates) and common_equity -> NaN. BANKs skip the debt term in shareholder yield.
    Computed in pandas (x/0 = inf). Production keeps ONE row per EntityId (drop_duplicates, first), not an
    average; the date filter is applied after that.
    """
    rd = _ws_to_date(run_date)
    last_month_date = pd.Timestamp(month_end_back(rd, months=1))
    last_year_date = month_end_back(rd, years=1)
    three_year_hist_date = month_end_back(rd, years=3)
    # The two_year_hist pulls in production (metrics.py 1059-1092) feed no output, so they are skipped.

    base = _base(univ)
    cur_a = _ws(get_ws, _ANN_CURRENT_ANNUAL, rd, annual=True)
    cur = _ws(get_ws, _ANN_CURRENT, rd)
    ly_a = _ws(get_ws, _ANN_HIST_ANNUAL, last_year_date, annual=True, suffix="_last_year")
    ly = _ws(get_ws, ["total_assets"], last_year_date, suffix="_last_year")
    h3_a = _ws(get_ws, _ANN_HIST_ANNUAL, three_year_hist_date, annual=True, suffix="_three_year_hist")
    h3 = _ws(get_ws, ["total_assets"], three_year_hist_date, suffix="_three_year_hist")

    # create_initial_datastore rows without a price date are dropped before the merge (metrics.py 932),
    # then drop_duplicates(subset=["EntityId"]) keeps the first row per entity (metrics.py 1161-1162).
    # Production row order is unspecified; we keep the first priced share line in univ order.
    priced = base[base["worldscope_id"].notna() & base["price_date"].notna()]
    d = _attach(priced, [cur, cur_a, ly, ly_a, h3, h3_a])
    d = d.drop_duplicates(subset=["worldscope_id"], keep="first").reset_index(drop=True)

    for c in ["total_assets", "total_assets_last_year", "total_assets_three_year_hist", "common_equity"]:
        d[c] = d[c].replace({0: np.nan})
    min_date = pd.Timestamp(1950, 1, 1)
    d["datadate_annual"] = d["datadate_annual"].fillna(min_date)
    d["datadate_annual_three_year_hist"] = d["datadate_annual_three_year_hist"].fillna(min_date)

    same_fy = (d["datadate_annual"] == d["datadate_annual_last_year"]).to_numpy(dtype=bool)
    too_close_3y = ((d["datadate_annual"] - pd.to_timedelta(700, unit="D"))
                    < d["datadate_annual_three_year_hist"]).to_numpy(dtype=bool)
    ta, ta_ly, ta_3 = d["total_assets"], d["total_assets_last_year"], d["total_assets_three_year_hist"]
    fx = d["worldscope_fx_rate"]

    with np.errstate(divide="ignore", invalid="ignore"):
        o = pd.DataFrame({"worldscope_id": d["worldscope_id"], "price_date": d["price_date"]})
        ni = d["net_income_starting_line"]
        o["cash_flow_growth_1_year_change"] = np.where(
            same_fy, np.nan, ni / ta - d["net_income_starting_line_last_year"] / ta_ly)
        o["cash_flow_growth_3_year_change"] = np.where(
            too_close_3y, np.nan, ni / ta - d["net_income_starting_line_three_year_hist"] / ta_3)

        ni_dil = d["net_income_used_to_calculate_diluted_earnings_per_share"]
        dda = d["trailing_twelve_months_depreciation_depletion_and_amortization"]
        capex = d["trailing_twelve_months_capital_expenditures"]
        o["free_cash_flow_yield"] = (ni_dil * fx + dda * fx - capex * fx) / (
            (d["price"] * d["price_fx_rate"]) * d["common_shares_outstanding"])
        o["free_cash_flow_conversion"] = (ni_dil + dda - capex) / d["common_equity"]
        o["cash_flow_from_financing_to_total_assets"] = d["net_cash_flow_financing"] / ((ta + ta_ly) / 2)

        op = d["trailing_12_months_operating_profit"].abs()
        o["extraordinary_credit_to_operating_income"] = d["extraordinary_credit_pretax"].abs() / op
        o["extraordinary_charge_to_operating_income"] = d["extraordinary_charge_pretax"].abs() / op
        o["unusual_items_to_operating_income"] = (
            d["extraordinary_charge_pretax"] - d["extraordinary_credit_pretax"]).abs() / op

        cfo = d["net_cash_flow_operating_activities"]
        o["operating_cash_flow_growth_1_year_change"] = np.where(
            same_fy, np.nan, cfo / ta - d["net_cash_flow_operating_activities_last_year"] / ta_ly)
        o["operating_cash_flow_growth_3_year_change"] = np.where(
            too_close_3y, np.nan, cfo / ta - d["net_cash_flow_operating_activities_three_year_hist"] / ta_3)

        is_bank = d["general_industry_classification"].isin(["BANK"]).to_numpy(dtype=bool)
        sy_cur = _shareholder_yield(d, "", is_bank)
        o["shareholder_yield_growth_hist"] = np.where(
            same_fy, np.nan, sy_cur - _shareholder_yield(d, "_last_year", is_bank))
        o["shareholder_yield_growth_3_year_hist"] = np.where(
            too_close_3y, np.nan, sy_cur - _shareholder_yield(d, "_three_year_hist", is_bank))

        o["net_income_to_risk_weighted_assets"] = (
            d["trailing_12_months_operating_profit"] - d["discontinued_operations"]) / d["risk_weighted_assets"]
        o["net_interest_income_to_net_loan"] = d["net_interest_income"] / d["loans_net"]
        o["interest_income_to_interest_expense"] = d["interest_income_total"] / d["interest_expense_total"]
        o["non_interest_income_to_non_interest_expense"] = d["non_interest_income"] / d["non_interest_expense"]
        o["loan_loss_provisions_to_net_loans"] = -d["provision_for_loan_losses"] / d["loans_net"]
        o["non_performing_loans_to_equity_and_loan_loss_provision"] = -d["non_performing_loans"] / (
            d["common_equity"] + d["provision_for_loan_losses"])

    o = o[o["price_date"] >= last_month_date]
    for c in ANNUALS_FACTORS:
        o[c] = pd.to_numeric(o[c], errors="coerce").astype(float)
    out = base[["security_id", "worldscope_id"]].merge(
        o[["worldscope_id", *ANNUALS_FACTORS]], on="worldscope_id", how="left")
    return out[["security_id", *ANNUALS_FACTORS]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# 3. worldscope_raw_over_time_metrics
# --------------------------------------------------------------------------------------------------
OVER_TIME_FACTORS = [
    "return_on_equity", "return_on_equity_5_year_vol", "return_on_invested_capital",
    "return_on_invested_capital_5_year_vol", "return_on_assets", "return_on_assets_1_year_change",
    "return_on_assets_5_year_vol", "accruals", "accruals_year_on_year_change", "sustainable_growth",
    "sustainable_growth_3_year_average", "eps_growth_1_year_hist", "eps_growth_3_year_hist_cagr",
    "eps_growth_5_year_hist_cagr", "earnings_yield_3_year_hist", "earnings_yield_5_year_hist",
    "net_income_margin_5_year_vol", "net_income_5_year_stability",
]

_NP, _EQ, _LTD, _TA = ("trailing_twelve_months_net_profit", "total_shareholders_equity", "long_term_debt",
                       "total_assets")
_CFO, _CAPEX = ("trailing_twelve_months_net_cash_flow_operating_activities",
                "trailing_twelve_months_capital_expenditures")
_DPS, _EPS, _NM = ("trailing_twelve_months_dividends_per_share", "trailing_twelve_months_earnings_per_share",
                   "trailing_twelve_months_net_margin")
_SFX = ["", "_last_year", "_two_year_hist", "_three_year_hist", "_four_year_hist", "_five_year_hist"]
_OT_PULLS = {
    "": [_NP, _EQ, _LTD, _TA, _CFO, _CAPEX, _DPS, _EPS, _NM],
    "_last_year": [_NP, _EQ, _LTD, _TA, _CFO, _CAPEX, _DPS, _EPS, _NM],
    "_two_year_hist": [_NP, _EQ, _LTD, _TA, _DPS, _EPS, _NM],
    "_three_year_hist": [_NP, _EQ, _LTD, _TA, _EPS, _NM],
    "_four_year_hist": [_NP, _EQ, _LTD, _TA, _NM],
    "_five_year_hist": [_NP, _EQ, _LTD, _TA, _EPS],
}


def compute_ws_over_time(univ: pd.DataFrame, get_ws, run_date) -> pd.DataFrame:
    """Production: worldscope_raw_over_time_metrics, metrics.py lines 2708-4221 (18 outputs).

    Six non-annual snapshots at run_date and run_date - relativedelta(years=n, day=31), n=1..5 (calendar
    spaced point-in-time TTM / balance-sheet values). ROE/ROIC/ROA use the average of the snapshot and the
    one a year earlier; *_5_year_vol = -population std of the 5 yearly ratios; accruals_year_on_year_change =
    -(acc / |acc_ly| - 1); eps growth piecewise sign rules as coded; net_income_5_year_stability =
    -sqrt(mean((NP_k - |mean NP|)^2)) / |mean NP| (deviation from |mean| kept as coded). Spark NULL semantics.
    """
    rd = _ws_to_date(run_date)
    base = _base(univ)
    frames = []
    for n, sfx in enumerate(_SFX):
        asof = rd if n == 0 else month_end_back(rd, years=n)
        frames.append(_ws(get_ws, _OT_PULLS[sfx], asof, suffix=sfx))
    d = _attach(base, frames)

    def col(name, sfx=""):
        return d[name + sfx]

    o = pd.DataFrame({"security_id": d["security_id"], "worldscope_id": d["worldscope_id"],
                      "price_date": d["price_date"]})

    roe = [_sdiv(col(_NP, _SFX[k]), (col(_EQ, _SFX[k]) + col(_EQ, _SFX[k + 1])) / 2) for k in range(5)]
    roic = [_sdiv(col(_NP, _SFX[k]), (col(_EQ, _SFX[k]) + col(_LTD, _SFX[k]) + col(_EQ, _SFX[k + 1])
                                      + col(_LTD, _SFX[k + 1])) / 2) for k in range(5)]
    roa = [_sdiv(col(_NP, _SFX[k]), (col(_TA, _SFX[k]) + col(_TA, _SFX[k + 1])) / 2) for k in range(5)]

    o["return_on_equity"] = roe[0]
    o["return_on_equity_5_year_vol"] = _neg_pop_std(roe)
    o["return_on_invested_capital"] = roic[0]
    o["return_on_invested_capital_5_year_vol"] = _neg_pop_std(roic)
    o["return_on_assets"] = roa[0]
    o["return_on_assets_5_year_vol"] = _neg_pop_std(roa)
    o["return_on_assets_1_year_change"] = roa[0] - roa[1]

    acc = _sdiv(col(_NP) - (col(_CFO) - col(_CAPEX)), (col(_TA) + col(_TA, "_last_year")) / 2)
    acc_ly = _sdiv(col(_NP, "_last_year") - (col(_CFO, "_last_year") - col(_CAPEX, "_last_year")),
                   (col(_TA, "_two_year_hist") + col(_TA, "_last_year")) / 2)
    o["accruals"] = acc
    o["accruals_year_on_year_change"] = -(_sdiv(acc, acc_ly.abs()) - 1)

    sg = [roe[k] * (1 - _sdiv(col(_DPS, _SFX[k]), col(_EPS, _SFX[k]))) for k in range(3)]
    o["sustainable_growth"] = sg[0]
    o["sustainable_growth_3_year_average"] = (sg[0] + sg[1] + sg[2]) / 3

    o["eps_growth_1_year_hist"] = _growth_piecewise(col(_EPS), col(_EPS, "_last_year"), None)
    o["eps_growth_3_year_hist_cagr"] = _growth_piecewise(col(_EPS), col(_EPS, "_three_year_hist"), 3)
    o["eps_growth_5_year_hist_cagr"] = _growth_piecewise(col(_EPS), col(_EPS, "_five_year_hist"), 5)

    fx, px = d["worldscope_fx_rate"], d["price_usd"]
    o["earnings_yield_3_year_hist"] = _sdiv(col(_EPS, "_three_year_hist") * fx, px)
    o["earnings_yield_5_year_hist"] = _sdiv(col(_EPS, "_five_year_hist") * fx, px)

    nim = [col(_NM, _SFX[k]) / 100 for k in range(5)]
    o["net_income_margin_5_year_vol"] = _neg_pop_std(nim)

    nps = [col(_NP, _SFX[k]) for k in range(5)]
    ni_mean = (reduce(lambda a, b: a + b, nps) / 5).abs()
    ni_vol = np.sqrt(reduce(lambda a, b: a + b, [(x - ni_mean) ** 2 for x in nps]) / 5)
    o["net_income_5_year_stability"] = _sdiv(-ni_vol, ni_mean)
    return _finalize_avg(o, base, OVER_TIME_FACTORS, run_date)


# --------------------------------------------------------------------------------------------------
# 4. worldscope_raw_over_time_ebit
# --------------------------------------------------------------------------------------------------
OVER_TIME_EBIT_FACTORS = ["ebit_growth_3_year_hist_cagr", "ebit_growth_5_year_hist_cagr",
                          "ebit_yield_3_year_hist", "ebit_yield_5_year_hist"]


def compute_ws_over_time_ebit(univ: pd.DataFrame, get_ws, run_date) -> pd.DataFrame:
    """Production: worldscope_raw_over_time_ebit, metrics.py lines 1934-2540 (4 outputs).

    final EBIT = coalesce(TTM EBIT, TTM pretax) at run_date and run_date - relativedelta(years=3|5, day=31).
    ebit_yield_N_year_hist = historical final EBIT * fx / (price_usd * CURRENT shares outstanding).
    ebit_growth_N_year_hist_cagr = piecewise CAGR (same sign rules as eps CAGR). Spark NULL semantics.
    """
    rd = _ws_to_date(run_date)
    base = _base(univ)
    pull = ["trailing_twelve_months_ebit", "trailing_twelve_months_pretax_income"]
    d = _attach(base, [
        _ws(get_ws, ["common_shares_outstanding", *pull], rd),
        _ws(get_ws, pull, month_end_back(rd, years=3), suffix="_three_year_hist"),
        _ws(get_ws, pull, month_end_back(rd, years=5), suffix="_five_year_hist"),
    ])
    e0, e3, e5 = _final_ebit(d), _final_ebit(d, "_three_year_hist"), _final_ebit(d, "_five_year_hist")
    mcap = d["price_usd"] * d["common_shares_outstanding"]
    o = pd.DataFrame({"security_id": d["security_id"], "worldscope_id": d["worldscope_id"],
                      "price_date": d["price_date"]})
    o["ebit_yield_3_year_hist"] = _sdiv(e3 * d["worldscope_fx_rate"], mcap)
    o["ebit_yield_5_year_hist"] = _sdiv(e5 * d["worldscope_fx_rate"], mcap)
    o["ebit_growth_3_year_hist_cagr"] = _growth_piecewise(e0, e3, 3)
    o["ebit_growth_5_year_hist_cagr"] = _growth_piecewise(e0, e5, 5)
    return _finalize_avg(o, base, OVER_TIME_EBIT_FACTORS, run_date)


# --------------------------------------------------------------------------------------------------
# 5. worldscope_raw_operating_leverage
# --------------------------------------------------------------------------------------------------
def compute_operating_leverage(univ: pd.DataFrame, get_ws, run_date) -> pd.DataFrame:
    """Production: worldscope_raw_operating_leverage, metrics.py lines 2542-2706 (1 output).

    OL(a,b) = (%chg final EBIT) / (%chg TTM sales) between snapshots (denominators |earlier|).
    operating_leverage = OL(current, last_year) if > 0 else OL(last_year, two_year_hist) if > 0 else NULL.
    Production applies NO price-date filter here (only EntityId and value not null) and takes distinct rows;
    the value has no price dependence, so it is identical across share lines and broadcast per worldscope_id.
    """
    rd = _ws_to_date(run_date)
    base = _base(univ)
    pull = ["trailing_twelve_months_ebit", "trailing_twelve_months_net_sales_revenue",
            "trailing_twelve_months_pretax_income"]
    d = _attach(base, [
        _ws(get_ws, pull, rd),
        _ws(get_ws, pull, month_end_back(rd, years=1), suffix="_last_year"),
        _ws(get_ws, pull, month_end_back(rd, years=2), suffix="_two_year_hist"),
    ])
    s = "trailing_twelve_months_net_sales_revenue"
    e0, e1, e2 = _final_ebit(d), _final_ebit(d, "_last_year"), _final_ebit(d, "_two_year_hist")
    s0, s1, s2 = d[s], d[s + "_last_year"], d[s + "_two_year_hist"]
    ol_01 = _sdiv(_sdiv(e0 - e1, e1.abs()), _sdiv(s0 - s1, s1.abs()))
    ol_12 = _sdiv(_sdiv(e1 - e2, e2.abs()), _sdiv(s1 - s2, s2.abs()))
    ol = ol_01.where(ol_01 > 0, ol_12.where(ol_12 > 0))
    o = pd.DataFrame({"worldscope_id": d["worldscope_id"], "operating_leverage": ol})
    o = o[o["worldscope_id"].notna() & o["operating_leverage"].notna()].drop_duplicates()
    agg = o.groupby("worldscope_id", sort=False)["operating_leverage"].mean().reset_index()
    out = base[["security_id", "worldscope_id"]].merge(agg, on="worldscope_id", how="left")
    return out[["security_id", "operating_leverage"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# all
# --------------------------------------------------------------------------------------------------
def compute_all_ws(univ: pd.DataFrame, get_ws, run_date) -> pd.DataFrame:
    """Merge of compute_ws_current, compute_ws_annuals, compute_ws_over_time, compute_ws_over_time_ebit and
    compute_operating_leverage on security_id (one row per univ security)."""
    parts = [
        compute_ws_current(univ, get_ws, run_date),
        compute_ws_annuals(univ, get_ws, run_date),
        compute_ws_over_time(univ, get_ws, run_date),
        compute_ws_over_time_ebit(univ, get_ws, run_date),
        compute_operating_leverage(univ, get_ws, run_date),
    ]
    return reduce(lambda a, b: a.merge(b, on="security_id", how="left"), parts)

# COMMAND ----------

# MAGIC %md ## 3. Price factors
# MAGIC Replicates `worldscope_raw_price_metrics` (metrics.py 1472-1918), excluding the `beta` placeholder.

# COMMAND ----------

"""Price-based GEMS factors rebuilt from Quantlake prices.

Replicates ``worldscope_raw_price_metrics.compute`` (psa_crisp_metrics/investments/metrics.py
lines 1472-1918) AS CODED, using only the coordinator loader ``get_prices``.

Not built here: ``beta`` (production writes a dummy = price, lines 1730-1731; excluded by INTERFACE).
"""
from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

RETURN_HORIZONS = (1, 3, 6, 9, 12)

PRICE_FACTORS = [
    "1_mth_price_return_local",
    "1_mth_price_return_usd",
    "3_mth_price_return_local",
    "3_mth_price_return_usd",
    "6_mth_price_return_local",
    "6_mth_price_return_usd",
    "9_mth_price_return_local",
    "9_mth_price_return_usd",
    "12_mth_price_return_local",
    "12_mth_price_return_usd",
    "regression_momentum_1_year",
    "1_year_volatility",
    "price_return_stability",
]


def _px_to_date(d):
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime):
        return d.date()
    return d


def _px_str_ids(s):
    return pd.Series([None if pd.isna(v) else str(v) for v in s.astype(object)], index=s.index, dtype=object)


def _clean_prices(df):
    df = df[["security_id", "date", "price_local", "price_usd"]].copy()
    df["security_id"] = df["security_id"].astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    df["price_local"] = pd.to_numeric(df["price_local"], errors="coerce").astype(float)
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce").astype(float)
    return df


def resample_monthly(prices):
    """Emulate DataAccess.get_security_prices(timeseries_frequency=MONTHLY).

    Deviation (data access): get_prices returns weekly+monthly observations mixed, so the monthly
    series is taken as the LAST stored observation of each calendar month (actual observation date kept).
    """
    df = _clean_prices(prices).sort_values(["security_id", "date"], kind="mergesort")
    per = df["date"].dt.year * 100 + df["date"].dt.month
    df = df.assign(_per=per.values)
    out = df.groupby(["security_id", "_per"], sort=False).tail(1)
    return out.drop(columns="_per").reset_index(drop=True)


def resample_weekly(prices):
    """Emulate DataAccess.get_security_prices(timeseries_frequency=WEEKLY).

    Deviation (data access): LAST stored observation of each ISO week (ISO year, ISO week number).
    """
    df = _clean_prices(prices).sort_values(["security_id", "date"], kind="mergesort")
    iso = df["date"].dt.isocalendar()
    per = iso["year"].astype("int64") * 100 + iso["week"].astype("int64")
    df = df.assign(_per=per.values)
    out = df.groupby(["security_id", "_per"], sort=False).tail(1)
    return out.drop(columns="_per").reset_index(drop=True)


def _pct_change_pad(df, col):
    # pandas<3 groupby pct_change default fill_method='pad' (production): group-wise ffill, then ratio.
    filled = df.groupby("security_id", sort=False)[col].ffill()
    prev = filled.groupby(df["security_id"], sort=False).shift(1)
    with np.errstate(all="ignore"):
        return filled / prev - 1


def _linregress_t(x, y, key):
    """Vectorised numpy re-implementation of scipy.stats.linregress (slope / slope stderr) per key.

    Mirrors scipy: ssxm, ssxym, ssym from np.cov(bias=1); r = ssxym/sqrt(ssxm*ssym) (0 if the
    denominator is 0, clipped to [-1,1]); slope = ssxym/ssxm; stderr = 0 if n == 2 else
    sqrt((1-r^2)*ssym/ssxm/(n-2)). Returns slope/stderr (numpy float semantics: x/0 -> +-inf/nan).
    """
    d = pd.DataFrame({"k": key.values, "x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    g = d.groupby("k", sort=True)
    n = g["x"].transform("count").astype(float)
    with np.errstate(all="ignore"):
        dx = d["x"] - g["x"].transform("mean")
        dy = d["y"] - g["y"].transform("mean")
        agg = pd.DataFrame({"k": d["k"], "xx": dx * dx, "yy": dy * dy, "xy": dx * dy, "n": n})
        a = agg.groupby("k", sort=True).agg(xx=("xx", "mean"), yy=("yy", "mean"), xy=("xy", "mean"), n=("n", "first"))
        ssxm, ssym, ssxym, nn = (a[c].to_numpy(dtype=float) for c in ("xx", "yy", "xy", "n"))
        r_den = np.sqrt(ssxm * ssym)
        r = np.where(r_den == 0.0, 0.0, ssxym / np.where(r_den == 0.0, 1.0, r_den))
        r = np.clip(r, -1.0, 1.0)
        slope = ssxym / ssxm
        stderr = np.where(nn == 2, 0.0, np.sqrt((1 - r**2) * ssym / ssxm / (nn - 2)))
        t = slope / stderr
    t = np.where(nn < 2, np.nan, t)
    return pd.Series(t, index=a.index)


def _weekly_stats(weekly):
    """regression_momentum_1_year and 1_year_volatility per security (metrics.py 1733-1806).

    calculate_t_stat (1738-1751) quirk kept: X = arange(len(group)); y = log(close_bid).dropna();
    linregress(X[:len(y)], y) -> when NaN prices are dropped, X is truncated (0..len(y)-1), not aligned.
    1_year_volatility: expanding().std() read on the max-date row == sample std (ddof=1) of all weekly
    USD returns, times sqrt(row_count) where row_count counts ALL weekly rows (incl. first, NaN-return row).
    """
    w = weekly.sort_values(["security_id", "date"], kind="mergesort").reset_index(drop=True)
    sids = pd.Index(np.sort(w["security_id"].unique()), name="security_id")
    with np.errstate(all="ignore"):
        logp = np.log(w["price_usd"])
    valid = logp.notna()
    wy = w.loc[valid, ["security_id"]].assign(y=logp[valid].values)
    wy["x"] = wy.groupby("security_id", sort=False).cumcount().astype(float)
    t = _linregress_t(wy["x"], wy["y"], wy["security_id"]) / 100
    reg = t.reindex(sids)
    chg = _pct_change_pad(w, "price_usd")
    sd = chg.groupby(w["security_id"], sort=True).std(ddof=1).reindex(sids)
    cnt = w.groupby("security_id", sort=True).size().reindex(sids).astype(float)
    vol = sd * np.sqrt(cnt)
    return pd.DataFrame(
        {"security_id": sids.values, "regression_momentum_1_year": reg.values, "1_year_volatility": vol.values}
    )


def _monthly_stability(monthly):
    """price_return_stability per security (metrics.py 1830-1869): -std(ddof=1) of monthly USD % changes x100
    (expanding std read on the max-date row == full-sample std)."""
    m = monthly.sort_values(["security_id", "date"], kind="mergesort").reset_index(drop=True)
    chg = _pct_change_pad(m, "price_usd") * 100
    sd = chg.groupby(m["security_id"], sort=True).std(ddof=1)
    return pd.DataFrame({"security_id": sd.index.values, "price_return_stability": -sd.values})


def _returns(monthly, run_date):
    """1/3/6/9/12 month local & USD returns (metrics.py 1544-1729)."""
    run_ts = pd.Timestamp(run_date)
    last_12_month_date = pd.Timestamp(run_date - relativedelta(months=12, day=31))
    df_prices = monthly[
        (monthly["date"] >= last_12_month_date - pd.Timedelta(days=30)) & (monthly["date"] <= run_ts)
    ]
    latest = df_prices[
        (df_prices["date"] >= run_ts - pd.Timedelta(days=30)) & (df_prices["date"] <= run_ts)
    ]
    latest = latest.sort_values(["security_id", "date"], kind="mergesort").groupby("security_id").tail(1)
    out = latest.rename(columns={"price_local": "price", "date": "datadate"})[
        ["security_id", "datadate", "price", "price_usd"]
    ].reset_index(drop=True)
    for m in RETURN_HORIZONS:
        ref = out["datadate"].apply(lambda d, m=m: pd.Timestamp(d.date() - relativedelta(months=m, day=31)))
        refs = pd.DataFrame({"security_id": out["security_id"], "_ref": ref.values})
        j = df_prices.merge(refs, on="security_id", how="inner")
        j = j[(j["date"] >= j["_ref"] - pd.Timedelta(days=30)) & (j["date"] <= j["_ref"])]
        j = j.sort_values(["security_id", "date"], kind="mergesort").groupby("security_id").tail(1)
        j = j[["security_id", "price_local", "price_usd"]].rename(
            columns={"price_local": f"_p_{m}", "price_usd": f"_pu_{m}"}
        )
        out = out.merge(j, on="security_id", how="left")
        with np.errstate(all="ignore"):
            out[f"{m}_mth_price_return_local"] = out["price"] / out[f"_p_{m}"] - 1
            out[f"{m}_mth_price_return_usd"] = out["price_usd"] / out[f"_pu_{m}"] - 1
    return out


def _pick_first(frame, univ_map, cols):
    """Production merges per-security rows onto Worldscope EntityId and keeps the FIRST row
    (drop_duplicates(subset=['EntityId']), metrics.py 1885-1887). Proxy for 'first': lowest security_id."""
    f = frame.merge(univ_map, on="security_id", how="inner")
    f = f[f["worldscope_id"].notna()]
    f = f.sort_values("security_id", kind="mergesort").drop_duplicates("worldscope_id", keep="first")
    return f[["worldscope_id", *cols]]


def compute_price_metrics(univ, get_prices, run_date, entity_agg="first"):
    """Price momentum / risk factors. Production: worldscope_raw_price_metrics (metrics.py 1472-1918).

    Sampling (as coded, all via dateutil relativedelta):
      three_year_date = run_date - relativedelta(years=3, day=31); monthly USD series [three_year_date, run_date]
      one_year_date   = run_date - relativedelta(years=1, day=31); weekly  USD series [one_year_date, run_date]
      Returns: monthly obs restricted to [last_12_month_date - 30d, run_date] (last_12_month_date =
        run_date - relativedelta(months=12, day=31)); latest = last obs in [run_date-30d, run_date];
        reference for horizon m = latest_date - relativedelta(months=m, day=31) (a month end); historical
        price = last obs in [ref-30d, ref]. return = latest/hist - 1 (local and USD).
      regression_momentum_1_year: t-stat of OLS slope of log(weekly USD price) on 0..n-1, /100.
      1_year_volatility: sample std of weekly USD returns x sqrt(number of weekly rows).
      price_return_stability: -(sample std of monthly USD % returns) over the 3y window.
    Entity: computed per security, then one value per worldscope_id taken from the lowest security_id
      ('first' row, as production drop_duplicates) and broadcast to all securities sharing worldscope_id.
      entity_agg='mean' instead averages share lines (INTERFACE default convention; not production).
    Post-filter (metrics.py 1889-1891): entity row kept only if its latest price date >= last_month_date
      (= run_date - relativedelta(months=1, day=31)); otherwise ALL factors are NaN for that entity.
    Deviation (data access): get_prices mixes weekly+monthly closes; monthly = last obs per calendar month,
      weekly = last obs per ISO week (see resample_monthly / resample_weekly).
    """
    run_date = _px_to_date(run_date)
    last_month_date = pd.Timestamp(run_date - relativedelta(months=1, day=31))
    one_year_date = run_date - relativedelta(years=1, day=31)
    three_year_date = run_date - relativedelta(years=3, day=31)

    raw = get_prices(three_year_date, run_date)
    raw = _clean_prices(raw)
    raw = raw[(raw["date"] >= pd.Timestamp(three_year_date)) & (raw["date"] <= pd.Timestamp(run_date))]
    monthly = resample_monthly(raw)
    weekly = resample_weekly(raw[raw["date"] >= pd.Timestamp(one_year_date)])

    rets = _returns(monthly, run_date)
    wk = _weekly_stats(weekly)
    st = _monthly_stability(monthly)

    univ_map = univ[["security_id", "worldscope_id"]].copy()
    univ_map["security_id"] = univ_map["security_id"].astype("int64")
    univ_map["worldscope_id"] = _px_str_ids(univ_map["worldscope_id"])

    ret_cols = [f"{m}_mth_price_return_{c}" for m in RETURN_HORIZONS for c in ("local", "usd")]
    wk_cols = ["regression_momentum_1_year", "1_year_volatility"]

    if entity_agg == "mean":
        r = rets.merge(univ_map, on="security_id", how="inner")
        r = r[r["worldscope_id"].notna() & (r["datadate"] >= last_month_date)]
        ent = r.groupby("worldscope_id")[ret_cols].mean().reset_index()
        for frame, cols in ((wk, wk_cols), (st, ["price_return_stability"])):
            x = frame.merge(univ_map, on="security_id", how="inner")
            x = x[x["worldscope_id"].isin(ent["worldscope_id"])]
            ent = ent.merge(x.groupby("worldscope_id")[cols].mean().reset_index(), on="worldscope_id", how="left")
    else:
        ent = _pick_first(rets, univ_map, ["datadate", *ret_cols])
        ent = ent.merge(_pick_first(wk, univ_map, wk_cols), on="worldscope_id", how="left")
        ent = ent.merge(_pick_first(st, univ_map, ["price_return_stability"]), on="worldscope_id", how="left")
        ent = ent[ent["datadate"].notna() & (ent["datadate"] >= last_month_date)].drop(columns="datadate")

    out = univ_map.merge(ent, on="worldscope_id", how="left")
    out = out[["security_id", *PRICE_FACTORS]].copy()
    for c in PRICE_FACTORS:
        out[c] = out[c].astype(float)
    return out.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md ## 4. I/B/E/S factors
# MAGIC Replicates `base_ntm` / `base_ltm` and the `ibes_raw_*` classes (metrics.py 406-536, 4315-6857), excluding the `eps_volatility` and `sales_volatility` placeholders.

# COMMAND ----------

"""I/B/E/S consensus-based GEMS factors.

Replicates (AS CODED) the production PSA_Crisp_Metrics classes:
  base_ntm / base_ltm            psa_crisp_metrics/base_classes/base_qwd.py 30-276 (+ metrics.py 406-536)
  base_qwd.create_initial_datastore  base_qwd.py 281-412 (entity row / price / FX selection)
  ibes_raw_eps_metrics           metrics.py 4315-5763
  ibes_raw_ebit_metrics          metrics.py 5765-6434
  ibes_raw_revenue_metrics       metrics.py 6436-6705
  ibes_raw_dividends_metrics     metrics.py 6707-6800
  ibes_raw_ntm_change_metrics    metrics.py 6802-6857
Only the coordinator loaders (get_estimates / get_actuals / get_ws) are used. Pure pandas + numpy.

Spark semantics emulated:
  * gate `f.when(n < 2, None).otherwise(x)`: NULL only when n is known and < 2 (NULL n keeps x).
  * `f.greatest(a, b)` ignores NULLs (np.fmax).
  * division by zero yields NULL (non-ANSI Spark), never +-inf.
  * pow(negative, 0.5) yields NaN.
Placeholders NOT built: eps_volatility, sales_volatility (dummy copies of eps_ntm / sales_ntm in production).
"""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

HIGH_UNIT_CURRENCIES = ("CLP", "COP", "IDR", "JPY", "KRW", "VEB")

MEASURE_NAMES = {"eps": "eps", "dps": "dps", "sal": "sales", "cps": "cfps", "ebi": "ebit", "roe": "roe"}

EPS_FACTORS = [
    "earnings_revisions_1_mth",
    "earnings_revisions_4_mth",
    "earnings_revision_persistence_6mth",
    "eps_growth_fy0_to_fy1",
    "eps_growth_fy1_to_fy2",
    "eps_growth_fy1_to_fy3",
    "eps_growth_fy2_to_fy3",
    "eps_growth_ltm_to_ntm",
    "eps_uncertainty",
    "standardised_unexpected_earnings",
    "earnings_yield_ntm",
    "earnings_yield_ltm",
    "earnings_yield_fy1",
    "earnings_yield_fy2",
    "earnings_yield_fy3",
]
EBIT_FACTORS = [
    "ebit_yield_fy1",
    "ebit_yield_fy2",
    "ebit_yield_fy3",
    "ebit_yield_ntm",
    "ebit_yield_ltm",
    "cash_flow_growth_ltm_to_ntm",
    "ebit_growth_ltm_to_ntm",
    "ebit_growth_fy1_to_fy2",
    "ebit_growth_fy2_to_fy3",
    "ebit_growth_fy1_to_fy3",
]
REVENUE_FACTORS = [
    "sales_yield_ntm",
    "sales_growth_ltm_to_ntm",
    "sales_growth_fy1_to_fy3",
    "sales_revisions_1_mth",
    "return_on_equity_ntm",
    "sustainable_growth_ntm",
]
DIVIDEND_FACTORS = ["dividend_yield_ntm"]
NTM_CHANGE_FACTORS = ["eps_ntm_1mth_chg", "eps_ntm_3mth_chg"]
ALL_IBES_FACTORS = EPS_FACTORS + EBIT_FACTORS + REVENUE_FACTORS + DIVIDEND_FACTORS + NTM_CHANGE_FACTORS

_EST_COLS = [
    "number_of_current_estimates",
    "mean_current",
    "median_current",
    "standard_deviation_current",
    "mean_three_months_ago",
    "number_of_estimates_raised_over_four_weeks",
    "number_of_estimates_lowered_over_four_week",
]
_UP = "number_of_estimates_raised_over_four_weeks"
_DN = "number_of_estimates_lowered_over_four_week"
_N = "number_of_current_estimates"


# ----------------------------------------------------------------------------- helpers
def _ib_to_date(d):
    if isinstance(d, (pd.Timestamp, datetime)):
        return d.date()
    return d


def previous_or_current_monday(given_date):
    """psa_common_functions/utils.py:87-88."""
    given_date = _ib_to_date(given_date)
    return given_date - timedelta(days=given_date.weekday())


def _ib_str_ids(s):
    return pd.Series([None if pd.isna(v) else str(v) for v in s.astype(object)], index=s.index, dtype=object)


def _ib_num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def _div(a, b):
    """Spark (non-ANSI) division: NULL when divisor is 0 or NULL."""
    a = pd.Series(a, dtype=float) if not isinstance(a, pd.Series) else a.astype(float)
    b = pd.Series(b, dtype=float, index=a.index) if not isinstance(b, pd.Series) else b.astype(float)
    with np.errstate(all="ignore"):
        out = a / b.where(b != 0)
    return out


def _gate(value, n):
    """f.when(n < 2, None).otherwise(value): NULL n -> comparison NULL -> keeps value."""
    n = n.astype(float)
    return value.where(~(n < 2))


def _gate2(value, n_a, n_b):
    """f.when(f.greatest(n_a, n_b) < 2, None).otherwise(value); greatest ignores NULLs."""
    with np.errstate(all="ignore"):
        g = pd.Series(np.fmax(n_a.astype(float).to_numpy(), n_b.astype(float).to_numpy()), index=value.index)
    return value.where(~(g < 2))


def _clip1(v):
    """when(v > 1, 1).otherwise(when(v < -1, -1).otherwise(v))."""
    return v.where(~(v > 1), 1.0).where(~(v < -1), -1.0)


def _sqrt(v):
    with np.errstate(all="ignore"):
        return pd.Series(np.power(v.to_numpy(dtype=float), 0.5), index=v.index)


def _growth(a, b, compound=False):
    """Sign-case growth table used for eps/ebit growth (e.g. metrics.py 4529-4611, 4704-4870).

    a = base, b = new. Branch order exactly as the nested f.when chain; any NULL input -> NULL.
    compound=True is the FY1->FY3 two-year version: sqrt((b-a)/|a| + 1) - 1, with the
    both-negative/deteriorating case -(sqrt(|(b-a)/|a|| + 1) - 1).
    """
    a = a.astype(float)
    b = b.astype(float)
    with np.errstate(all="ignore"):
        rel = (b - a) / a.abs()
        if compound:
            grow = _sqrt(rel + 1) - 1
            neg_det = -(_sqrt(rel.abs() + 1) - 1)
        else:
            grow = rel
            neg_det = rel
    conds = [
        (a == 0) & (b == 0),
        (a == 0) & (b < 0),
        (a == 0) & (b > 0),
        (a > 0) & (b == 0),
        (a > 0) & (b > 0),
        (a > 0) & (b < 0),
        (a < 0) & (b == 0),
        (a < 0) & (b > 0),
    ]
    choices = [0.0, -1.0, 1.0, -1.0, grow, -1.0, 1.0, grow]
    if compound:
        conds += [(a < 0) & (b < 0) & (b < a), (a < 0) & (b < 0) & (b > a)]
        choices += [neg_det, grow]
    else:
        conds += [(a < 0) & (b < 0)]
        choices += [rel]
    conds += [a == b]
    choices += [0.0]
    conds = [c.to_numpy(dtype=bool) for c in conds]
    choices = [np.broadcast_to(np.asarray(c, dtype=float), a.shape) for c in choices]
    return pd.Series(np.select(conds, choices, default=np.nan), index=a.index)


def _period_end(year, month, current_date):
    """last_day(if(fiscal_period_year = 0, current_date(), to_date(year-month-01)))."""
    year = _ib_num(year)
    month = _ib_num(month)
    cur = pd.Timestamp(current_date) + pd.offsets.MonthEnd(0)
    ok = year.notna() & month.notna() & (year > 0) & (month >= 1) & (month <= 12)
    out = pd.Series(pd.NaT, index=year.index, dtype="datetime64[ns]")
    if ok.any():
        d = pd.to_datetime(
            pd.DataFrame({"year": year[ok].astype("int64"), "month": month[ok].astype("int64"), "day": 1})
        ) + pd.offsets.MonthEnd(0)
        out.loc[ok] = d.astype("datetime64[ns]").values
    out.loc[year == 0] = cur
    return out


def _days(a, b):
    """(a - b).cast('int') on Spark dates -> whole days."""
    return (pd.to_datetime(a) - pd.to_datetime(b)).dt.days.astype(float)


def _prep_est(df):
    df = df.copy()
    df["ibes_id"] = _ib_str_ids(df["ibes_id"])
    df["forecast_period"] = df["forecast_period"].astype(str)
    for c in _EST_COLS:
        if c in df.columns:
            df[c] = _ib_num(df[c])
    return df[df["ibes_id"].notna()]


def _wide(est, periods, cols=None, suffix_fmt="{c}_{p}"):
    """Split long estimates by forecast_period and suffix columns (production renames *_FY1 etc.)."""
    cols = cols or _EST_COLS
    out = None
    for p in periods:
        sub = est[est["forecast_period"] == p]
        keep = [c for c in cols if c in sub.columns]
        # crash guard: production assumes one row per entity & period
        sub = sub[["ibes_id", *keep]].drop_duplicates("ibes_id", keep="first")
        sub = sub.rename(columns={c: suffix_fmt.format(c=c, p=p) for c in keep})
        for c in cols:
            name = suffix_fmt.format(c=c, p=p)
            if name not in sub.columns:
                sub[name] = np.nan
        out = sub if out is None else out.merge(sub, on="ibes_id", how="outer")
    return out


def _empty_ids(name):
    return pd.DataFrame({"ibes_id": pd.Series([], dtype=object), name: pd.Series([], dtype=float)})


# ----------------------------------------------------------------------------- NTM / LTM
def ntm(measure, asof, get_estimates, current_date=None):
    """base_ntm (base_qwd.py 30-149) for one estimate measure ('eps','dps','sal','cps','ebi','roe').

    as_of = Monday on/before asof; FY1 & FY2 consensus mean_current (inner join on ibes);
    period end = last day of fiscal_period_year/month (year 0 -> current_date month end);
    TWD = Monday + 1 year; D = FY2_end - FY1_end (days); NTM = (FY2_end-TWD)/D*FY1 + (TWD-FY1_end)/D*FY2
    (weights not clipped; D == 0 -> NULL).
    Deviation: Spark current_date() (wall clock) replaced by current_date (defaults to asof).
    Returns DataFrame [ibes_id, '<name>_ntm'].
    """
    name = f"{MEASURE_NAMES.get(measure, measure)}_ntm"
    monday = previous_or_current_monday(asof)
    current_date = _ib_to_date(current_date) if current_date is not None else _ib_to_date(asof)
    est = get_estimates(measure, monday, periods=("FY1", "FY2"))
    if est is None or len(est) == 0:
        return _empty_ids(name)
    est = _prep_est(est)
    est["_end"] = _period_end(est["fiscal_period_year"], est["fiscal_period_month"], current_date)
    f1 = est[est["forecast_period"] == "FY1"][["ibes_id", "mean_current", "_end"]].rename(
        columns={"mean_current": "m1", "_end": "e1"}
    )
    f2 = est[est["forecast_period"] == "FY2"][["ibes_id", "mean_current", "_end"]].rename(
        columns={"mean_current": "m2", "_end": "e2"}
    )
    df = f1.merge(f2, on="ibes_id", how="inner")
    twd = pd.Timestamp(monday) + pd.DateOffset(years=1)
    d = _days(df["e2"], df["e1"])
    r1 = _div(_days(df["e2"], pd.Series(twd, index=df.index)), d)
    r2 = _div(_days(pd.Series(twd, index=df.index), df["e1"]), d)
    df[name] = r1 * df["m1"] + r2 * df["m2"]
    return df[["ibes_id", name]].drop_duplicates("ibes_id", keep="first").reset_index(drop=True)


def ltm(measure, asof, get_estimates, get_actuals, current_date=None):
    """base_ltm (base_qwd.py 152-275).

    as_of = Monday on/before asof; FY0 = reported actual_value (get_actuals) and its fiscal period end,
    FY1 = consensus mean_current; TWD = Monday (no offset); D = FY1_end - FY0_end;
    LTM = (FY1_end-TWD)/D*FY0_actual + (TWD-FY0_end)/D*FY1_mean (weights not clipped; D == 0 -> NULL).
    Deviation (data access): FY0 'actual_value' comes from get_actuals instead of the FY0 estimates row.
    Returns DataFrame [ibes_id, '<name>_ltm'].
    """
    name = f"{MEASURE_NAMES.get(measure, measure)}_ltm"
    monday = previous_or_current_monday(asof)
    current_date = _ib_to_date(current_date) if current_date is not None else _ib_to_date(asof)
    est = get_estimates(measure, monday, periods=("FY1",))
    act = get_actuals(measure, monday)
    if est is None or len(est) == 0 or act is None or len(act) == 0:
        return _empty_ids(name)
    est = _prep_est(est)
    est = est[est["forecast_period"] == "FY1"].copy()
    est["e1"] = _period_end(est["fiscal_period_year"], est["fiscal_period_month"], current_date)
    act = act.copy()
    act["ibes_id"] = _ib_str_ids(act["ibes_id"])
    act["a0"] = _ib_num(act["actual_value"])
    act["e0"] = _period_end(act["fiscal_period_year"], act["fiscal_period_month"], current_date)
    df = act[["ibes_id", "a0", "e0"]].merge(
        est[["ibes_id", "mean_current", "e1"]].rename(columns={"mean_current": "m1"}), on="ibes_id", how="inner"
    )
    twd = pd.Series(pd.Timestamp(monday), index=df.index)
    d = _days(df["e1"], df["e0"])
    r1 = _div(_days(df["e1"], twd), d)
    r2 = _div(_days(twd, df["e0"]), d)
    df[name] = r1 * df["a0"] + r2 * df["m1"]
    return df[["ibes_id", name]].drop_duplicates("ibes_id", keep="first").reset_index(drop=True)


# ----------------------------------------------------------------------------- entity frame
def ibes_entity_frame(univ, run_date):
    """One row per ibes_id, emulating base_qwd.create_initial_datastore (base_qwd.py 281-412) followed by
    `.orderBy(price_usd asc_nulls_last).dropDuplicates(['EntityId'])` in each ibes_raw_* class.

    * Missing values of every column except worldscope_id/worldscope_currency are filled from the first
      security of the same (worldscope_id, worldscope_currency) group (base_qwd.py 403-410), so a share
      line without an ibes_id inherits its sibling's (only affects which line represents the entity).
    * Representative line per ibes_id = lowest price_usd (NULLs last; ties -> lowest security_id;
      Spark does not guarantee the survivor, this is the intended ordering).
    * keep = price_date not null and >= last_month_date (= run_date - relativedelta(months=1, day=31)).
    * currency_units = 1e9 for CLP, COP, IDR, JPY, KRW, VEB else 1e6 (by ibes_currency).
    """
    run_date = _ib_to_date(run_date)
    last_month_date = pd.Timestamp(run_date - relativedelta(months=1, day=31))
    u = univ.copy()
    u["security_id"] = u["security_id"].astype("int64")
    for c in ("worldscope_id", "ibes_id", "ibes_currency", "worldscope_currency"):
        if c not in u.columns:
            u[c] = None
        u[c] = _ib_str_ids(u[c])
    for c in ("price_usd", "ibes_fx_rate", "worldscope_fx_rate"):
        if c not in u.columns:
            u[c] = np.nan
        u[c] = _ib_num(u[c])
    u["price_date"] = pd.to_datetime(u["price_date"]) if "price_date" in u.columns else pd.NaT
    u = u.sort_values("security_id", kind="mergesort").reset_index(drop=True)

    cols = ["ibes_id", "price_usd", "price_date", "ibes_currency", "ibes_fx_rate", "worldscope_fx_rate"]
    keyed = u["worldscope_id"].notna() & u["worldscope_currency"].notna()
    if keyed.any():
        grp = u[keyed].groupby(["worldscope_id", "worldscope_currency"], sort=False)
        for c in cols:
            first = grp[c].transform("first").reindex(u.index)
            u[c] = u[c].where(u[c].notna() | ~keyed, first)
    u["ibes_id"] = _ib_str_ids(u["ibes_id"])
    u["price_usd"] = _ib_num(u["price_usd"])
    u["ibes_fx_rate"] = _ib_num(u["ibes_fx_rate"])
    u["worldscope_fx_rate"] = _ib_num(u["worldscope_fx_rate"])
    u["price_date"] = pd.to_datetime(u["price_date"])

    e = u[u["ibes_id"].notna()].copy()
    e["_nullp"] = e["price_usd"].isna().astype(int)
    e = e.sort_values(["_nullp", "price_usd", "security_id"], kind="mergesort").drop_duplicates("ibes_id")
    e["currency_units"] = np.where(e["ibes_currency"].isin(HIGH_UNIT_CURRENCIES), 1e9, 1e6)
    e["keep"] = e["price_date"].notna() & (e["price_date"] >= last_month_date)
    return e[
        [
            "ibes_id",
            "security_id",
            "worldscope_id",
            "price_usd",
            "price_date",
            "ibes_currency",
            "ibes_fx_rate",
            "worldscope_fx_rate",
            "currency_units",
            "keep",
        ]
    ].reset_index(drop=True)


def _broadcast(univ, ent, factors, apply_keep=True):
    """Map entity-level (ibes_id) values back to every security in univ by its own ibes_id."""
    ent = ent.copy()
    if apply_keep and "keep" in ent.columns:
        ent.loc[~ent["keep"].astype(bool), factors] = np.nan
    out = pd.DataFrame({"security_id": univ["security_id"].astype("int64").values, "ibes_id": _ib_str_ids(univ["ibes_id"]).values})
    vals = ent[["ibes_id", *factors]].drop_duplicates("ibes_id")
    vals = vals[vals["ibes_id"].notna()]
    out = out.merge(vals, on="ibes_id", how="left")
    out = out[["security_id", *factors]].copy()
    for c in factors:
        out[c] = out[c].astype(float)
    return out.reset_index(drop=True)


def _ws_frame(get_ws, metrics, run_date):
    ws = get_ws(list(metrics), run_date)
    ws = ws.copy()
    ws["worldscope_id"] = _ib_str_ids(ws["worldscope_id"])
    for m in metrics:
        ws[m] = _ib_num(ws[m]) if m in ws.columns else np.nan
    return ws[["worldscope_id", *metrics]].drop_duplicates("worldscope_id")


def _breadth(df, k=None):
    """0.5*(up_FY1-down_FY1)/n_FY1 + 0.5*(up_FY2-down_FY2)/n_FY2 (suffix _FYp or _FYp_k)."""
    s = (lambda c, p: f"{c}_{p}") if k is None else (lambda c, p: f"{c}_{p}_{k}")
    return _div(df[s(_UP, "FY1")] - df[s(_DN, "FY1")], df[s(_N, "FY1")]) * 0.5 + _div(
        df[s(_UP, "FY2")] - df[s(_DN, "FY2")], df[s(_N, "FY2")]
    ) * 0.5


def _sign(v):
    """when(v>0,1).otherwise(when(v==0,0).otherwise(when(v<0,-1))) -> NULL for NULL v."""
    return pd.Series(np.select([v > 0, v == 0, v < 0], [1.0, 0.0, -1.0], default=np.nan), index=v.index)


# ----------------------------------------------------------------------------- EPS
def compute_ibes_eps(univ, get_estimates, get_actuals, get_ws, run_date, current_date=None):
    """ibes_raw_eps_metrics (metrics.py 4315-5763).

    Main estimates: EPS FY1..FY3 at run_date (NOT Monday-rolled) + FY0 actual_value (get_actuals at
    run_date). eps_ntm / eps_ltm: base_ntm / base_ltm at the Monday on/before run_date.
    Revision snapshots k=0..5: run_date and month ends run_date - relativedelta(months=k, day=31), each
    FY1/FY2 inner-joined (entity missing either period at a snapshot -> that snapshot NULL).
    Factors and gates (n = number_of_current_estimates, see module docstring for NULL semantics):
      earnings_revisions_1_mth  greatest(nFY1,nFY2)<2 ; breadth(FY1,FY2) clipped to [-1,1]
      eps_growth_fy0_to_fy1     nFY1<2 ; sign-case(actual_FY0 -> mean_FY1)
      eps_growth_fy1_to_fy2     greatest(nFY1,nFY2)<2 ; sign-case(FY1 -> FY2)
      eps_growth_fy1_to_fy3     greatest(nFY1,nFY3)<2 ; compound sign-case(FY1 -> FY3)
      eps_growth_fy2_to_fy3     greatest(nFY2,nFY3)<2 ; sign-case(FY2 -> FY3)
      eps_growth_ltm_to_ntm     nFY1<2 ; sign-case(eps_ltm -> eps_ntm)
      eps_uncertainty           greatest(nFY1,nFY2)<2 ; sd_FY1/|mean_FY1|
      standardised_unexpected_earnings greatest(nFY1,nFY2)<2 ;
                                ((mean_FY1-mean3m_FY1)/sd_FY1 + (mean_FY2-mean3m_FY2)/sd_FY2)/2
      earnings_yield_ntm        greatest(nFY1,nFY2)<2 ; eps_ntm*ibes_fx/price_usd
      earnings_yield_ltm        nFY1<2 ; eps_ltm*ibes_fx/price_usd
      earnings_yield_fyK        nFYK<2 ; mean_FYK*ibes_fx/price_usd
      earnings_revisions_4_mth  greatest(nFY1_0,nFY2_0)<2 ; clip(0.4B0+0.3B1+0.2B2+0.1B3)
      earnings_revision_persistence_6mth greatest(nFY1_0,nFY2_0)<2 ; sum_k=0..5 sign(B_k)
    Entity rows / post price-date filter: see ibes_entity_frame. get_ws unused (signature uniformity).
    """
    run_date = _ib_to_date(run_date)
    cd = current_date or run_date
    ent = ibes_entity_frame(univ, run_date)

    est_raw = _prep_est(get_estimates("eps", run_date, periods=("FY1", "FY2", "FY3")))
    est = _wide(est_raw, ("FY1", "FY2", "FY3"))
    act = get_actuals("eps", run_date).copy()
    act["ibes_id"] = _ib_str_ids(act["ibes_id"])
    act["actual_value_FY0"] = _ib_num(act["actual_value"])
    act = act[["ibes_id", "actual_value_FY0"]].drop_duplicates("ibes_id")

    snaps = None
    dates = [run_date] + [run_date - relativedelta(months=k, day=31) for k in range(1, 6)]
    snap_cols = [_UP, _DN, _N]
    for k, d in enumerate(dates):
        s_raw = est_raw if k == 0 else _prep_est(get_estimates("eps", d, periods=("FY1", "FY2")))
        f1 = _wide(s_raw, ("FY1",), snap_cols, suffix_fmt="{c}_{p}_" + str(k))
        f2 = _wide(s_raw, ("FY2",), snap_cols, suffix_fmt="{c}_{p}_" + str(k))
        s = f1.merge(f2, on="ibes_id", how="inner")
        snaps = s if snaps is None else snaps.merge(s, on="ibes_id", how="outer")

    df = (
        ent.merge(est, on="ibes_id", how="left")
        .merge(act, on="ibes_id", how="left")
        .merge(ntm("eps", run_date, get_estimates, cd), on="ibes_id", how="left")
        .merge(ltm("eps", run_date, get_estimates, get_actuals, cd), on="ibes_id", how="left")
        .merge(snaps, on="ibes_id", how="left")
    )
    n1, n2, n3 = df[f"{_N}_FY1"], df[f"{_N}_FY2"], df[f"{_N}_FY3"]
    m1, m2, m3 = df["mean_current_FY1"], df["mean_current_FY2"], df["mean_current_FY3"]
    fx, px = df["ibes_fx_rate"], df["price_usd"]

    df["earnings_revisions_1_mth"] = _clip1(_gate2(_breadth(df), n1, n2))
    df["eps_growth_fy0_to_fy1"] = _gate(_growth(df["actual_value_FY0"], m1), n1)
    df["eps_growth_fy1_to_fy2"] = _gate2(_growth(m1, m2), n1, n2)
    df["eps_growth_fy1_to_fy3"] = _gate2(_growth(m1, m3, compound=True), n1, n3)
    df["eps_growth_fy2_to_fy3"] = _gate2(_growth(m2, m3), n2, n3)
    df["eps_growth_ltm_to_ntm"] = _gate(_growth(df["eps_ltm"], df["eps_ntm"]), n1)
    df["eps_uncertainty"] = _gate2(_div(df["standard_deviation_current_FY1"], m1.abs()), n1, n2)
    sue = (
        _div(m1 - df["mean_three_months_ago_FY1"], df["standard_deviation_current_FY1"])
        + _div(m2 - df["mean_three_months_ago_FY2"], df["standard_deviation_current_FY2"])
    ) / 2
    df["standardised_unexpected_earnings"] = _gate2(sue, n1, n2)
    df["earnings_yield_ntm"] = _gate2(_div(df["eps_ntm"] * fx, px), n1, n2)
    df["earnings_yield_ltm"] = _gate(_div(df["eps_ltm"] * fx, px), n1)
    df["earnings_yield_fy1"] = _gate(_div(m1 * fx, px), n1)
    df["earnings_yield_fy2"] = _gate(_div(m2 * fx, px), n2)
    df["earnings_yield_fy3"] = _gate(_div(m3 * fx, px), n3)

    b = {k: _breadth(df, k) for k in range(6)}
    n1_0, n2_0 = df[f"{_N}_FY1_0"], df[f"{_N}_FY2_0"]
    r4 = b[0] * 0.4 + b[1] * 0.3 + b[2] * 0.2 + b[3] * 0.1
    df["earnings_revisions_4_mth"] = _clip1(_gate2(r4, n1_0, n2_0))
    pers = sum(_sign(b[k]) for k in range(6))
    df["earnings_revision_persistence_6mth"] = _gate2(pers, n1_0, n2_0)
    return _broadcast(univ, df, EPS_FACTORS)


# ----------------------------------------------------------------------------- EBIT
def compute_ibes_ebit(univ, get_estimates, get_actuals, get_ws, run_date, current_date=None):
    """ibes_raw_ebit_metrics (metrics.py 5765-6434).

    EBIT (EBIT_NON_PER_SHARE, 'ebi') FY1..FY3 at run_date; ebit_ntm/ebit_ltm and cfps_ntm/cfps_ltm
    ('cps') via base_ntm/base_ltm (Monday rule); common_shares_outstanding & total_assets from
    get_ws(..., run_date) for the entity's representative security (production joins
    Financials on quantlake_security_id).
      ebit_yield_fyK   nFYK<2 ; mean_FYK*ibes_fx*currency_units/(shares*price_usd)
      ebit_yield_ntm   nFY1<2 ; ebit_ntm*ibes_fx*currency_units/(shares*price_usd)
      ebit_yield_ltm   nFY1<2 ; ebit_ltm*...
      cash_flow_growth_ltm_to_ntm  nFY1(EBIT)<2 ;
          cfps_ntm*ibes_fx*shares/(total_assets*ws_fx) - cfps_ltm*ibes_fx*shares/(total_assets*ws_fx)
          (no currency_units, as coded)
      ebit_growth_ltm_to_ntm  greatest(nFY1,nFY2)<2 ; sign-case(ebit_ltm -> ebit_ntm)
      ebit_growth_fy1_to_fy2  greatest(nFY1,nFY2)<2 ; sign-case ; fy2_to_fy3 greatest(nFY2,nFY3)<2 ;
      ebit_growth_fy1_to_fy3  greatest(nFY1,nFY3)<2 ; compound sign-case.
    """
    run_date = _ib_to_date(run_date)
    cd = current_date or run_date
    ent = ibes_entity_frame(univ, run_date)
    est = _wide(_prep_est(get_estimates("ebi", run_date, periods=("FY1", "FY2", "FY3"))), ("FY1", "FY2", "FY3"))
    ws = _ws_frame(get_ws, ["common_shares_outstanding", "total_assets"], run_date)
    df = (
        ent.merge(est, on="ibes_id", how="left")
        .merge(ws, on="worldscope_id", how="left")
        .merge(ntm("ebi", run_date, get_estimates, cd), on="ibes_id", how="left")
        .merge(ltm("ebi", run_date, get_estimates, get_actuals, cd), on="ibes_id", how="left")
        .merge(ntm("cps", run_date, get_estimates, cd), on="ibes_id", how="left")
        .merge(ltm("cps", run_date, get_estimates, get_actuals, cd), on="ibes_id", how="left")
    )
    n1, n2, n3 = df[f"{_N}_FY1"], df[f"{_N}_FY2"], df[f"{_N}_FY3"]
    m1, m2, m3 = df["mean_current_FY1"], df["mean_current_FY2"], df["mean_current_FY3"]
    fx, px, cu = df["ibes_fx_rate"], df["price_usd"], df["currency_units"]
    sh, ta, wfx = df["common_shares_outstanding"], df["total_assets"], df["worldscope_fx_rate"]
    mcap = sh * px

    df["ebit_yield_fy1"] = _gate(_div(m1 * fx * cu, mcap), n1)
    df["ebit_yield_fy2"] = _gate(_div(m2 * fx * cu, mcap), n2)
    df["ebit_yield_fy3"] = _gate(_div(m3 * fx * cu, mcap), n3)
    df["ebit_yield_ntm"] = _gate(_div(df["ebit_ntm"] * fx * cu, mcap), n1)
    df["ebit_yield_ltm"] = _gate(_div(df["ebit_ltm"] * fx * cu, mcap), n1)
    den = ta * wfx
    df["cash_flow_growth_ltm_to_ntm"] = _gate(
        _div(df["cfps_ntm"] * fx * sh, den) - _div(df["cfps_ltm"] * fx * sh, den), n1
    )
    df["ebit_growth_ltm_to_ntm"] = _gate2(_growth(df["ebit_ltm"], df["ebit_ntm"]), n1, n2)
    df["ebit_growth_fy1_to_fy2"] = _gate2(_growth(m1, m2), n1, n2)
    df["ebit_growth_fy2_to_fy3"] = _gate2(_growth(m2, m3), n2, n3)
    df["ebit_growth_fy1_to_fy3"] = _gate2(_growth(m1, m3, compound=True), n1, n3)
    return _broadcast(univ, df, EBIT_FACTORS)


# ----------------------------------------------------------------------------- Revenue
def compute_ibes_revenue(univ, get_estimates, get_actuals, get_ws, run_date, current_date=None):
    """ibes_raw_revenue_metrics (metrics.py 6436-6705).

    Revenue ('sal') FY1..FY3 at run_date (counts below are REVENUE counts); variables sales_ntm,
    sales_ltm, roe_ntm, dps_ntm, eps_ntm via base_ntm/base_ltm; shares from get_ws.
      sales_yield_ntm          nFY1<2 ; sales_ntm*ibes_fx*currency_units/(shares*price_usd)
      sales_growth_ltm_to_ntm  nFY1<2 ; (sales_ntm-sales_ltm)/|sales_ltm|  (no sign-case table)
      sales_growth_fy1_to_fy3  greatest(nFY1,nFY3)<2 ; ((FY3-FY1)/|FY1| + 1)^0.5 - 1 (NaN if base < 0)
      sales_revisions_1_mth    greatest(nFY1,nFY2)<2 ; breadth clipped [-1,1]
      return_on_equity_ntm     nFY1<2 ; roe_ntm/100
      sustainable_growth_ntm   nFY1<2 ; roe_ntm*(1 - dps_ntm/eps_ntm)/100
    """
    run_date = _ib_to_date(run_date)
    cd = current_date or run_date
    ent = ibes_entity_frame(univ, run_date)
    est = _wide(_prep_est(get_estimates("sal", run_date, periods=("FY1", "FY2", "FY3"))), ("FY1", "FY2", "FY3"))
    ws = _ws_frame(get_ws, ["common_shares_outstanding"], run_date)
    df = ent.merge(est, on="ibes_id", how="left").merge(ws, on="worldscope_id", how="left")
    for frame in (
        ntm("sal", run_date, get_estimates, cd),
        ltm("sal", run_date, get_estimates, get_actuals, cd),
        ntm("roe", run_date, get_estimates, cd),
        ntm("dps", run_date, get_estimates, cd),
        ntm("eps", run_date, get_estimates, cd),
    ):
        df = df.merge(frame, on="ibes_id", how="left")
    n1, n2, n3 = df[f"{_N}_FY1"], df[f"{_N}_FY2"], df[f"{_N}_FY3"]
    m1, m3 = df["mean_current_FY1"], df["mean_current_FY3"]
    fx, px, cu, sh = df["ibes_fx_rate"], df["price_usd"], df["currency_units"], df["common_shares_outstanding"]

    df["sales_yield_ntm"] = _gate(_div(df["sales_ntm"] * fx * cu, sh * px), n1)
    df["sales_growth_ltm_to_ntm"] = _gate(_div(df["sales_ntm"] - df["sales_ltm"], df["sales_ltm"].abs()), n1)
    df["sales_growth_fy1_to_fy3"] = _gate2(_sqrt(_div(m3 - m1, m1.abs()) + 1) - 1, n1, n3)
    df["sales_revisions_1_mth"] = _clip1(_gate2(_breadth(df), n1, n2))
    df["return_on_equity_ntm"] = _gate(df["roe_ntm"] / 100, n1)
    df["sustainable_growth_ntm"] = _gate(df["roe_ntm"] * (1 - _div(df["dps_ntm"], df["eps_ntm"])) / 100, n1)
    return _broadcast(univ, df, REVENUE_FACTORS)


# ----------------------------------------------------------------------------- Dividends
def compute_ibes_dividends(univ, get_estimates, get_actuals, get_ws, run_date, current_date=None):
    """ibes_raw_dividends_metrics (metrics.py 6707-6800).

      dividend_yield_ntm  nFY1(DPS, at run_date)<2 ; dps_ntm*ibes_fx/price_usd
    """
    run_date = _ib_to_date(run_date)
    cd = current_date or run_date
    ent = ibes_entity_frame(univ, run_date)
    est = _wide(_prep_est(get_estimates("dps", run_date, periods=("FY1",))), ("FY1",))
    df = ent.merge(est, on="ibes_id", how="left").merge(ntm("dps", run_date, get_estimates, cd), on="ibes_id", how="left")
    df["dividend_yield_ntm"] = _gate(_div(df["dps_ntm"] * df["ibes_fx_rate"], df["price_usd"]), df[f"{_N}_FY1"])
    return _broadcast(univ, df, DIVIDEND_FACTORS)


# ----------------------------------------------------------------------------- NTM change
def compute_ibes_ntm_change(univ, get_estimates, get_actuals, get_ws, run_date, current_date=None):
    """ibes_raw_ntm_change_metrics (metrics.py 6802-6857).

    Production reads the STORED eps_ntm metric at CRISP offsets '0D', '-1M', '-3M'. Deviation (data
    access): recomputed here with ntm('eps', run_date - relativedelta(months=k)) (base_ntm, Monday rule),
    i.e. the value eps_ntm would have been stored with on/before that date.
      eps_ntm_1mth_chg  n(EPS FY1 at run_date)<2 ; (eps_ntm_0 - eps_ntm_-1M)/|eps_ntm_-1M|
      eps_ntm_3mth_chg  same gate ; (eps_ntm_0 - eps_ntm_-3M)/|eps_ntm_-3M|
    No price-date filter / entity dedupe in production for this class.
    """
    run_date = _ib_to_date(run_date)
    cd = current_date or run_date
    cur = ntm("eps", run_date, get_estimates, cd).rename(columns={"eps_ntm": "cur"})
    l1 = ntm("eps", run_date - relativedelta(months=1), get_estimates, cd).rename(columns={"eps_ntm": "l1"})
    l3 = ntm("eps", run_date - relativedelta(months=3), get_estimates, cd).rename(columns={"eps_ntm": "l3"})
    ds = cur.merge(l1, on="ibes_id", how="outer").merge(l3, on="ibes_id", how="outer")
    fy1 = _wide(_prep_est(get_estimates("eps", run_date, periods=("FY1",))), ("FY1",), [_N], suffix_fmt="{c}")
    df = ds.merge(fy1, on="ibes_id", how="left")
    df["eps_ntm_1mth_chg"] = _gate(_div(df["cur"] - df["l1"], df["l1"].abs()), df[_N])
    df["eps_ntm_3mth_chg"] = _gate(_div(df["cur"] - df["l3"], df["l3"].abs()), df[_N])
    return _broadcast(univ, df, NTM_CHANGE_FACTORS, apply_keep=False)


# ----------------------------------------------------------------------------- all
def compute_all_ibes(univ, get_estimates, get_actuals, get_ws, run_date, current_date=None):
    """All I/B/E/S factors (eps, ebit, revenue, dividends, ntm change) merged on security_id."""
    out = pd.DataFrame({"security_id": univ["security_id"].astype("int64").values})
    for fn in (compute_ibes_eps, compute_ibes_ebit, compute_ibes_revenue, compute_ibes_dividends, compute_ibes_ntm_change):
        res = fn(univ, get_estimates, get_actuals, get_ws, run_date, current_date)
        out = out.merge(res.drop_duplicates("security_id"), on="security_id", how="left")
    return out[["security_id", *ALL_IBES_FACTORS]]

# COMMAND ----------

# MAGIC %md ## 5. Cleaning, z-scores and composites
# MAGIC Cleaning: values outside the `settings.metric_rules` limits are set to null (not clipped); financials are excluded per rule.
# MAGIC Composites replicate metrics.py 6919-9100 (value, growth, quality, momentum, 4-month revisions, QGARP).

# COMMAND ----------

"""GEMS composite factors – pandas/numpy replication of production PSA_Crisp_Metrics.

Sources replicated (AS CODED, quirks included):
  * psa_common_functions/utils.py 9-84            generate_clean_factors
  * psa_crisp_metrics/base_classes/base_qwd.py 415-588  base_qwd_clean_factors
  * psa_crisp_metrics/base_classes/base_qwd.py 591-699  base_qwd_zscore_pct_rank
  * psa_crisp_metrics/investments/metrics.py 6919-9100  pct_rank_value_composite, pct_rank_growth_composite,
    pct_rank_quality_composite, zscore_momentum_composite, zscore_earnings_revisions_4_mth_composite, pct_rank_qgarp

Conventions shared by all composite functions (production "DataDate" groupby == one run date here, so a single
cross-section):
  * Cross-section = the rows of ``univ`` (qwd_global universe, one row per security_id). Production left-joins the
    universe to the datastore and then INNER-merges an industry table from Financials.get (securities with no
    Financials industry record are dropped from the population). Here industry comes from
    ``univ.general_industry_classification`` and every universe security is kept (unclassified -> industry_group
    NaN). This is a data-access substitution.
  * Production stores intermediate pct ranks in object arrays (np.where(..., None)); pandas treats those exactly
    like float columns with NaN for rank/mean/arithmetics, so float64 + NaN is used throughout.
  * pandas percentile rank pattern: ``groupby(keys)[f].rank(pct=True, ascending=False)`` (method='average',
    pct = rank / n_non_null, NaN group keys dropped) -> ``ceil(pct*100)`` with the (never firing) ``== 0 -> 1``
    guard; null input -> null. 1 = highest raw value.
"""
import calendar
import datetime as _dt
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd

_ID_COLS = {"security_id", "worldscope_id", "ibes_id", "EntityId", "DataDate"}
_DD = "_datadate"  # constant stand-in for production's DataDate groupby key (single run date)


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def _cp_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _industry(univ: pd.DataFrame) -> pd.Series:
    if "general_industry_classification" not in univ.columns:
        return pd.Series([None] * len(univ), index=univ.index, dtype=object)
    s = univ["general_industry_classification"]
    return s.where(s.notna(), None).astype(object)


def _base_frame(univ: pd.DataFrame, data: Optional[pd.DataFrame], cols: Iterable[str]) -> pd.DataFrame:
    """One row per univ security with the requested numeric columns (missing column -> all NaN, crash guard)."""
    base = pd.DataFrame({"security_id": univ["security_id"].to_numpy()})
    base["general_industry_classification"] = _industry(univ).to_numpy()
    cols = list(dict.fromkeys(cols))
    if data is not None:
        avail = [c for c in cols if c in data.columns]
        d = data[["security_id"] + avail].drop_duplicates(subset=["security_id"])
        base = base.merge(d, on="security_id", how="left")
    for c in cols:
        if c not in base.columns:
            # crash guard: production would fail on a missing column; here it is treated as all-null
            base[c] = np.nan
        base[c] = _cp_num(base[c])
    base[_DD] = 0
    return base


def _pd_pct_rank(df: pd.DataFrame, col: str, keys: List[str], ascending: bool = False) -> np.ndarray:
    """Production pattern:
    np.where(~x.isnull(), np.where(rank == 0, 1, np.ceil(rank * 100)), None),
    rank = df.groupby(keys)[col].rank(pct=True, ascending=ascending)."""
    r = df.groupby(keys)[col].rank(pct=True, ascending=ascending)
    inner = np.where(r == 0, 1, np.ceil(r * 100))
    return np.where(~df[col].isnull(), inner, np.nan).astype("float64")


def _notnull_count(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    out = pd.Series(0, index=df.index)
    for c in cols:
        out = out + pd.notnull(df[c]).astype(int)
    return out


def _neg_mean(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """-df[cols].mean(axis=1) (skipna; all-null -> NaN)."""
    return -df[cols].mean(axis=1)


def _pick_by_group(group: pd.Series, mapping: dict) -> np.ndarray:
    out = np.full(len(group), np.nan)
    g = group.to_numpy()
    for k, values in mapping.items():
        out = np.where(g == k, np.asarray(values, dtype="float64"), out)
    return out


def _zscore_sample(x: pd.Series) -> pd.Series:
    """production ``def zscore(x): return (x - x.mean()) / x.std()`` (pandas, skipna, ddof=1)."""
    return (x - x.mean()) / x.std()


def _clip3(z) -> np.ndarray:
    z = np.asarray(z, dtype="float64")
    return np.where(z < -3, -3, np.where(z > 3, 3, z))


def _month_end_back(d: _dt.date, k: int) -> _dt.date:
    """datadate - relativedelta(months=k, day=31) == last calendar day of the month k months before."""
    m0 = d.year * 12 + (d.month - 1) - k
    y, m = divmod(m0, 12)
    m += 1
    return _dt.date(y, m, calendar.monthrange(y, m)[1])


def _cp_to_date(d) -> _dt.date:
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    return pd.Timestamp(d).date()


# ----------------------------------------------------------------------------------------------------------------
# 1. cleaning
# ----------------------------------------------------------------------------------------------------------------
def clean_factors(raw: pd.DataFrame, rules: pd.DataFrame, univ: pd.DataFrame) -> pd.DataFrame:
    """Replicates base_qwd_clean_factors.clean_factors (base_qwd.py 415-496) + utils.generate_clean_factors
    (utils.py 9-84).

    For every factor column F of ``raw`` (all columns except id columns) returns ``F_clean``:
      value -> NULL if  NOT (value BETWEEN lower_limit AND upper_limit)         (inclusive)
                    OR (exclude_financials == 1 AND industry IN (BANK, INSURANCE, OTHER FINANCIAL))
                    OR (exclude_financials == 2 AND industry IN (BANK, INSURANCE))
    evaluated with Spark three-valued logic inside ``F.when(cond, None).otherwise(value)`` – only a TRUE
    condition nulls the value. Consequences replicated here:
      * no rule row / both limits null -> BETWEEN is NULL -> value kept;
      * only one limit present -> ``x>=lo AND NULL`` is FALSE when x<lo (nulled) else NULL (kept): i.e. the
        present limit is enforced, the missing one is not;
      * exclude_financials null or industry null -> that term is NULL/FALSE -> no exclusion.
    Values outside limits are EXCLUDED (set null), not clipped. Null inputs stay null.
    Several rule rows for one barings_metric_name (pivot by marketdata_metric_id) -> production pivots with
    max(metric_value_rules) -> value kept if ANY rule row keeps it.

    rules: barings_metric_name, lower_limit, upper_limit, exclude_financials (floats, NaN if absent).
    Industry: univ.general_industry_classification by security_id (production: Financials.get industry with
    primary-issue back-fill joined on the worldscope/ibes EntityId – data-access substitution).

    NOT replicated: base_qwd_clean_factors.map_parent_data (base_qwd.py 498-571) – the ultimate-parent
    coalesce(child, parent) back-fill used by worldscope_clean_metrics needs the securities_vendor_mappings
    parent hierarchy, which is not available to this notebook. (ibes_clean_metrics, metrics.py 6859-6906,
    does not call it anyway.)
    """
    factors = [c for c in raw.columns if c not in _ID_COLS]
    df = raw[["security_id"] + factors].copy()
    ind = univ[["security_id"]].copy()
    ind["_ind"] = _industry(univ).to_numpy()
    ind = ind.drop_duplicates(subset=["security_id"])
    df = df.merge(ind, on="security_id", how="left")
    industry = df["_ind"]
    fin1 = industry.isin(["BANK", "INSURANCE", "OTHER FINANCIAL"]).to_numpy()
    fin2 = industry.isin(["BANK", "INSURANCE"]).to_numpy()

    rules = rules if rules is not None else pd.DataFrame(columns=["barings_metric_name"])
    cols = {"security_id": df["security_id"].to_numpy()}
    for f in factors:
        x = _cp_num(df[f]).to_numpy()
        r = rules[rules["barings_metric_name"].astype(str) == str(f)] if len(rules) else rules
        if len(r) == 0:
            # left join finds no rule -> all rule columns NULL -> condition never TRUE -> value kept
            cols[f"{f}_clean"] = x
            continue
        keep_any = np.zeros(len(x), dtype=bool)
        for _, rule in r.iterrows():
            lo = pd.to_numeric(rule.get("lower_limit", np.nan), errors="coerce")
            hi = pd.to_numeric(rule.get("upper_limit", np.nan), errors="coerce")
            ex = pd.to_numeric(rule.get("exclude_financials", np.nan), errors="coerce")
            with np.errstate(invalid="ignore"):
                cond = np.zeros(len(x), dtype=bool)
                if pd.notna(lo):
                    cond |= x < lo
                if pd.notna(hi):
                    cond |= x > hi
            if pd.notna(ex) and ex == 1:
                cond |= fin1
            if pd.notna(ex) and ex == 2:
                cond |= fin2
            keep_any |= ~cond
        cols[f"{f}_clean"] = np.where(keep_any & ~np.isnan(x), x, np.nan)
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------------------------------------------
# 2. z-score / pct rank (base_qwd_zscore_pct_rank)
# ----------------------------------------------------------------------------------------------------------------
def zscore_pct_rank(clean: pd.DataFrame, cols: List[str], univ: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Replicates base_qwd_zscore_pct_rank.compute (base_qwd.py 591-699), used by
    worldscope_zscore_pct_rank_price_metrics (metrics.py 1920, *_mth_price_return_local_clean) and
    ibes_zscore_pct_rank_metrics (metrics.py 6909, earnings_revisions_1_mth_clean).

      zscore_<col>   = (x - avg(x)) / sqrt(avg(x*x) - avg(x)^2)   (Spark avg ignores nulls -> equal-weighted
                       mean, POPULATION std; division by 0 -> null; no clipping)
      pct_rank_<col> = on non-null x: Spark percent_rank() ordered by -x, i.e. (rank_min - 1)/(n - 1)
                       (ties share the lowest rank; n == 1 -> 0); 0 -> 1 else ceil(pr*100). 1 = highest value.

    Cross-section: production INNER-joins the entity datastore to the qwd_global universe (EntityId == ibes or
    worldscope of each universe security), so the population is one row per UNIVERSE SECURITY (an entity with
    several share lines in the universe is counted once per line). Pass ``clean`` with one row per univ security
    (entity values broadcast to each line) – or pass ``univ`` to restrict ``clean`` to univ.security_id.
    Output: security_id + zscore_<col> + pct_rank_<col>, one row per row of the (restricted) ``clean``.
    """
    df = clean
    if univ is not None:
        df = df[df["security_id"].isin(univ["security_id"])]
    df = df.reset_index(drop=True)
    out = pd.DataFrame({"security_id": df["security_id"].to_numpy()})
    for c in cols:
        x = _cp_num(df[c]) if c in df.columns else pd.Series(np.nan, index=df.index)
        valid = x.notna()
        n = int(valid.sum())
        if n:
            xv = x[valid].to_numpy()
            avg = xv.sum() / n
            avg_sq = (xv * xv).sum() / n
            with np.errstate(invalid="ignore"):
                sd = np.sqrt(avg_sq - avg * avg)
            if sd == 0:
                z = np.full(len(x), np.nan)  # Spark x/0 -> null
            else:
                z = ((x - avg) / sd).to_numpy()
        else:
            z = np.full(len(x), np.nan)
        out[f"zscore_{c}"] = z

        rank_min = (-x).rank(method="min", ascending=True)  # orderBy(-x) ascending
        if n > 1:
            pr = (rank_min - 1.0) / float(n - 1)
        else:
            pr = rank_min * 0.0  # Spark percent_rank of a single-row partition is 0
        pct = np.where(pr == 0, 1.0, np.ceil(pr * 100))
        out[f"pct_rank_{c}"] = np.where(valid, pct, np.nan)
    return out


# ----------------------------------------------------------------------------------------------------------------
# 3. composites
# ----------------------------------------------------------------------------------------------------------------
VALUE_INPUTS = [
    "earnings_yield_ntm_clean", "earnings_yield_clean", "book_yield_clean", "cash_flow_yield_clean",
    "free_cash_flow_yield_clean", "sales_yield_clean", "ebitda_to_ev_clean",
]


def compute_value_composite(clean: pd.DataFrame, univ: pd.DataFrame) -> pd.DataFrame:
    """Replicates class pct_rank_value_composite (metrics.py 6919-7236).

    industry_group: 1 = BANK/INSURANCE/OTHER FINANCIAL, 2 = INDUSTRIAL/UTILITY/TRANSPORTATION, else null.
    earnings_yield_combined = earnings_yield_ntm_clean (IBES) else earnings_yield_clean (Worldscope).
    cash_flow/free_cash_flow/sales yield and ebitda_to_ev nulled for industry_group 1 ("_ex_fins").
    Each of the 6 constituents pct-ranked descending across the WHOLE universe (DataDate only, incl. unclassified).
    value_score_financials     = -mean(2 fin ranks)  if >=1 non-null else null
    value_score_non_financials = -mean(6 ranks)      if >=4 non-null else null
    pct_rank_value_composite               = universe-wide re-rank of the group's score (1 = cheapest)
    pct_rank_value_composite_fins_neutral  = same score re-ranked within (DataDate, industry_group)
    Returns security_id, pct_rank_value_composite, pct_rank_value_composite_fins_neutral.
    """
    df = _base_frame(univ, clean, VALUE_INPUTS)
    gic = df["general_industry_classification"]
    df["industry_group"] = np.where(
        gic.isin(["BANK", "INSURANCE", "OTHER FINANCIAL"]), 1,
        np.where(gic.isin(["INDUSTRIAL", "UTILITY", "TRANSPORTATION"]), 2, np.nan),
    ).astype("float64")

    df["earnings_yield_combined"] = np.where(
        df["earnings_yield_ntm_clean"].isnull(), df["earnings_yield_clean"], df["earnings_yield_ntm_clean"]
    )
    for f in ["cash_flow_yield_clean", "free_cash_flow_yield_clean", "sales_yield_clean", "ebitda_to_ev_clean"]:
        df[f"{f}_ex_fins"] = np.where(df["industry_group"] == 1, np.nan, df[f])

    factors = [
        "earnings_yield_combined", "book_yield_clean", "cash_flow_yield_clean_ex_fins",
        "free_cash_flow_yield_clean_ex_fins", "sales_yield_clean_ex_fins", "ebitda_to_ev_clean_ex_fins",
    ]
    for f in factors:
        df[f"pct_rank_{f}"] = _pd_pct_rank(df, f, [_DD])

    fin_cols = ["pct_rank_earnings_yield_combined", "pct_rank_book_yield_clean"]
    all_cols = [f"pct_rank_{f}" for f in factors]
    df["value_score_financials"] = np.where(_notnull_count(df, fin_cols) > 0, _neg_mean(df, fin_cols), np.nan)
    df["value_score_non_financials"] = np.where(_notnull_count(df, all_cols) > 3, _neg_mean(df, all_cols), np.nan)

    for f in ["value_score_financials", "value_score_non_financials"]:
        df[f"pct_rank_{f}"] = _pd_pct_rank(df, f, [_DD])
        # production: outer ==0 test uses the DataDate rank, value uses the (DataDate, industry_group) rank
        r_all = df.groupby([_DD])[f].rank(pct=True, ascending=False)
        r_grp = df.groupby([_DD, "industry_group"])[f].rank(pct=True, ascending=False)
        df[f"pct_rank_{f}_fins_neutral"] = np.where(
            ~df[f].isnull(), np.where(r_all == 0, 1, np.ceil(r_grp * 100)), np.nan
        )

    grp = df["industry_group"]
    out = pd.DataFrame({"security_id": df["security_id"].to_numpy()})
    out["pct_rank_value_composite"] = _pick_by_group(
        grp, {1: df["pct_rank_value_score_financials"], 2: df["pct_rank_value_score_non_financials"]})
    out["pct_rank_value_composite_fins_neutral"] = _pick_by_group(
        grp, {1: df["pct_rank_value_score_financials_fins_neutral"],
              2: df["pct_rank_value_score_non_financials_fins_neutral"]})
    return out


GROWTH_INPUTS = [
    "cash_flow_growth_ltm_to_ntm_clean", "cash_flow_growth_1_year_change_clean",
    "cash_flow_growth_3_year_change_clean", "eps_growth_ltm_to_ntm_clean", "eps_growth_1_year_hist_clean",
    "eps_growth_3_year_hist_cagr_clean", "shareholder_yield_growth_hist_clean",
    "shareholder_yield_growth_3_year_hist_clean", "sustainable_growth_ntm_clean", "sustainable_growth_clean",
    "sustainable_growth_3_year_average_clean",
]


def compute_growth_composite(clean: pd.DataFrame, univ: pd.DataFrame) -> pd.DataFrame:
    """Replicates class pct_rank_growth_composite (metrics.py 7239-7578).

    industry_group: 1 = BANK/INSURANCE ("bi"), 2 = INDUSTRIAL/UTILITY/TRANSPORTATION/OTHER FINANCIAL, else null.
    All ranks within (DataDate, industry_group), descending; unclassified rows get no rank.
    cf_gth_score  = -mean(3 cash-flow growth ranks)   (no min count)
    eps_gth_score = -mean(3 eps growth ranks)
    sy_gth_score  = -mean(2 shareholder-yield growth ranks)
    sgr_score     = -(0.33*r_ntm + 0.33*r_ltm + 0.33*r_3y) if that is non-null else -(0.33*r_ntm + 0.33*r_ltm)
                    (quirk: 2-term fallback has smaller magnitude; null if r_ntm or r_ltm null)
    growth_score_bi     = -mean(r_eps, r_sy, r_sgr)        if >=2 non-null
    growth_score_non_bi = -mean(r_cf, r_eps, r_sy, r_sgr)  if >=3 non-null
    pct_rank_growth_composite = rank(growth_score_bi) for group 1, rank(growth_score_non_bi) for group 2.
    """
    df = _base_frame(univ, clean, GROWTH_INPUTS)
    gic = df["general_industry_classification"]
    df["industry_group"] = np.where(
        gic.isin(["BANK", "INSURANCE"]), 1,
        np.where(gic.isin(["INDUSTRIAL", "UTILITY", "TRANSPORTATION", "OTHER FINANCIAL"]), 2, np.nan),
    ).astype("float64")
    keys = [_DD, "industry_group"]

    for f in GROWTH_INPUTS:
        df[f"pct_rank_{f}"] = _pd_pct_rank(df, f, keys)

    p = lambda f: f"pct_rank_{f}"  # noqa: E731
    df["cf_gth_score"] = _neg_mean(df, [p("cash_flow_growth_ltm_to_ntm_clean"),
                                        p("cash_flow_growth_1_year_change_clean"),
                                        p("cash_flow_growth_3_year_change_clean")])
    df["eps_gth_score"] = _neg_mean(df, [p("eps_growth_ltm_to_ntm_clean"), p("eps_growth_1_year_hist_clean"),
                                         p("eps_growth_3_year_hist_cagr_clean")])
    df["sy_gth_score"] = _neg_mean(df, [p("shareholder_yield_growth_hist_clean"),
                                        p("shareholder_yield_growth_3_year_hist_clean")])
    s3 = (df[p("sustainable_growth_ntm_clean")] * 0.33 + df[p("sustainable_growth_clean")] * 0.33
          + df[p("sustainable_growth_3_year_average_clean")] * 0.33)
    s2 = df[p("sustainable_growth_ntm_clean")] * 0.33 + df[p("sustainable_growth_clean")] * 0.33
    df["sgr_score"] = np.where(pd.notnull(s3), -s3, -s2)

    for f in ["cf_gth_score", "eps_gth_score", "sy_gth_score", "sgr_score"]:
        df[p(f)] = _pd_pct_rank(df, f, keys)

    bi = [p("eps_gth_score"), p("sy_gth_score"), p("sgr_score")]
    non_bi = [p("cf_gth_score")] + bi
    df["growth_score_bi"] = np.where(_notnull_count(df, bi) > 1, _neg_mean(df, bi), np.nan)
    df["growth_score_non_bi"] = np.where(_notnull_count(df, non_bi) > 2, _neg_mean(df, non_bi), np.nan)
    for f in ["growth_score_bi", "growth_score_non_bi"]:
        df[p(f)] = _pd_pct_rank(df, f, keys)

    out = pd.DataFrame({"security_id": df["security_id"].to_numpy()})
    out["pct_rank_growth_composite"] = _pick_by_group(
        df["industry_group"], {1: df[p("growth_score_bi")], 2: df[p("growth_score_non_bi")]})
    return out


QUALITY_INPUTS = [
    "return_on_equity_clean", "return_on_equity_5_year_vol_clean", "return_on_assets_clean",
    "return_on_assets_5_year_vol_clean", "return_on_invested_capital_clean",
    "return_on_invested_capital_5_year_vol_clean", "net_income_margin_clean", "net_income_margin_5_year_vol_clean",
    "free_cash_flow_conversion_clean", "unusual_items_to_operating_income_clean",
    "cash_flow_from_financing_to_total_assets_clean", "debt_to_assets_clean", "interest_coverage_clean",
    "short_term_debt_to_total_debt_clean", "eps_uncertainty_clean", "net_income_5_year_stability_clean",
    "interest_income_to_interest_expense_clean", "loan_loss_provisions_to_net_loans_clean",
    "loans_to_deposits_clean", "net_income_to_risk_weighted_assets_clean", "net_interest_income_to_net_loan_clean",
    "non_interest_income_to_non_interest_expense_clean",
    "non_performing_loans_to_equity_and_loan_loss_provision_clean", "capital_adequacy_ratio_tier_1_capital_clean",
    "capital_adequacy_ratio_total_capital_clean", "accruals_clean", "accruals_year_on_year_change_clean",
]


def compute_quality_composite(clean: pd.DataFrame, univ: pd.DataFrame) -> pd.DataFrame:
    """Replicates class pct_rank_quality_composite (metrics.py 7581-8491).

    industry_group: 1 = BANK, 2 = INSURANCE/OTHER FINANCIAL, 3 = INDUSTRIAL/UTILITY/TRANSPORTATION, else null.
    Sign flips (higher = better before ranking): unusual_items_to_operating_income, eps_uncertainty,
    cash_flow_from_financing_to_total_assets, debt_to_assets, accruals are negated;
    interest_coverage := 1000 where (negated) debt_to_assets == 0; short_term_debt_to_total_debt := 0 where
    debt_to_assets == 0 else its negation. NOTE (as coded): the 5-year vol / loan-loss / loans-to-deposits /
    NPL / accruals_yoy inputs are NOT negated, so higher volatility etc. ranks better.
    All ranks descending within (DataDate, industry_group). Sub-scores (-mean of ranks, skipna, re-ranked),
    gates on non-null counts and final group selection exactly as lines 7914-8473.
    Returns security_id, pct_rank_quality_composite, pct_rank_quality_alpha, pct_rank_quality_defensive,
    pct_rank_banks_quality_composite (group 1 only), pct_rank_financials_quality_composite (group 2 only),
    pct_rank_non_financials_quality_composite (group 3 only).
    """
    df = _base_frame(univ, clean, QUALITY_INPUTS)
    gic = df["general_industry_classification"]
    df["industry_group"] = np.where(
        gic.isin(["BANK"]), 1,
        np.where(gic.isin(["INSURANCE", "OTHER FINANCIAL"]), 2,
                 np.where(gic.isin(["INDUSTRIAL", "UTILITY", "TRANSPORTATION"]), 3, np.nan)),
    ).astype("float64")
    keys = [_DD, "industry_group"]
    p = lambda f: f"pct_rank_{f}"  # noqa: E731

    for f in ["unusual_items_to_operating_income_clean", "eps_uncertainty_clean",
              "cash_flow_from_financing_to_total_assets_clean", "debt_to_assets_clean", "accruals_clean"]:
        df[f] = -df[f]
    df["interest_coverage_clean"] = np.where(df["debt_to_assets_clean"] == 0, 1000, df["interest_coverage_clean"])
    df["short_term_debt_to_total_debt_clean"] = np.where(
        df["debt_to_assets_clean"] == 0, 0, -df["short_term_debt_to_total_debt_clean"])

    for f in QUALITY_INPUTS:
        df[p(f)] = _pd_pct_rank(df, f, keys)

    def rerank(names):
        for f in names:
            df[p(f)] = _pd_pct_rank(df, f, keys)

    df["banks_quality_alpha_score"] = _neg_mean(df, [p(c) for c in [
        "net_income_to_risk_weighted_assets_clean", "net_interest_income_to_net_loan_clean",
        "non_interest_income_to_non_interest_expense_clean", "capital_adequacy_ratio_tier_1_capital_clean",
        "capital_adequacy_ratio_total_capital_clean"]])
    df["banks_quality_defensive_score"] = _neg_mean(df, [p(c) for c in [
        "interest_income_to_interest_expense_clean", "loan_loss_provisions_to_net_loans_clean",
        "loans_to_deposits_clean", "non_performing_loans_to_equity_and_loan_loss_provision_clean"]])
    df["stable_prof_score"] = _neg_mean(df, [p(c) for c in [
        "return_on_equity_5_year_vol_clean", "return_on_assets_5_year_vol_clean",
        "return_on_invested_capital_5_year_vol_clean", "net_income_margin_5_year_vol_clean",
        "net_income_5_year_stability_clean"]])
    rerank(["banks_quality_alpha_score", "banks_quality_defensive_score", "stable_prof_score"])

    df["stable_prod_eps_uncertainty_score"] = _neg_mean(df, [p("stable_prof_score"), p("eps_uncertainty_clean")])
    df["debt_quality_score"] = _neg_mean(df, [p(c) for c in [
        "debt_to_assets_clean", "interest_coverage_clean", "short_term_debt_to_total_debt_clean"]])
    prof_cols = [p(c) for c in ["return_on_equity_clean", "return_on_assets_clean",
                                "return_on_invested_capital_clean", "net_income_margin_clean",
                                "unusual_items_to_operating_income_clean"]]
    df["profitability_score"] = _neg_mean(df, prof_cols)
    df["cashflow_quality_score"] = _neg_mean(df, [p("free_cash_flow_conversion_clean"),
                                                  p("cash_flow_from_financing_to_total_assets_clean")])
    df["accruals_quality_score"] = _neg_mean(df, [p("accruals_clean"), p("accruals_year_on_year_change_clean")])
    rerank(["stable_prod_eps_uncertainty_score", "debt_quality_score", "profitability_score",
            "cashflow_quality_score", "accruals_quality_score"])

    banks_p1_cols = prof_cols + [p("free_cash_flow_conversion_clean")]
    fin_alpha_cols = prof_cols + [p("cash_flow_from_financing_to_total_assets_clean")]
    df["banks_only_quality_alpha_score_p1"] = df[banks_p1_cols].mean(axis=1)  # NOT negated (as coded)
    df["banks_only_quality_alpha_score"] = _neg_mean(
        df, ["banks_only_quality_alpha_score_p1", p("banks_quality_alpha_score")])
    df["financials_quality_alpha_score"] = _neg_mean(df, fin_alpha_cols)
    df["non_financials_quality_alpha_score"] = _neg_mean(
        df, [p("profitability_score"), p("cashflow_quality_score"), p("accruals_quality_score")])
    df["banks_only_quality_defensive_score"] = _neg_mean(
        df, [p("stable_prod_eps_uncertainty_score"), p("banks_quality_defensive_score")])
    df["financials_quality_defensive_score"] = _neg_mean(
        df, [p("stable_prod_eps_uncertainty_score"), p("debt_quality_score")])
    df["non_financials_quality_defensive_score"] = _neg_mean(
        df, [p("stable_prod_eps_uncertainty_score"), p("debt_quality_score")])
    rerank(["banks_only_quality_alpha_score", "banks_only_quality_defensive_score",
            "financials_quality_alpha_score", "financials_quality_defensive_score",
            "non_financials_quality_alpha_score", "non_financials_quality_defensive_score"])

    banks_comp = _neg_mean(df, [p("banks_only_quality_alpha_score"), p("banks_only_quality_defensive_score")])
    fin_comp = _neg_mean(df, [p("financials_quality_alpha_score"), p("financials_quality_defensive_score")])
    nonfin_comp = _neg_mean(df, [p("non_financials_quality_alpha_score"),
                                 p("non_financials_quality_defensive_score")])

    bank_alpha_raw = [p(c) for c in [
        "net_income_to_risk_weighted_assets_clean", "net_interest_income_to_net_loan_clean",
        "non_interest_income_to_non_interest_expense_clean", "capital_adequacy_ratio_tier_1_capital_clean",
        "capital_adequacy_ratio_total_capital_clean"]]
    bank_def_raw = [p(c) for c in [
        "interest_income_to_interest_expense_clean", "loan_loss_provisions_to_net_loans_clean",
        "loans_to_deposits_clean", "non_performing_loans_to_equity_and_loan_loss_provision_clean"]]
    defensive_pair = [p("stable_prod_eps_uncertainty_score"), p("debt_quality_score")]

    df["banks_quality_composite"] = np.where(
        (_notnull_count(df, banks_p1_cols) > 0) & (_notnull_count(df, bank_alpha_raw) > 0)
        & (_notnull_count(df, bank_def_raw) > 0)
        & (_notnull_count(df, [p("stable_prof_score"), p("eps_uncertainty_clean")]) > 0),
        banks_comp, np.nan)
    df["financials_quality_composite"] = np.where(
        (_notnull_count(df, fin_alpha_cols) > 1) & (_notnull_count(df, defensive_pair) > 0), fin_comp, np.nan)
    df["non_financials_quality_composite"] = np.where(
        (_notnull_count(df, [p("profitability_score"), p("cashflow_quality_score"),
                             p("accruals_quality_score")]) > 1)
        & (_notnull_count(df, defensive_pair) > 0), nonfin_comp, np.nan)
    rerank(["banks_quality_composite", "financials_quality_composite", "non_financials_quality_composite"])

    g = df["industry_group"]
    comp = _pick_by_group(g, {1: df[p("banks_quality_composite")], 2: df[p("financials_quality_composite")],
                              3: df[p("non_financials_quality_composite")]})
    alpha = _pick_by_group(g, {1: df[p("banks_only_quality_alpha_score")],
                               2: df[p("financials_quality_alpha_score")],
                               3: df[p("non_financials_quality_alpha_score")]})
    defensive = _pick_by_group(g, {1: df[p("banks_only_quality_defensive_score")],
                                   2: df[p("financials_quality_defensive_score")],
                                   3: df[p("non_financials_quality_defensive_score")]})
    out = pd.DataFrame({"security_id": df["security_id"].to_numpy()})
    out["pct_rank_quality_composite"] = comp
    out["pct_rank_quality_alpha"] = np.where(np.isnan(comp), np.nan, alpha)
    out["pct_rank_quality_defensive"] = np.where(np.isnan(comp), np.nan, defensive)
    out["pct_rank_banks_quality_composite"] = np.where(g == 1, df[p("banks_quality_composite")], np.nan)
    out["pct_rank_financials_quality_composite"] = np.where(g == 2, df[p("financials_quality_composite")], np.nan)
    out["pct_rank_non_financials_quality_composite"] = np.where(
        g == 3, df[p("non_financials_quality_composite")], np.nan)
    return out


MOMENTUM_INPUTS = [f"zscore_{h}_mth_price_return_local_clean" for h in (1, 3, 6, 9, 12)]


def compute_momentum_composite(zs: pd.DataFrame, univ: pd.DataFrame) -> pd.DataFrame:
    """Replicates class zscore_momentum_composite (metrics.py 8494-8658).

    Inputs: zscore_{1,3,6,9,12}_mth_price_return_local_clean (outputs of zscore_pct_rank on the
    *_price_return_local_clean factors, population z-scores). Each is RE-z-scored across the universe with
    pandas (x - mean)/std (sample std, ddof=1, skipna) and clipped to [-3, 3]; average_zscore = mean of available
    clipped z-scores if >= 3 of 5 non-null else null; zscore_momentum_composite = sample z-score of
    average_zscore, clipped to [-3, 3]. No industry grouping. Cross-section = univ rows.
    """
    df = _base_frame(univ, zs, MOMENTUM_INPUTS)
    lim = []
    for f in MOMENTUM_INPUTS:
        c = f"zscore_limit_{f}"
        df[c] = _clip3(df.groupby([_DD])[f].transform(_zscore_sample))
        lim.append(c)
    df["average_zscore"] = np.where(_notnull_count(df, lim) > 2, df[lim].mean(axis=1), np.nan)
    comp = df.groupby([_DD])["average_zscore"].transform(_zscore_sample)
    return pd.DataFrame({"security_id": df["security_id"].to_numpy(),
                         "zscore_momentum_composite": _clip3(comp)})


def revisions_dates(run_date) -> List[_dt.date]:
    """[datadate, datadate - relativedelta(months=k, day=31) for k=1..5] (metrics.py 8668-8689)."""
    d = _cp_to_date(run_date)
    return [d] + [_month_end_back(d, k) for k in range(1, 6)]


def compute_revisions_4m_composite(univ: pd.DataFrame, get_estimates: Callable, run_date) -> pd.DataFrame:
    """Replicates class zscore_earnings_revisions_4_mth_composite (metrics.py 8661-8928).

    For each of 6 dates (run_date and the 5 previous month-ends, passed as-is – production calls
    Estimates.get(as_of_date=date) without Monday shift) EPS FY1 and FY2 estimates are INNER-joined to the
    universe on ibes id: a security missing an FY1 or FY2 record on ANY of the 6 dates (incl. the never-used
    4- and 5-month-ago dates) is dropped (-> null output). For k = 0..3:
      rev_k = null if greatest(n_FY1, n_FY2) < 2 (Spark greatest skips nulls) else
              (up_FY1 - down_FY1)/n_FY1*0.5 + (up_FY2 - down_FY2)/n_FY2*0.5   (x/0 -> null, Spark)
    Each rev_k z-scored over all joined rows (pandas, sample std), clipped [-3,3];
    raw = 0.4 z0 + 0.3 z1 + 0.2 z2 + 0.1 z3 (null if any null); composite = clip(zscore(raw), -3, 3).
    """
    cols = ["number_of_estimates_raised_over_four_weeks", "number_of_estimates_lowered_over_four_week",
            "number_of_current_estimates"]
    base = univ[["security_id", "ibes_id"]].copy()
    base = base[base["ibes_id"].notna()]  # Spark equi-join never matches a null key
    base["ibes_id"] = base["ibes_id"].astype(str)
    df = base
    for key, d in enumerate(revisions_dates(run_date)):
        est = get_estimates("eps", d, periods=("FY1", "FY2"))
        for fp in ("FY1", "FY2"):
            if est is None or len(est) == 0:
                e = pd.DataFrame(columns=["ibes_id"] + cols)
            else:
                e = est[est["forecast_period"].astype(str) == fp]
                e = e[e["ibes_id"].notna()][["ibes_id"] + cols].copy()
            e["ibes_id"] = e["ibes_id"].astype(str)
            for c in cols:
                e[c] = _cp_num(e[c])
            e = e.rename(columns={c: f"{c}_{fp}_{key}" for c in cols})
            df = df.merge(e, on="ibes_id", how="inner")

    def ratio(fp, k):
        n = df[f"number_of_current_estimates_{fp}_{k}"]
        up = df[f"number_of_estimates_raised_over_four_weeks_{fp}_{k}"]
        dn = df[f"number_of_estimates_lowered_over_four_week_{fp}_{k}"]
        return (up - dn) / n.where(n != 0)  # Spark: division by zero -> null

    for k in range(4):
        n1 = df[f"number_of_current_estimates_FY1_{k}"].to_numpy(dtype="float64")
        n2 = df[f"number_of_current_estimates_FY2_{k}"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore"):
            g = np.fmax(n1, n2)
            null_rule = g < 2
        val = ratio("FY1", k) * 0.5 + ratio("FY2", k) * 0.5
        df[f"earnings_revisions_1_mth_{k}"] = np.where(null_rule, np.nan, val)

    for k in range(4):
        f = f"earnings_revisions_1_mth_{k}"
        df[f"zscore_limit_{f}"] = _clip3(_zscore_sample(df[f].astype("float64")))
    raw = (0.4 * df["zscore_limit_earnings_revisions_1_mth_0"] + 0.3 * df["zscore_limit_earnings_revisions_1_mth_1"]
           + 0.2 * df["zscore_limit_earnings_revisions_1_mth_2"] + 0.1 * df["zscore_limit_earnings_revisions_1_mth_3"])
    df["zscore_earnings_revisions_4_mth_composite"] = _clip3(_zscore_sample(raw))
    # production has no dropDuplicates here; keep one row per security for the output convention
    res = df[["security_id", "zscore_earnings_revisions_4_mth_composite"]].drop_duplicates(subset=["security_id"])
    out = univ[["security_id"]].merge(res, on="security_id", how="left")
    return out.reset_index(drop=True)


QGARP_INPUTS = ["pct_rank_quality_alpha", "pct_rank_quality_defensive", "pct_rank_growth_composite",
                "pct_rank_value_composite", "zscore_earnings_revisions_1_mth_clean"]


def compute_qgarp(composites_df: pd.DataFrame, zs: pd.DataFrame, univ: pd.DataFrame,
                  fix_revisions_direction: bool = False) -> pd.DataFrame:
    """Replicates class pct_rank_qgarp (metrics.py 8931-9100).

    Each input re-ranked across the universe with rank(pct=True) ASCENDING (pandas default), ceil(*100):
    the four pct_rank_* inputs are 1 = best so ascending keeps 1 = best. PRODUCTION BUG replicated by default:
    zscore_earnings_revisions_1_mth_clean (higher = better) is also ranked ascending, so the WORST revisions get
    rank ~1 and are rewarded. ``fix_revisions_direction=True`` is a deliberate, optional fix (NOT production):
    ranks the revisions z-score descending so the best revisions get rank ~1.
    qgarp = -(0.2 r_alpha + 0.1 r_defensive + 0.3 r_growth + 0.3 r_value + 0.1 r_revisions), missing ranks -> 50.5.
    pct_rank_qgarp = ceil(rank(qgarp, pct, descending)*100): 1 = best. Every univ row gets a value (all-missing
    -> qgarp -50.5). zscore_qgarp (computed, unused in production) is not returned.
    composites_df: security_id + the 4 pct_rank_* columns; zs: security_id + zscore_earnings_revisions_1_mth_clean.
    """
    comp_cols = QGARP_INPUTS[:4]
    df = _base_frame(univ, composites_df, comp_cols)
    rev = _base_frame(univ, zs, ["zscore_earnings_revisions_1_mth_clean"])
    df["zscore_earnings_revisions_1_mth_clean"] = rev["zscore_earnings_revisions_1_mth_clean"].to_numpy()

    for f in QGARP_INPUTS:
        asc = not (fix_revisions_direction and f == "zscore_earnings_revisions_1_mth_clean")
        df[f"pct_rank_{f}"] = _pd_pct_rank(df, f, [_DD], ascending=asc)

    def filled(c):
        return np.where(pd.notnull(df[c]), df[c], 50.5)

    df["qgarp"] = -(0.2 * filled("pct_rank_pct_rank_quality_alpha")
                    + 0.1 * filled("pct_rank_pct_rank_quality_defensive")
                    + 0.3 * filled("pct_rank_pct_rank_growth_composite")
                    + 0.3 * filled("pct_rank_pct_rank_value_composite")
                    + 0.1 * filled("pct_rank_zscore_earnings_revisions_1_mth_clean"))
    return pd.DataFrame({"security_id": df["security_id"].to_numpy(),
                         "pct_rank_qgarp": _pd_pct_rank(df, "qgarp", [_DD])})


def compute_all_composites(clean: pd.DataFrame, zs: pd.DataFrame, univ: pd.DataFrame, get_estimates: Callable,
                           run_date, fix_revisions_direction: bool = False) -> pd.DataFrame:
    """All composites (metrics.py 6919-9100) merged on security_id, one row per univ security.

    clean: security_id + '<factor>_clean' inputs (value/growth/quality); zs: security_id +
    zscore_{1,3,6,9,12}_mth_price_return_local_clean + zscore_earnings_revisions_1_mth_clean.
    """
    out = univ[["security_id"]].drop_duplicates().reset_index(drop=True)
    parts = [
        compute_value_composite(clean, univ),
        compute_growth_composite(clean, univ),
        compute_quality_composite(clean, univ),
        compute_momentum_composite(zs, univ),
        compute_revisions_4m_composite(univ, get_estimates, run_date),
    ]
    for part in parts:
        out = out.merge(part.drop_duplicates(subset=["security_id"]), on="security_id", how="left")
    q = compute_qgarp(out, zs, univ, fix_revisions_direction=fix_revisions_direction)
    out = out.merge(q.drop_duplicates(subset=["security_id"]), on="security_id", how="left")
    return out

# COMMAND ----------

# MAGIC %md ## 6. Orchestration

# COMMAND ----------

# Orchestration used by the notebook (pure pandas; L is a QuantlakeLoader or a fake with the same methods)
import pandas as pd, numpy as np

EXCLUDED_FACTORS = {"beta": "Placeholder in production (beta = local share price); no market-return regression exists",
                    "eps_volatility": "Placeholder in production (copy of NTM EPS)",
                    "sales_volatility": "Placeholder in production (copy of NTM sales)"}

ZSCORE_INPUTS = [f"{h}_mth_price_return_local_clean" for h in (1, 3, 6, 9, 12)] + ["earnings_revisions_1_mth_clean"]


def run_pipeline(L, univ, run_date, fix_qgarp_revisions_direction=False, log=print):
    log("Worldscope factors ...")
    ws = compute_all_ws(univ, L.get_ws, run_date)
    log("Price factors ...")
    px = compute_price_metrics(univ, L.get_prices, run_date)
    log("I/B/E/S factors ...")
    ib = compute_all_ibes(univ, L.get_estimates, L.get_actuals, L.get_ws, run_date)
    raw = univ[["security_id"]].merge(ws, on="security_id", how="left") \
                               .merge(px, on="security_id", how="left") \
                               .merge(ib, on="security_id", how="left")
    dup = [c for c in raw.columns if c.endswith("_x") or c.endswith("_y")]
    assert not dup, f"duplicate factor columns across modules: {dup}"
    raw = raw.replace([np.inf, -np.inf], np.nan)  # Spark returns null for x/0; pandas gives inf
    log("Cleaning ...")
    rules = L.get_rules()
    clean = clean_factors(raw, rules, univ)
    log("Z-scores ...")
    zs = zscore_pct_rank(clean, ZSCORE_INPUTS, univ=univ)
    log("Composites ...")
    comp = compute_all_composites(clean, zs, univ, L.get_estimates, run_date,
                                  fix_revisions_direction=fix_qgarp_revisions_direction)
    return raw, clean, zs, comp


def to_long(univ, raw, clean, comp, run_date, factor_names):
    """One row per security x factor: raw value, clean value (composites: value itself)."""
    ids = univ[["security_id", "worldscope_id", "ibes_id", "security_name", "sedol", "isin"]]
    r = raw.melt(id_vars="security_id", var_name="factor", value_name="raw_value")
    c = clean.melt(id_vars="security_id", var_name="factor", value_name="clean_value")
    c["factor"] = c["factor"].str.replace(r"_clean$", "", regex=True)
    out = r.merge(c, on=["security_id", "factor"], how="outer")
    k = comp.melt(id_vars="security_id", var_name="factor", value_name="raw_value")
    k["clean_value"] = k["raw_value"]
    out = pd.concat([out, k], ignore_index=True)
    out = out[out["factor"].isin(factor_names)]
    out["datadate"] = pd.Timestamp(run_date)
    return ids.merge(out, on="security_id", how="right")

# COMMAND ----------

# Factor list from the GEMS Factor Construction Library (129 names; 3 placeholders excluded)
FACTOR_LIBRARY_NAMES = [
    "book_yield",
    "dividend_yield",
    "earnings_yield",
    "earnings_yield_ntm",
    "cash_flow_yield",
    "free_cash_flow_yield",
    "sales_yield_ntm",
    "ebitda_to_ev",
    "pct_rank_value_composite",
    "dividend_yield_ntm",
    "sales_to_ev",
    "sales_yield",
    "ebitda_to_price",
    "earnings_yield_fy1",
    "earnings_yield_fy2",
    "earnings_yield_fy3",
    "earnings_yield_3_year_hist",
    "earnings_yield_5_year_hist",
    "earnings_yield_ltm",
    "ebit_yield_ltm",
    "ebit_yield_ntm",
    "ebit_yield_fy1",
    "ebit_yield_fy2",
    "ebit_yield_fy3",
    "ebit_yield",
    "ebit_yield_3_year_hist",
    "ebit_yield_5_year_hist",
    "pct_rank_value_composite_fins_neutral",
    "return_on_equity",
    "net_income_margin",
    "debt_to_equity",
    "pct_rank_quality_composite",
    "pct_rank_quality_alpha",
    "pct_rank_quality_defensive",
    "return_on_invested_capital",
    "accruals",
    "return_on_assets_1_year_change",
    "return_on_assets",
    "ebit_margin",
    "current_ratio",
    "asset_turnover",
    "operating_margin",
    "quick_ratio",
    "eps_uncertainty",
    "free_cash_flow_conversion",
    "capex_to_depreciation",
    "short_term_debt_to_total_debt",
    "short_term_debt_to_total_assets",
    "interest_coverage",
    "cash_flow_from_financing_to_total_assets",
    "debt_to_assets",
    "operating_leverage",
    "extraordinary_charge_to_operating_income",
    "extraordinary_credit_to_operating_income",
    "unusual_items_to_operating_income",
    "price_return_stability",
    "return_on_equity_ntm",
    "cash_flow_conversion",
    "capital_adequacy_ratio_tier_1_capital",
    "capital_adequacy_ratio_total_capital",
    "net_income_to_risk_weighted_assets",
    "net_interest_income_to_net_loan",
    "non_interest_income_to_non_interest_expense",
    "accruals_year_on_year_change",
    "interest_income_to_interest_expense",
    "loan_loss_provisions_to_net_loans",
    "loans_to_deposits",
    "non_performing_loans_to_equity_and_loan_loss_provision",
    "return_on_equity_5_year_vol",
    "return_on_invested_capital_5_year_vol",
    "return_on_assets_5_year_vol",
    "net_income_margin_5_year_vol",
    "net_income_5_year_stability",
    "eps_growth_3_year_hist_cagr",
    "sales_growth_hist",
    "eps_growth_ltm_to_ntm",
    "earnings_revisions_1_mth",
    "zscore_earnings_revisions_4_mth_composite",
    "sustainable_growth",
    "pct_rank_growth_composite",
    "earnings_revisions_4_mth",
    "dividend_growth_hist",
    "eps_ntm_3mth_chg",
    "sales_growth_ltm_to_ntm",
    "sales_revisions_1_mth",
    "eps_growth_fy0_to_fy1",
    "eps_growth_fy1_to_fy2",
    "eps_growth_5_year_hist_cagr",
    "eps_growth_fy1_to_fy3",
    "eps_growth_fy2_to_fy3",
    "sales_growth_fy1_to_fy3",
    "earnings_revision_persistence_6mth",
    "cash_flow_growth_1_year_change",
    "cash_flow_growth_3_year_change",
    "cash_flow_growth_ltm_to_ntm",
    "operating_cash_flow_growth_1_year_change",
    "operating_cash_flow_growth_3_year_change",
    "sustainable_growth_ntm",
    "shareholder_yield_growth_hist",
    "shareholder_yield_growth_3_year_hist",
    "ebit_growth_fy1_to_fy2",
    "ebit_growth_fy2_to_fy3",
    "ebit_growth_fy1_to_fy3",
    "ebit_growth_ltm_to_ntm",
    "ebit_growth_3_year_hist_cagr",
    "ebit_growth_5_year_hist_cagr",
    "eps_growth_1_year_hist",
    "eps_ntm_1mth_chg",
    "sustainable_growth_3_year_average",
    "market_capitalization",
    "6_mth_price_return_local",
    "12_mth_price_return_local",
    "zscore_momentum_composite",
    "12_mth_price_return_usd",
    "payout_ratio",
    "1_year_volatility",
    "1_mth_price_return_usd",
    "6_mth_price_return_usd",
    "standardised_unexpected_earnings",
    "1_mth_price_return_local",
    "3_mth_price_return_local",
    "3_mth_price_return_usd",
    "9_mth_price_return_local",
    "9_mth_price_return_usd",
    "pct_rank_qgarp",
    "regression_momentum_1_year",
]

# COMMAND ----------

# MAGIC %md ## 7. Run
# MAGIC ### 7a. Universe, identifiers, prices and FX

# COMMAND ----------

L = QuantlakeLoader(spark, ql_catalog=QL_CATALOG, sage_catalog=SAGE_CATALOG, point_in_time=POINT_IN_TIME)
univ = L.build_universe(RUN_DATE)
print(f"Universe securities: {len(univ):,}")
summary = {
    "with Worldscope id": univ.worldscope_id.notna().sum(),
    "with I/B/E/S id": univ.ibes_id.notna().sum(),
    "with >1 Worldscope code (first kept)": (univ.n_ws_codes > 1).sum(),
    "with price on/before run date": univ.price.notna().sum(),
    "with USD price": univ.price_usd.notna().sum(),
    "with Worldscope FX rate": univ.worldscope_fx_rate.notna().sum(),
    "with I/B/E/S FX rate": univ.ibes_fx_rate.notna().sum(),
    "with industry classification": univ.general_industry_classification.notna().sum(),
}
display(pd.DataFrame({"count": summary}).assign(share=lambda d: (d["count"] / len(univ)).round(3)))
display(univ.general_industry_classification.value_counts(dropna=False).rename("securities").to_frame())

# COMMAND ----------

# MAGIC %md ### 7b. Data checks
# MAGIC Labels the loaders assume. If a check fails, adjust the loader constant named in the message before running 7c.

# COMMAND ----------

labels = spark.sql("""
    select 'estimates' src, forecast_period, fiscal_period, count(*) n from gems_est group by 1, 2, 3
    union all
    select 'actuals', forecast_period, fiscal_period, count(*) from {ql}.marketdata.analyst_estimates_actuals_surprises
    where as_of_date >= date_sub('{d}', 400) group by 1, 2, 3
""".format(ql=QL_CATALOG, d=RUN_DATE)).toPandas()
display(labels)
assert {"FY1", "FY2"} <= set(labels.loc[labels.src == "estimates", "forecast_period"]), \
    "I/B/E/S forecast_period labels are not 'FY1'/'FY2' - adjust get_estimates()"
act_fp = set(labels.loc[labels.src == "actuals", "fiscal_period"].astype(str).str.lower())
assert act_fp & set(L.annual_ibes_fiscal_periods), \
    f"Annual label for FY0 actuals not found in {act_fp} - set L.annual_ibes_fiscal_periods"

# Spot check: TSMC (Worldscope C76073140) market cap, EV, shares, TTM sales at the run date
tsmc = L.get_ws(["market_capitalization", "enterprise_value", "common_shares_outstanding",
                 "trailing_twelve_months_net_sales_revenue"], RUN_DATE)
display(tsmc[tsmc.worldscope_id == "C76073140"])

# COMMAND ----------

# MAGIC %md ### 7c. Build all factors

# COMMAND ----------

raw, clean, zs, comp = run_pipeline(L, univ, RUN_DATE, fix_qgarp_revisions_direction=FIX_QGARP)
FACTOR_NAMES = FACTOR_LIBRARY_NAMES
result = to_long(univ, raw, clean, comp, RUN_DATE, FACTOR_NAMES)
coverage = (result.groupby("factor")
            .agg(securities=("security_id", "nunique"),
                 raw_non_null=("raw_value", lambda s: int(s.notna().sum())),
                 clean_non_null=("clean_value", lambda s: int(s.notna().sum())))
            .assign(clean_coverage=lambda d: (d.clean_non_null / len(univ)).round(3))
            .reset_index())
print(f"Factors built: {coverage.factor.nunique()} (excluded: {', '.join(EXCLUDED_FACTORS)})")
display(coverage.sort_values("clean_coverage"))

# COMMAND ----------

# MAGIC %md ## 8. Reconciliation with production
# MAGIC Compares raw and clean values with `reporting.qwd_position_level` for the run date (securities held in a portfolio or benchmark on that date only).
# MAGIC Tolerance: relative 1e-6 or absolute 1e-9.

# COMMAND ----------

prod = L.get_production_values(RUN_DATE, FACTOR_NAMES)
cmp = result.merge(prod.rename(columns={"metric_name": "factor"}), on=["security_id", "factor"], how="inner")
def _match(a, b):
    both_null = a.isna() & b.isna()
    close = (a - b).abs() <= np.maximum(1e-9, 1e-6 * b.abs())
    return both_null | close
cmp["raw_match"] = _match(cmp.raw_value, cmp.prod_raw_value)
cmp["clean_match"] = _match(cmp.clean_value, cmp.prod_clean_value)
recon = (cmp.groupby("factor")
         .agg(compared=("security_id", "count"),
              raw_match_rate=("raw_match", "mean"),
              clean_match_rate=("clean_match", "mean"),
              median_abs_diff=("raw_value", lambda s: float(np.nanmedian(np.abs(s - cmp.loc[s.index, "prod_raw_value"]))) if s.notna().any() else np.nan))
         .reset_index().sort_values("raw_match_rate"))
print(f"Securities compared: {cmp.security_id.nunique():,}; factors compared: {recon.factor.nunique()}")
display(recon)

# COMMAND ----------

recon.factor.iloc[113]

# COMMAND ----------

# Worst mismatches for one factor (change the name to drill down)
FACTOR_TO_INSPECT = recon.factor.iloc[113] if len(recon) else None
if FACTOR_TO_INSPECT:
    d = cmp[(cmp.factor == FACTOR_TO_INSPECT) & ~cmp.raw_match].copy()
    d["abs_diff"] = (d.raw_value - d.prod_raw_value).abs()
    display(d.sort_values("abs_diff", ascending=False)[["security_id", "security_name", "worldscope_id", "ibes_id",
                                                        "raw_value", "prod_raw_value", "clean_value", "prod_clean_value"]].head(50))

# COMMAND ----------

# MAGIC %md ## 9. Save (optional)

# COMMAND ----------

if OUTPUT_TABLE:
    (spark.createDataFrame(result.astype({"worldscope_id": "object", "ibes_id": "object"}))
          .write.mode("overwrite").option("replaceWhere", f"datadate = '{RUN_DATE}'")
          .saveAsTable(OUTPUT_TABLE))
    print(f"Written {len(result):,} rows to {OUTPUT_TABLE}")
else:
    print("No output table set - results are in the `result` DataFrame")