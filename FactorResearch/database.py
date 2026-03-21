"""SQLAlchemy ORM Module to connect to database objects."""

import datetime
import re
import time
from datetime import date
from typing import List, Optional, Tuple, Type
from urllib import parse

import keyring
import pandas as pd
import sqlalchemy as sql
from pandas import DataFrame
from sqlalchemy import Engine, delete, update
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# ---------------------------------------------------------------------------
# ORM Base & Models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """SQLAlchemy Base Class."""
    pass


class SecurityMaster(Base):
    """Maps to dbo.security_master table."""

    __tablename__ = "security_master"
    __table_args__ = {"schema": "dbo"}

    security_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column()
    name: Mapped[str] = mapped_column()
    isin: Mapped[str] = mapped_column()
    sedol: Mapped[str] = mapped_column()
    cusip: Mapped[str] = mapped_column()
    figi: Mapped[str] = mapped_column()
    loanxid: Mapped[str] = mapped_column()
    country: Mapped[str] = mapped_column()
    currency: Mapped[str] = mapped_column()
    sector: Mapped[str] = mapped_column()
    industry_group: Mapped[str] = mapped_column()
    industry: Mapped[str] = mapped_column()
    security_type: Mapped[str] = mapped_column()
    asset_class: Mapped[str] = mapped_column()
    exchange: Mapped[str] = mapped_column()
    is_active: Mapped[str] = mapped_column()
    source_vendor: Mapped[str] = mapped_column()
    upsert_date: Mapped[str] = mapped_column()
    upsert_by: Mapped[str] = mapped_column()


class SecurityFundamentals(Base):
    """Maps to dbo.security_fundamentals table."""

    __tablename__ = "security_fundamentals"
    __table_args__ = {"schema": "dbo"}
    __mapper_args__ = {"primary_key": ["security_id", "metric_type", "effective_date", "source_vendor"]}

    security_id: Mapped[int] = mapped_column()
    metric_type: Mapped[str] = mapped_column()
    metric_value: Mapped[float] = mapped_column()
    source_vendor: Mapped[str] = mapped_column()
    effective_date: Mapped[datetime.date] = mapped_column()
    end_date: Mapped[datetime.date] = mapped_column(nullable=True)


class MarketData(Base):
    """Maps to dbo.market_data table."""

    __tablename__ = "market_data"
    __table_args__ = {"schema": "dbo"}

    md_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[str] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    open: Mapped[float] = mapped_column()
    high: Mapped[float] = mapped_column()
    low: Mapped[float] = mapped_column()
    close: Mapped[float] = mapped_column()
    adj_close: Mapped[float] = mapped_column()
    volume: Mapped[int] = mapped_column()
    dividends: Mapped[float] = mapped_column()
    stock_splits: Mapped[float] = mapped_column()
    interval: Mapped[str] = mapped_column()
    dataload_date: Mapped[str] = mapped_column()


class Portfolio(Base):
    """Maps to dbo.portfolio table."""

    __tablename__ = "portfolio"
    __table_args__ = {"schema": "dbo"}

    port_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_short_name: Mapped[str] = mapped_column()
    portfolio_name: Mapped[str] = mapped_column()
    portfolio_type: Mapped[str] = mapped_column()
    is_active: Mapped[str] = mapped_column()


class PortfolioHoldings(Base):
    """Maps to dbo.portfolio_holdings table."""

    __tablename__ = "portfolio_holdings"
    __table_args__ = {"schema": "dbo"}

    ph_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[str] = mapped_column()
    port_id: Mapped[int] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    held_shares: Mapped[float] = mapped_column()
    upsert_date: Mapped[str] = mapped_column()
    upsert_by: Mapped[str] = mapped_column()


class IndexConstituents(Base):
    """Maps to reference.index_constituents table."""

    __tablename__ = "index_constituents"
    __table_args__ = {"schema": "reference"}

    constituent_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    index_id: Mapped[int] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    exchange_ticker: Mapped[str] = mapped_column()
    start_date: Mapped[str] = mapped_column()
    end_date: Mapped[str] = mapped_column(nullable=True)
    source_vendor: Mapped[str] = mapped_column()
    upsert_date: Mapped[str] = mapped_column()
    upsert_by: Mapped[str] = mapped_column()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> str:
    """Parse and reformat a date string."""
    return datetime.datetime.strptime(date_str, fmt).strftime(fmt)


def _execute_with_session(orm_session: Session, operation, *args, **kwargs) -> None:
    """
    Execute a database operation with standardised error handling.

    Commits on success, rolls back on failure, and always closes the session.
    """
    try:
        operation(*args, **kwargs)
        orm_session.commit()
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        orm_session.rollback()
    except Exception as e:
        print(f"Unexpected error: {e}")
        orm_session.rollback()
    finally:
        orm_session.close()


def _read_table(orm_session: Session, orm_engine: Engine, model: Type[Base]) -> DataFrame:
    """Fetch all columns from an ORM-mapped table."""
    query = orm_session.query(*model.__table__.columns)
    return pd.read_sql_query(query.statement, con=orm_engine)


def _bulk_delete_insert(
    orm_session: Session,
    model: Type[Base],
    data_list: List[dict],
    *filter_criteria,
) -> None:
    """Delete matching rows and bulk-insert new records."""
    orm_session.query(model).filter(*filter_criteria).delete(synchronize_session=False)
    orm_session.bulk_insert_mappings(model, data_list)  # type: ignore


def _camel_to_snake(name: str) -> str:
    """Convert a CamelCase string to snake_case."""
    return re.sub("([A-Z])", r"_\1", name).lower().lstrip("_")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_db_connection(
    service_name: str = "ihub_sql_connection",
    server: str = "ops-store-server.database.windows.net",
    driver: str = "ODBC Driver 18 for SQL Server",
    max_retries: int = 3,
    retry_interval_minutes: int = 2,
) -> Tuple[sql.Engine, sql.Connection, str, Session]:
    """
    Establish a connection to SQL Server with retry logic.

    Returns:
        Tuple of (engine, connection, connection_string, session).
    """
    db = keyring.get_password(service_name, "db")
    db_user = keyring.get_password(service_name, "uid")
    db_password = keyring.get_password(service_name, "pwd")

    connection_string = (
        f"mssql+pyodbc://{db_user}:{db_password}"
        f"@{server}:1433/{db}"
        f"?driver={parse.quote_plus(driver)}&Encrypt=yes&TrustServerCertificate=no&autocommit=true"
    )

    for attempt in range(1, max_retries + 1):
        try:
            engine = sql.create_engine(connection_string)
            connection = engine.connect()
            session = Session(engine)
            print("Database connection successful.")
            return engine, connection, connection_string, session
        except OperationalError as e:
            print(f"Attempt {attempt} failed with error:\n{e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_interval_minutes} minutes...")
                time.sleep(retry_interval_minutes * 60)
            else:
                print("All retry attempts failed. Exiting.")
                raise

    raise RuntimeError("Database connection failed: maximum retries exceeded")


# ---------------------------------------------------------------------------
# Security Master
# ---------------------------------------------------------------------------


def read_security_master(orm_session: Session, orm_engine: Engine) -> DataFrame:
    """Fetch all records from security_master table."""
    query = orm_session.query(SecurityMaster)
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_security_master(equities_df: DataFrame, orm_session: Session) -> None:
    """Insert records into security_master table."""
    def _insert(data):
        orm_session.bulk_insert_mappings(SecurityMaster, data)  # type: ignore

    _execute_with_session(orm_session, _insert, equities_df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def read_portfolio(
    orm_session: Session,
    orm_engine: Engine,
    portfolio_short_names: List[str],
) -> DataFrame:
    """Fetch records from portfolio table filtered by short names."""
    query = orm_session.query(
        Portfolio.port_id,
        Portfolio.portfolio_short_name,
        Portfolio.portfolio_name,
        Portfolio.portfolio_type,
    ).filter(Portfolio.portfolio_short_name.in_(portfolio_short_names))
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_portfolio_holdings(df_holdings: DataFrame, orm_session: Session) -> None:
    """
    Upsert records into portfolio_holdings table.

    Deletes existing records for the same portfolio and date before inserting.
    """
    if df_holdings.empty:
        print("No data to write.")
        return

    as_of_date = df_holdings["as_of_date"].iloc[0]
    port_ids = df_holdings["port_id"].unique().tolist()

    df_holdings = df_holdings.copy()
    df_holdings["upsert_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_holdings["upsert_by"] = "daily_portfolio_load.py"

    def _upsert(data_list):
        _bulk_delete_insert(
            orm_session,
            PortfolioHoldings,
            data_list,
            PortfolioHoldings.as_of_date == as_of_date,
            PortfolioHoldings.port_id.in_(port_ids),
        )

    _execute_with_session(orm_session, _upsert, df_holdings.to_dict(orient="records"))


def read_portfolio_holdings(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Fetch records from portfolio_holdings table within a date range."""
    start_date, end_date = _parse_date(start_date), _parse_date(end_date)

    query = orm_session.query(
        PortfolioHoldings.as_of_date,
        PortfolioHoldings.port_id,
        PortfolioHoldings.security_id,
        PortfolioHoldings.held_shares,
    ).filter(PortfolioHoldings.as_of_date.between(start_date, end_date))

    return pd.read_sql_query(query.statement, con=orm_engine)


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------


def read_market_data(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Fetch records from market_data table within a date range."""
    start_date, end_date = _parse_date(start_date), _parse_date(end_date)

    query = orm_session.query(*MarketData.__table__.columns).filter(
        MarketData.as_of_date.between(start_date, end_date)
    )
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_market_data(market_data: DataFrame, orm_session: Session) -> None:
    """
    Upsert records into market_data table.

    Deletes existing records for the same security and date before inserting.
    """
    as_of_dates = market_data["as_of_date"].unique().tolist()
    security_ids = market_data["security_id"].unique().tolist()

    def _upsert(data_list):
        _bulk_delete_insert(
            orm_session,
            MarketData,
            data_list,
            MarketData.as_of_date.in_(as_of_dates),
            MarketData.security_id.in_(security_ids),
        )

    _execute_with_session(orm_session, _upsert, market_data.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Index Constituents
# ---------------------------------------------------------------------------


def read_index_constituents(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Fetch all records from index_constituents table."""
    _parse_date(start_date)  # validate inputs
    _parse_date(end_date)
    return _read_table(orm_session, orm_engine, IndexConstituents)


def write_index_constituents(index_constituents: DataFrame, orm_session: Session) -> None:
    """
    Upsert records into index_constituents table.

    Deletes existing records for the same index before inserting.
    """
    index_ids = index_constituents["index_id"].unique().tolist()

    def _upsert(data_list):
        _bulk_delete_insert(
            orm_session,
            IndexConstituents,
            data_list,
            IndexConstituents.index_id.in_(index_ids),
        )

    _execute_with_session(orm_session, _upsert, index_constituents.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Security Fundamentals
# ---------------------------------------------------------------------------


def read_security_fundamentals(
    orm_session: Session,
    orm_engine: Engine,
    metric_type: Optional[str] = None,
) -> DataFrame:
    """
    Fetch records from security_fundamentals table.

    Args:
        metric_type: If provided, filters by metric_type and renames
                     metric_value column to the snake_case metric name.
    """
    query = orm_session.query(*SecurityFundamentals.__table__.columns)
    if metric_type:
        query = query.filter(SecurityFundamentals.metric_type == metric_type)

    df = pd.read_sql_query(query.statement, con=orm_engine)

    if metric_type and "metric_value" in df.columns:
        df.rename(columns={"metric_value": _camel_to_snake(metric_type)}, inplace=True)

    return df


def write_security_fundamentals(fundamental_data: DataFrame, orm_session: Session) -> None:
    """
    Write security fundamentals with upsert/versioning logic.

    - **New data**: Inserts records when no existing match is found.
    - **Same count**: Overwrites existing active records.
    - **Different count**: Closes existing records (sets end_date) and inserts new ones.

    Args:
        fundamental_data: DataFrame with columns: security_id, metric_type,
            metric_value, source_vendor, effective_date, and optionally end_date.
        orm_session: SQLAlchemy Session object.
    """
    if fundamental_data.empty:
        print("No fundamental data to insert.")
        return

    source_vendor = fundamental_data["source_vendor"].iloc[0]
    effective_date = fundamental_data["effective_date"].iloc[0]
    security_ids = fundamental_data["security_id"].unique().tolist()
    metric_types = fundamental_data["metric_type"].unique().tolist()

    # Common WHERE filters for active records
    active_record_filters = [
        SecurityFundamentals.security_id.in_(security_ids),
        SecurityFundamentals.metric_type.in_(metric_types),
        SecurityFundamentals.source_vendor == source_vendor,
        SecurityFundamentals.effective_date == effective_date,
        SecurityFundamentals.end_date.is_(None),
    ]

    existing_count = (
        orm_session.query(SecurityFundamentals).filter(*active_record_filters).count()
    )
    new_count = len(fundamental_data)

    data_list = fundamental_data.to_dict(orient="records")
    for record in data_list:
        record["end_date"] = None

    def _write(data_list):
        if existing_count == 0:
            print(f"No existing records found. Inserting {new_count} new records.")
            orm_session.bulk_insert_mappings(SecurityFundamentals, data_list)  # type: ignore

        elif existing_count == new_count:
            print(f"Record counts match ({existing_count}). Overwriting existing records.")
            orm_session.execute(
                delete(SecurityFundamentals).where(*active_record_filters)
            )
            orm_session.bulk_insert_mappings(SecurityFundamentals, data_list)  # type: ignore

        else:
            print(f"Record counts differ (existing: {existing_count}, new: {new_count}). Versioning data.")
            orm_session.execute(
                update(SecurityFundamentals)
                .where(*active_record_filters)
                .values(end_date=date.today())
            )
            orm_session.bulk_insert_mappings(SecurityFundamentals, data_list)  # type: ignore

        print("Security fundamentals data successfully written.")

    _execute_with_session(orm_session, _write, data_list)


# ---------------------------------------------------------------------------
# Composite Queries
# ---------------------------------------------------------------------------


def get_portfolio_market_data(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
    portfolio_short_names: List[str],
) -> DataFrame:
    """
    Join portfolio, holdings, security master, market data, and fundamentals.

    Args:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy Engine object.
        start_date: Start date string in "YYYY-MM-DD" format.
        end_date: End date string in "YYYY-MM-DD" format.
        portfolio_short_names: List of portfolio short names to filter by.

    Returns:
        Merged DataFrame containing portfolio market data with fundamentals.
    """
    df_securities = read_security_master(orm_session, orm_engine)
    df_market_data = read_market_data(orm_session, orm_engine, start_date, end_date)
    df_portfolio = read_portfolio(orm_session, orm_engine, portfolio_short_names)
    df_holdings = read_portfolio_holdings(orm_session, orm_engine, start_date, end_date)
    df_fundamentals = read_security_fundamentals(orm_session, orm_engine, "shares_outstanding")

    # Merge portfolio → holdings → securities → market data
    df_portfolio_market_data = (
        df_portfolio
        .merge(df_holdings, on="port_id")
        .merge(df_securities, on="security_id")
        .merge(df_market_data, on=["security_id", "as_of_date"])
    )

    # Coerce dates and drop invalid rows
    df_portfolio_market_data["as_of_date"] = pd.to_datetime(
        df_portfolio_market_data["as_of_date"], errors="coerce"
    )
    df_fundamentals["effective_date"] = pd.to_datetime(
        df_fundamentals["effective_date"], errors="coerce"
    )
    df_portfolio_market_data.dropna(subset=["as_of_date"], inplace=True)
    df_fundamentals.dropna(subset=["effective_date"], inplace=True)

    # Align dtypes for join key
    df_portfolio_market_data["security_id"] = df_portfolio_market_data["security_id"].astype(int)
    df_fundamentals["security_id"] = df_fundamentals["security_id"].astype(int)

    # Group-wise backward merge_asof to align fundamentals to market dates
    merged_rows = []
    fundamental_cols = set(df_fundamentals.columns) - set(df_portfolio_market_data.columns)

    for sec_id, df_left in df_portfolio_market_data.groupby("security_id"):
        df_left = df_left.sort_values("as_of_date").reset_index(drop=True)
        df_right = df_fundamentals[df_fundamentals["security_id"] == sec_id].sort_values("effective_date").reset_index(drop=True)

        if not df_right.empty:
            df_merged = pd.merge_asof(
                df_left,
                df_right,
                by="security_id",
                left_on="as_of_date",
                right_on="effective_date",
                direction="backward",
            )
        else:
            df_merged = df_left.copy()
            for col in fundamental_cols:
                df_merged[col] = pd.NA

        merged_rows.append(df_merged)

    return pd.concat(
        [df.dropna(axis=1, how="all") for df in merged_rows if not df.empty],
        ignore_index=True,
    )
