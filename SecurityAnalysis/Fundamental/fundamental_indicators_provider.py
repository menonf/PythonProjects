import fundamentalanalysis as fa

ticker = "AAPL"
api_key = "eaa83187bd18f51cf2c84830f00e89ae"

# Show the available companies
companies = fa.available_companies(api_key)

# Collect general company information
profile = fa.profile(ticker, api_key)

# Collect recent company quotes
quotes = fa.quote(ticker, api_key)

# Collect market cap and enterprise value
entreprise_value = fa.enterprise(ticker, api_key)

# Show recommendations of Analysts
ratings = fa.rating(ticker, api_key)

# Obtain DCFs over time
dcf_annually = fa.discounted_cash_flow(ticker, api_key, period="annual")


# Collect the Balance Sheet statements
balance_sheet_annually = fa.balance_sheet_statement(ticker, api_key, period="annual")


# Collect the Income Statements
income_statement_annually = fa.income_statement(ticker, api_key, period="annual")


# Collect the Cash Flow Statements
cash_flow_statement_annually = fa.cash_flow_statement(ticker, api_key, period="annual")


# Show Key Metrics
key_metrics_annually = fa.key_metrics(ticker, api_key, period="annual")


# Show a large set of in-depth ratios
financial_ratios_annually = fa.financial_ratios(ticker, api_key, period="annual")


# Show the growth of the company
growth_annually = fa.financial_statement_growth(ticker, api_key, period="annual")


# Download general stock data
stock_data = fa.stock_data(ticker, period="ytd", interval="1d")

# Download detailed stock data
stock_data_detailed = fa.stock_data_detailed(ticker, api_key, begin="2018-01-01", end="2021-10-10")
print(stock_data_detailed)