# Improving the investment strategies: suggestions

Review of notebooks 3.1 to 3.6 (equity, investment strategies) and the code behind them.
Prepared 20 November 2026.

## How to read this

**Basis.** I read the six notebooks with their saved outputs, and the scoring, portfolio, backtest, data and risk code they call. I also read notebooks 4.1, 4.2, 5.5 and 5.6 where their results bear on section 3.

**Limit.** The research data is not in the zip, so no backtest was rerun. Nothing below is a new backtest result.

**Evidence labels.** Each point says where it comes from.

| Label | Meaning |
|---|---|
| Output | Read from your saved notebook outputs |
| Calculation | Worked out by me from the yearly tables in your notebooks |
| Test | Your own scoring and portfolio functions, run by me on simulated data |
| To verify | A strong indication that needs your data to confirm |

## Summary

The scores carry some real information. The portfolios lose most of it to sector, size and beta bets, and several issues distort the numbers being tuned against. The suggested order is: fix the measurements, change how the books are built, then prune the signals.

| # | Suggestion | Priority | Effect on existing results |
|---|---|---|---|
| 1.1 | Verify that the analyst revision counts are a true history | Critical | Changes QGARP results if confirmed |
| 1.2 | Compare learned value only over dates when its score exists | Critical | Changes notebook 3.3 |
| 1.3 | Make the factor mix genuinely equal-weighted | High | Changes the multi-factor runs |
| 1.4 | Benchmark against your own cap-weighted index | High | Changes active measures |
| 1.5 | Trace the one-day spike in mid-2020 | Medium | May change risk figures |
| 1.6 | Keep the write-ups in step with the outputs | Medium | None |
| 2.1 | Hold beta at 1 and judge on alpha | High | New trial |
| 2.2 | Remove the sector bet | High | New trial |
| 2.3 | Remove the size bet | High | New trial |
| 2.4 | Build the books with the risk model you already have | High | New trial |
| 2.5 | Use the bottom of the ranking | High | New trial |
| 2.6 | Spend active risk in mid and small caps | Medium | New trial |
| 2.7 | Cap single-name weights | Medium | New trial |
| 3 | Prune and reweight the signals | Medium | New trials |
| 4 | Rebuild the long-short book | Medium | New trials |
| 5 | Cut turnover | Medium | New trials |
| 6 | Test one written specification, then replicate on UK data | High | None |

Items 1.1 to 1.5 move registered numbers. Your regression check will flag them, so retake its baseline afterwards.

---

## 1. Fix first: these change the results

### 1.1 Verify that the analyst revision counts are a true history

**What the outputs show**

| Measure (notebook 3.6 triage, S&P 1500) | Month-to-month rank autocorrelation | Turnover of the top fifth |
|---|---|---|
| Revision breadth from analyst counts, 1 month | 1.00 | 5% |
| Revision breadth from analyst counts, 4 months | 1.00 | 3% |
| Consensus revision, 1 month | 0.25 | 70% |
| Change in next-twelve-month EPS, 1 month | 0.29 | 61% |

- Output: the 4-month count measure's top decile beat the universe by 7.5% a year with a t-statistic of 6.7 (S&P 1500), and by 6.0% with 4.3 (S&P 500).
- Output: the notebooks record that a delisted company answers "0 raised and 0 lowered".

**Why it matters**

- A monthly revision count cannot keep almost the same ranking month after month. The two genuine monthly measures in the table show what a real one looks like.
- The pattern is what you would see if the vendor returned today's counts for every past date.
- To verify: if so, this is look-ahead, not survivorship bias. Stocks would be ranked in 2004 on their 2026 revisions.
- At least one user on LSEG's developer forum lists these two fields among those that cannot be given a historical date: https://community.developers.lseg.com/discussion/51727/where-to-get-historical-data-for-eikon-fields-not-supporting-sdate/p1. Treat that as a reason to check, not as proof.

**Suggestions**

- Take a handful of large companies and read their stored raised and lowered counts across month ends. If they barely change from 2010 to 2026, the history is a snapshot.
- If confirmed, remove the counts from every backtest and triage. Keep them for the live screen only, where today's counts are real.
- Use the consensus revisions for all history. The QGARP results to quote are then the consensus-revision runs:

| QGARP run | With analyst counts | With consensus revisions | With no revisions |
|---|---|---|---|
| S&P 1500, active return against SPTM | +0.2% a year | -0.1% a year | -0.3% a year |
| S&P 500, active return against SPY | -0.8% a year | -1.5% a year | -1.2% a year |

- Re-read the findings that rest on the counts:
  - "correcting the revisions direction is worth about 0.9 points a year"
  - the four strongest entries of the S&P 1500 triage
  - three of the four discoveries on the S&P 500
  - QGARP's own ranking strength, which includes the counts at a 10% weight

### 1.2 Compare learned value only over dates when its score exists

**What the outputs show**

- Output (3.3): the learned-value-only book trailed SPTM by 8.8, 6.7 and 14.8 points in 2004, 2005 and 2006.
- Output (3.6): SPTM returned +8.8%, +6.7% and +14.8% in those years. The book therefore returned 0%.
- Output (3.3 chart): its line is flat at $100 until early 2007.

**Why it happens**

- The model needs eight quarter-end training dates whose one-year outcome is complete. Its first score comes about three years after its data starts: 2007 for data from 2004.
- With no score, no names are selected, and the backtest holds cash without saying so.

**Effect**

- Calculation: SPTM gained 33.3% over those three years. That is about 1.4 points a year of CAGR over the 21-year test.
- Output: the reported gap between learned value (8.3% a year) and plain value (10.0%) is 1.7 points, so most of it is the cash period.
- Output: the write-up in 3.3 still says the learned score beats formula value.

**Suggestions**

- Compare the two scores over the same dates, starting when the learned score first exists.
- Or lengthen the training history, now that the database reaches back to 2000, so that the score exists from 2004.
- Make a run stop, or at least warn, when its book is empty on the first rebalance dates. The same check protects any other score with a late start.

### 1.3 Make the factor mix genuinely equal-weighted

**What the code does**

- Value and quality are each the average of four z-scores. That average is not standardised again.
- Momentum, low volatility and net issuance each enter as a single z-score.
- A missing factor score is ignored, so a name is scored on whatever it has.

**Effect**

- Test: value and quality have 0.5 to 0.8 of the spread of the single-score factors, depending on how correlated their four ratios are.
- Test: they therefore carry roughly 25% to 35% of the five-factor score instead of 40%, and 33% to 44% of the four-factor score instead of 50%.
- Test: a name with two of five scores is about twice as likely to reach the top decile, 20% against 9%. The same holds at the bottom.

**Why it matters**

- Output (3.2): the factors left overweight include momentum and low volatility, whose top deciles lagged the universe by 1.2% and 1.7% a year.
- Output (3.2): value (+2.6% a year) and net issuance (+2.1%) had the best top deciles of the five.

**Suggestions**

- Standardise each factor's composite again before mixing.
- Count a missing factor as neutral, or require a minimum number of factors before a name is scored.
- Check score coverage by sector. Financials are the likely case: several of the ratios cannot be formed for banks, and the first version's book is 37.4% financials.
- Give the EBIT-to-enterprise-value ratio the same protection against a zero or negative enterprise value that the free-cash-flow yield already has.

### 1.4 Benchmark against your own cap-weighted index

**Facts**

- SPTM launched on 4 October 2000.
- It tracked the Dow Jones U.S. Total Stock Market Index until July 2013, the Russell 3000 until November 2017 and an SSGA total-market index until 24 January 2020. Only since then has it tracked the S&P Composite 1500.
- Source: SSGA fact sheet, https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-sptm.pdf
- Your own code notes that SPTM's daily correlation with SPY was 0.63 in 2008, a sign of stale prices in the fund's early years.
- Output (3.1): the headline CAGRs in the first table are not over identical dates. The strategy starts in early 2000, and SPTM cannot start before its launch. The like-for-like figure is the +1.8% a year active return further down.

**Suggestions**

- Build the cap-weighted S&P 1500 and S&P 500 from your own constituents and use them as the primary benchmarks. The index-replication notebooks already have the function.
- Report equally weighted books against the equal-weight index as well.
- Keep SPTM and SPY as investable cross-checks.

### 1.5 Trace the one-day spike in mid-2020

- Output (charts in 3.4 and 3.1): a drop of roughly 10 points or more reverses the next day, in both the long-short curve and the long-only relative chart.
- A diversified book does not move like that. It looks like a price or corporate-action error on one day.
- It inflates volatility and drawdown. The long-short book's 2020 trough of about -48% is that spike.

**Suggestions**

- Find the date and list the names with the largest one-day moves.
- Correct or exclude the bad prices.
- Add a data check for large moves that fully reverse the next day.

### 1.6 Keep the write-ups in step with the outputs

Output: several findings no longer match the tables above them.

| Notebook | The text says | The output shows |
|---|---|---|
| 3.1 | Runs labelled "2010-2024" | The runs start in 2000 |
| 3.3 | The learned score beats formula value (13.5% against 11.2% a year) | Learned value 8.3%, plain value 10.0% |
| 3.5, part 1 | Rebuild from June 2009: 13.2% against SPY's 14.9% | Rebuild from June 2004: 10.8% against 10.9% |
| 3.6 | 2010 to 2024: 12.9% against SPTM's 14.0%; 14 discoveries | 2004 to 2026: 11.0% against 10.7%; 26 discoveries |

**Suggestions**

- Have each findings section read its numbers from the results instead of typed text.
- Show alpha beside active return in every summary table. The code already computes it; the multi-factor notebooks do not display it.

---

## 2. Portfolio construction: the biggest lever

### 2.1 Hold beta at 1 and judge on alpha

Output, except where marked:

| Book | Beta | Active return a year | Alpha a year |
|---|---|---|---|
| Multi-factor, second version, 2004 to 2024 | 0.85 | +0.2% | about +1.5% to +2% (Calculation, approximate) |
| QGARP, S&P 1500 | 1.13 | +0.2% | -0.7% |
| QGARP, S&P 500 | 1.05 | -0.8% | -1.2% |
| S&P GARP rebuild | 1.12 | -0.1% | -0.9% |

- The same +0.2% active return hides a positive alpha in one book and a negative alpha in the other.
- The low-volatility tilt pulls the multi-factor book's beta to 0.85. In a rising market that gives back what selection earns.

**Suggestions**

- Target a predicted beta of 1 when building long-only books.
- Make alpha and the beta-adjusted information ratio the headline measures.

### 2.2 Remove the sector bet

- Output (3.1 chart): the first version's book at 31 December 2024 is 37.4% financials and 3.9% information technology, by number of holdings.
- For scale: information technology was about 38% of SPTM by weight and financials about 12% at 30 September 2026 (SSGA fact sheet above; a different date and basis).
- Scores are ranked across the whole universe, so a cheap sector fills the book.

**Suggestions**

- Rank value and quality within sectors.
- Or cap each sector's active weight at a few points.

### 2.3 Remove the size bet

**QGARP on the S&P 1500 (equally weighted)**

- Calculation: its yearly active return against SPTM has a 0.84 correlation with the equal-weight-minus-SPTM spread, 2005 to 2024.
- Calculation: the volatility of its yearly active return is 7.6% against SPTM and 4.3% against the equal-weight S&P 1500.
- Calculation: over 2005 to 2024 QGARP returned 10.0% a year, SPTM 10.3% and the equal-weight index 9.8%. The equal-weight index pays no trading costs here; QGARP is net of them.
- So most of its good and bad years are the weighting, not the score. The exception is 2020, 12.3 points behind even the equal-weight index.

**Multi-factor first version (equally weighted top decile)**

- Output: 10.8% a year against the equal-weight S&P 1500's 10.5%, 2000 to 2024.
- Calculation: it was 2.2% a year ahead of equal weight in 2001 to 2005 and 1.0% a year behind in 2006 to 2024.
- Output: against SPTM it is 1.6% a year behind from 2006.

Notebook 4.1 reaches the same conclusion for the factor books: "Most portfolios are one bet in disguise."

**Suggestions**

- As a first trial, weight QGARP by market value. The setting already exists.
- Better, build every long-only book relative to index weights (2.4).

### 2.4 Build the books with the risk model you already have

- The section 3 strategies select a top slice and weight it equally or by market value. They do not use the risk model of notebook 4.2 or the allocation tools of notebook 5.5.
- Output (5.5): weighting by score over the risk model's covariance "helped every signal, by 0.9 to 4.7 points a year".

**Suggestion: build each long-only book as index weights plus active tilts**

- tilt size in proportion to the score and in inverse proportion to stock-specific risk
- predicted beta held at 1
- sector and size exposures neutral, or within narrow bands
- a tracking-error target, for example 3% to 4%
- a cap on each active weight
- a limit or penalty on turnover

### 2.5 Use the bottom of the ranking

Output: next-month return against the universe average, a year.

| Score | Worst tenth | Best tenth |
|---|---|---|
| Quality (3.2) | -2.4% | +0.4% |
| Net issuance (3.2) | -3.2% | +2.1% |
| Value (3.2) | -2.2% | +2.6% |
| QGARP, S&P 500 (3.5) | -1.9% | 0.0% |
| QGARP, S&P 1500 (3.6) | -1.9% | +2.1% |

- Output (3.6): "the information sits more at the bottom".
- Output (3.1): the second version was designed to "avoid them and hold most of the rest", yet it holds only the top 30%.
- Output (5.6): long-only top-decile books passed 0.20 to 0.32 of their signal into the portfolio. Long-short books passed 0.63 to 0.71.

**Suggestions**

- Hold the index minus the worst fifth, weighted by market value. The existing entry and exit thresholds can express this.
- Or allow a limited short extension such as 130/30.

### 2.6 Spend active risk in mid and small caps

- Output: 26 factors and composites pass the false-discovery test on the S&P 1500, against 4 on the S&P 500. Both counts include the suspect revision measures of 1.1.
- Output (3.2): "many anomalies are concentrated in smaller stocks".

**Suggestion**

- Hold the largest names near index weight and concentrate active positions in mid and small caps, where the scores work.

### 2.7 Cap single-name weights

- The market-value-weighted book has no limit on a single name. Holding the top 30% of names scales up each holding's index weight.
- Output (3.3): an earlier data error gave Alphabet a third of the portfolio in 2015. A cap would have contained it.

**Suggestion**

- Set a maximum active weight per name, for example 1 to 2 points.

---

## 3. Signals: prune before adding

Output (4.2): over 2007 to 2024 your risk model's pure Value factor earned nothing, and Momentum and Profitability 0.6% a year each. The gain is in which measures are used, not in adding factors.

### 3.1 Reweight QGARP towards what registers

Output: t-statistic of each input alone against next month's return.

| Input | QGARP weight | S&P 1500 | S&P 500 |
|---|---|---|---|
| Value composite | 30% | +1.49 | +0.30 |
| Growth composite | 30% | +1.19 | +0.47 |
| Quality, alpha half | 20% | +2.96 | +2.02 |
| Quality, defensive half | 10% | +2.38 | +1.07 |
| Revisions from analyst counts | 10% | suspect (1.1) | suspect (1.1) |
| Consensus revisions, 1 month | not used | +2.70 | +1.47 |
| Momentum composite | not used | -0.03 | +0.27 |

- Value and growth carry 60% of the weight and are the two weakest inputs on both universes.
- Your own rule applies: "changing the weights after seeing this table would be fitting it". Any reweighting is one new trial, written down first.

**Suggestion**

- Test one variant that moves weight from value and growth to quality and consensus revisions.

### 3.2 Narrow the growth composite

- Output (3.6): of its 11 inputs, the three sustainable-growth measures have t-statistics of 2.8 to 3.5.
- Output (3.6): the other eight are below 1.9, and four are slightly negative.
- Sustainable growth is return on equity times the share of profit retained, so it is a profitability measure.

**Suggestion**

- Test a growth composite built on sustainable growth alone, or fold it into quality.

### 3.3 Rebuild the value score on earnings and cash flow

Output (3.6, S&P 1500): t-statistic of each yield alone.

| Yield | t-statistic |
|---|---|
| Earnings yield, next twelve months | +3.15 |
| Earnings yield, trailing | +3.05 |
| Free-cash-flow yield | +2.42 |
| EBIT yield | +2.41 |
| Sales yield | +0.71 |
| Book yield | -0.24 |

- The simple value score is EBIT to enterprise value, earnings yield, book yield and sales yield. Half of it is the two weakest measures.
- Output (3.5): on the S&P 500 no yield registers. Value's contribution will come from mid and small caps.

**Suggestion**

- Test a value score built on earnings yield, EBIT yield and free-cash-flow yield, ranked within sectors.

### 3.4 Take momentum and low volatility out of the long-only mix

- Output (3.2): top deciles 1.2% and 1.7% a year behind the universe.
- Output (3.4): momentum as a long-short lost 4.1% a year with an 81% drawdown and the highest turnover, 9.4.
- Output (3.6, 3.5): the momentum composite's t-statistic is -0.03 and +0.27.

**Suggestions**

- Drop momentum from the long-only composite, or use it only to exclude the worst names.
- Treat low volatility as a risk control (2.1), not as a source of return.

### 3.5 Build on the signals that hold up

- **Net issuance.** Output (3.2): worst tenth -3.2%, best tenth +2.1% (t 1.9).
- **Profitability.** Output (4.1): the top 20% by quality, weighted by market value on the S&P 500, beat SPY by 1.9% a year over 15 years with a 4.5% tracking error. It is the only factor book ahead of SPY.
- **Issuance with return on capital.** Output (5.6): the steadiest signal of section 5, with a mean IC of +0.0252 that is "steady or strengthening".
- **Cash-flow yield.** Output (3.2): best tenth +3.0% a year (t 2.0), the strongest top decile tested there. It added nothing in the market-value-weighted book, which points to construction and to smaller names.
- **Consensus revisions.** Output: t +2.70 on the S&P 1500 and +1.47 on the S&P 500, with no survivorship problem.

**Suggestion**

- Test the 4-month consensus revision. The code already calculates it, but it is never tested or offered as a QGARP input. It should turn over far less than the 1-month version's 70%.

### 3.6 Stabilise the learned value score

- The model predicts each stock's raw return over the next year, with deep trees (depth 10).
- Output (3.3): results moved between reruns, from 14.5% a year and 0.4% ahead of SPTM to 13.5% and 0.4% behind. The notebook notes that "a learned score moves this much with small changes to its inputs".

**Suggestions**

- Predict the cross-sectional rank or the return over the universe, not the raw return.
- Use shallower trees with stronger regularisation, and average several fits.
- Reuse the validation protocol of section 5.

---

## 4. Long-short strategy

Output (3.4): 4.5% a year, 16.4% volatility, Sharpe 0.35, maximum drawdown -49%, turnover 7.05. The two worst years are 2009 (-29.7%) and 2020 (-31.0%).

### 4.1 Replace the beta-neutral leg scaling

- The code scales each leg to a beta of 1. A low-beta long leg is therefore levered up and a high-beta short leg scaled down.
- Test: with leg betas of 0.7 and 1.4 the result is 143% long and 71% short. With 0.8 and 1.3 it is 125% long and 77% short.
- That is a net-long, levered book that depends on the beta estimates. It is the exposure that hurts in sharp rebounds.
- Output (3.4): its daily beta to SPY is 0.19, not zero.

**Suggestions**

- Make the book dollar-neutral, beta-neutral and sector-neutral with the risk model.
- Cap gross exposure.

### 4.2 Combine factors as separate sleeves

Output (3.4): the same book on one factor at a time.

| Factor | Return a year | Sharpe | Maximum drawdown | Turnover |
|---|---|---|---|---|
| Value | -0.8% | 0.01 | -64% | 4.6 |
| Quality | +4.6% | 0.47 | -25% | 3.8 |
| Momentum | -4.1% | -0.14 | -81% | 9.4 |
| Low volatility | +7.2% | 0.42 | -66% | 4.6 |
| All four | +4.5% | 0.35 | -49% | 7.1 |

- Averaging the scores did not diversify. Quality alone beats the mix on Sharpe and drawdown.

**Suggestions**

- Build one book per factor, scale each to the same risk, then combine them.
- Scale momentum by its recent volatility, or leave it out.

### 4.3 Adopt the construction that worked in notebook 5.5

- Output (5.5): the issuance and return-on-capital composite, long-short with risk-model weights, returned 6.9% a year with a Sharpe ratio of 0.65 and a -24% drawdown over 2015 to 2024.
- Calculation: the 3.4 book returned 4.5% a year over the same ten years.

### 4.4 Cut the turnover

- Calculation: at 7.05 times turnover and 25 to 30 basis points a trade, costs are roughly 1.8 to 2.1 points a year, before borrowing fees. The net return is 4.5%.
- The controls in section 5 apply here first.

---

## 5. Turnover and costs

Output: turnover a year.

| Book | Turnover |
|---|---|
| Multi-factor, first version | 3.0 |
| Multi-factor, second version | 1.2 to 1.5 |
| S&P GARP rebuild | 1.6 |
| QGARP, S&P 500 | 2.5 |
| QGARP, S&P 1500 | 2.6 |
| Long-short | 7.05 |

- Every rebalance resets all holdings to equal weight.
- Test: that alone creates 0.7 to 1.0 of turnover a year with no change of names.
- Output (5.6): a top-decile equal-weight book goes from +0.02% gross to -0.77% net active return at a turnover of 3.1.
- Calculation: costs are about 0.7% a year for the first version and 0.65% to 0.8% for QGARP on the S&P 1500.
- Output (5.6): an estimate from daily prices puts the large-company cost at 22 basis points, against the 10 charged.

**Suggestions**

- Trade only entries and exits, and let existing holdings drift within a band.
- Widen the exit threshold.
- Or rebalance quarterly in staggered thirds.
- Rerun the section 3 books at twice the costs, as notebook 5.6 does for section 5.

---

## 6. Testing discipline

**Where the evidence stands**

- Output: the hold-out from 2025 was opened on 14 November 2026 for the GARP rebuild and for QGARP on both universes. It is still closed for the multi-factor families.
- Output: the run registry was restarted on 15 November 2026, so trial counts understate what has been tried. QGARP's deflated active Sharpe is 41% on the new registry and 2% counting the eighteen earlier trials.
- Output (3.1): about seven variants of the second version were tried without being registered.
- Calculation: an information ratio of 0.3 over 21 years has a t-statistic of about 1.4. It would take about 44 years to reach 2. One sample cannot confirm a modest edge.

**Suggestions**

- Write one specification down before running it: signals, weights, construction, benchmark, cost assumption and pass mark.
- Run it once. Count the earlier trials when deflating its result.
- Replicate it unchanged on the FTSE All-Share. That universe is already loaded, and no section 3 strategy uses it.
- Rely on replication across universes and on economic rationale, not on the significance of one backtest.

**A realistic yardstick**

The best long-only results on the platform so far:

- Output (4.1): quality, weighted by market value on the S&P 500: +1.9% a year against SPY over 15 years.
- Output (5.5): the logistic model with risk-model weights: +1.5% a year against SPTM, information ratio +0.29, over 2015 to 2024.

---

## Suggested order of work

1. Check the revision counts (1.1) and the mid-2020 spike (1.5).
2. Fix the learned-value comparison (1.2), the factor mix (1.3) and the benchmark (1.4). Retake the regression baseline.
3. Refresh the write-ups (1.6) so that decisions rest on current numbers.
4. Write down one revised long-only specification: signals from section 3, construction from section 2.
5. Run it once on the S&P 1500, then unchanged on the FTSE All-Share.
6. Rebuild the long-short book (section 4) on the same construction.
7. Apply the turnover controls (section 5) and repeat the cost stress.

---

## Appendix: what was checked, and how

**Calculations from your yearly tables**

- Learned value's cash period and its cost in CAGR (1.2)
- The second version's approximate alpha, from its reported beta, tracking error and information ratio and SPTM's average return (2.1)
- QGARP and the first version against SPTM and the equal-weight S&P 1500 (2.3)
- The long-short book's 2015 to 2024 return (4.3)
- Cost estimates from turnover and the per-segment costs (4.4, 5)
- The t-statistic of a given information ratio (6)

**Tests of your functions on simulated data**

- Spread of the value and quality composites against single z-scores (1.3)
- Selection odds for names with missing scores (1.3)
- Leg sizes after the beta-neutral scaling (4.1)
- Turnover from monthly re-equalising with no change of names (5)

**External sources**

- SSGA fact sheet for SPTM's launch date and index history (1.4, 2.2)
- LSEG developer forum thread on the revision-count fields (1.1)

**Not checked**

- Anything that needs the research data: the revision-count history itself, the cause of the mid-2020 spike, score coverage by sector, and the effect of each suggestion on returns.
