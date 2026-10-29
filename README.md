# DECA SMG pump-reversal scanner

This project ranks stocks that have made unusually large, high-volume moves by
3:40 p.m. Eastern and estimates their short return from that day's official
closing price through the next trading day's official close. It writes a CSV for a three-person team to review and
uses a strictly chronological walk-forward backtest.

The model is a decision aid. A large move can reflect real information, and a
missing headline does not prove that a move is unjustified. Every selected row
therefore requires a catalyst and short-interest review before an order is
entered.

## What it measures

The first version combines:

- one-, two-, five-, and twenty-day price acceleration;
- abnormal return relative to SPY and rolling beta;
- volume versus the prior 20-session median and dollar liquidity;
- realized volatility, intraday range, upper wick, and closing location;
- distance from the 20-session average and a continuation-risk flag;
- optional current market cap, short interest, and binary-event risk fields.

The hard filter also requires at least $500,000 in normal median daily dollar
volume, so a one-day volume burst in an otherwise dormant ticker does not pass.
For the latest candidates, the scanner also retrieves free Yahoo headlines and
short-interest fields. It flags words associated with material events but does
not infer that silence proves a pump. Headline coverage is incomplete, so the
team's catalyst review remains required.

Micro-cap status is a useful universe and risk feature, but not evidence that a
stock will fall. Beta is also secondary: abnormal return, volume, price shape,
catalyst quality, and squeeze risk are closer to the hypothesis being tested.

## Timing and leakage

Each feature row contains only information known by 3:40 p.m. Eastern. The Yahoo
fetcher aggregates fully completed five-minute bars ending by 3:40, then separately records the
official-close proxy solely as the realized entry/exit label. This matches the
DECA end-of-day rule: an order entered before 4:00 p.m. is filled at that day's
closing price. The official close is never used as a same-day model feature.

Free Yahoo intraday history is short and the current-symbol universe creates
survivorship bias. Treat the first backtest as a screening experiment. Start
saving the daily snapshots now; a paid point-in-time source is the first upgrade
if the strategy shows an edge.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Build a current U.S. micro- and small-cap universe, then fetch the snapshots.
SPY is added automatically:

```bash
deca-smg discover --output data/universe.csv --maximum-tickers 1000
deca-smg fetch --tickers data/universe.csv --output data/snapshots.csv
deca-smg backtest --input data/snapshots.csv --output-dir outputs/backtest
deca-smg scan --input data/snapshots.csv \
  --model outputs/backtest/model.joblib \
  --output outputs/shortlist.csv
```

Save each day's universe file. Reusing a universe discovered today for older
dates omits stocks that were delisted or no longer meet the filters and makes a
historical result look cleaner than it was.

The backtest writes trades, an equity curve, metrics, a slippage-sensitivity
table, and a model-versus-fixed-heuristic comparison. The scanner writes all qualifying stocks and marks at most two as
selected. The team should fill in `catalyst_review`, `short_interest_review`,
and `team_notes` before trading.

On a first run, there may be fewer than 120 labeled pump candidates across 20
prior candidate dates. In that case, `backtest` reports the available sample,
writes an honest fixed-heuristic backtest, and saves a heuristic-mode artifact.
`scan` then ranks candidates for research with that fixed formula, but it does
not mark any row selected. The walk-forward model can mark rows selected only
after at least 30 out-of-sample trades, positive net total and average trade
returns, and no worse than a 15% out-of-sample drawdown. The CSV's
`validation_status` and `selection_reason` columns make that state explicit.
Re-run `fetch` and `backtest` as more history accumulates; the same commands
switch to the walk-forward model once the training and validation minimums are
met.

## Daily Google Sheet

The included GitHub Actions workflow can run the complete scan on weekdays at
3:45 p.m. New York time and replace a `Shortlist` tab in an existing Google
Sheet. Follow the one-time [Google Sheets setup](google-sheets/SETUP.md), then
run the workflow manually once from the repository's **Actions** page. Each run
also keeps its CSV files as a downloadable GitHub artifact for 14 days.

## Verified 2026–27 competition constraints

The [official DECA guidelines](https://www.deca.org/compete/stock-market-game)
and the [linked SIFMA Program Rules](https://www.deca.org/smg/rules) establish:

- the qualifying game runs September 8 through December 4, 2026;
- orders entered before 4:00 p.m. ET receive that day's closing price;
- stock short sells and covers are allowed in this margin game;
- eligible stocks must be NYSE/Nasdaq, at least $25 million market cap, and at
  least $3 on both the day before and day of entry; the minimum order is 10 shares;
- each transaction costs $5 plus an SEC fee on sells;
- a team may borrow up to 50% of total equity, and the DECA per-security maximum
  is 20% of total equity multiplied by 1.5, effectively 30%;
- by October 23, teams must hold at least $10,000 in each of stocks, mutual
  funds, and bonds through the end of the competition; only long stock counts;
- ranking is percent return relative to S&P 500 growth, and the top 25 teams in
  each DECA region qualify for ICDC.

SMG can exclude extremely volatile or infrequently traded securities even when
they pass the numeric filters. The game's Security Table and any local rules are
therefore the final eligibility check. The scanner uses a stricter $50 million
research floor and leaves a rule-check column in the team CSV.

## Interpreting the result

Do not optimize only for total return. Require enough trades, compare the model
with the heuristic baseline, inspect the maximum drawdown, and verify that the
edge survives the official transaction cost. A top-five goal calls for risk,
but one unbounded squeeze can erase many correct one-day calls. The strategy
limits itself to two positions at 20% each, below the game's 30% ceiling. Keep
the required $30,000 of long stock, mutual-fund, and bond holdings separate from
the tactical short allocation.
