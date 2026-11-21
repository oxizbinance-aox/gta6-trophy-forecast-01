# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oxizbinance-aox/gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oxizbinance-aox/gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast every
Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00:00Z.

## Run

```sh
node main.ts
```

Reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): JSON
`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`, one row per trophy.

Node 22, standard library only, deterministic (two runs byte-identical),
~0.2 s per case on this machine (limit: 8 s).

## Model

Per trophy, from the case's history of Exophase reads:

1. **Dedupe.** Exophase refreshes rarely, so most reads repeat the previous
   page verbatim. Consecutive identical reads are collapsed, so the fit sees
   only real changes.
2. **Trend fit (when history moves).** Weighted least squares of
   `logit(rate)` against `ln(1 + hours since release 2026-11-19T00:00:00Z)`,
   over distinct reads from the last 24 h (recency half-life 12 h).
   Extrapolate to the target, anchored at the fitted value at the last read.
   The slope is shrunk toward 0 when the fit is noisy and clamped to
   [-4, 4] logit per e-fold. The median is capped by a tier prior
   (platinum <= 3%, gold <= 25%, silver <= 40%, bronze <= 90%).
3. **Flat history (launch night).** No trend exists before the first refresh,
   so the median is shifted down for refresh dilution: players_tracked jumps
   (launch night 638 -> 3,868) while most new players have earned nothing
   yet. New-player earn fraction by tier: bronze 0.35, silver 0.15,
   gold/platinum 0.05. The trophy at 100% stayed 100% through the first
   refresh, so it is forecast near 100% with a downward tail instead.
4. **Spread.** Combines the random-walk term (0.5 * sqrt(days), as the house
   baseline), the fit's residual SD, and a slope-uncertainty term growing
   with extrapolation distance; flat history adds a wide base spread
   (1.1 logit). Quantiles are `logistic(median + z_tau * spread)`.

## Local validation

Practice cases p00/p01 (flat launch-night history): valid shape (55 rows,
ids 1..55, non-decreasing quantiles), byte-identical across runs,
~0.17 s per run. The kit ships no practice answers, so pinball loss was
checked against simulated refresh answers (players 638 -> 3,868, new
players earning at fraction f of the old rate): loss 0.32-0.41 for
f = 0.1-0.7 with 90% coverage 0.98, vs the house baseline's 0.20-1.28
with coverage 0.00 (the house states ~1.51 on the real practice).

A synthetic final-like case (2.5 days of reads, rising story trophies,
growing players_tracked) exercises the trend path: story-trophy medians
rise above their last rates, platinum stays <= 3%, gold final-mission
<= 25%, all rows valid. Edge cases (null early rates, single-read history)
produce valid output.

## License

MIT — see LICENSE.
