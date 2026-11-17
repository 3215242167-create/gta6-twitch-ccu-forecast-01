[![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/3215242167-create/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/3215242167-create/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

A forecaster for make.fun's [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty ($250 in USDC). Given a case file with the first hour of Grand Theft Auto VI's release day on Twitch, it writes five quantiles (5/25/50/75/95%) of the category's concurrent viewers for each of the remaining 23 UTC hours.

## Method

Comparable-launch kernel quantiles. Each case carries ~40 past big game launches with their full hourly Twitch curves. A launch day's shape — each hour's viewers as a log ratio to its own hour 1 — clusters by what kind of launch it was, and that is visible at the checkpoint from four features: how far hour 1 sits above the game's pre-release baseline, the level and momentum of the hours just before day start, and the launch's sheer scale. For every forecast hour the program pools the comparables' log ratios for that hour, weights each comparable by a Gaussian kernel on its standardised feature distance to this case, and reads off the weighted quantiles. Chosen by leave-one-out backtesting over the 37 complete comparable launch days in the practice cases: mean pinball loss on the log scale 0.28, against 0.46 for the flat persistence baseline. With too little usable data it falls back to that persistence forecast, so the output is always a valid forecast.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input directory, writes `<id>.out` beside the requested output directory. Deterministic: two runs on the same case write the same bytes.

## License

MIT — see `LICENSE`.
