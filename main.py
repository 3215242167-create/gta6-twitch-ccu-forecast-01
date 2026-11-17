"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (make.fun gta6-twitch-ccu-forecast-01).

Approach: comparable-launch kernel quantiles.

Every case carries ~40 past big game launches ("comparables"), each with its full hourly
Twitch category curve from 14 days before its release day to 3 days after. A launch day's
shape, measured as the log ratio of each hour's viewers to that launch's own hour-1 viewers,
falls into families: launches already live worldwide at 00:00 UTC decay from a huge hour 1,
while launches that unlock mid-day start near zero and peak in the evening. Which family a
new launch belongs to is visible at the checkpoint from four features of its own case:

  f1 = log(hour-1 viewers / its pre-release baseline)   (how far above normal the category is)
  f2 = log(mean viewers of the 6 hours before day start / hour-1 viewers)
  f3 = log(the last hour before day start / hour-1 viewers)
  lp = log(hour-1 viewers)                              (the launch's sheer scale)

For each forecast hour, this program pools the comparables' log ratios for that hour, weights
each comparable by a Gaussian kernel on the standardised feature distance to this case
(bandwidth 1.2, weight floor 0.10 so no launch is fully ignored), and reads off the weighted
5/25/50/75/95% quantiles. The forecast is hour 1's print times exp(quantile). Model and
hyperparameters were chosen by leave-one-out backtesting over the 37 complete comparable
launch days in the practice cases: mean pinball loss on the log scale 0.28, against 0.46 for
the flat persistence baseline.

Fallbacks keep the output valid no matter what: with too few usable comparables, or a case
whose history supports no feature at all, it answers with the persistence forecast (every
hour at hour 1's print, spread widening like a random walk, sigma 0.15 per root hour). A
history that supports only some features still forecasts: the kernel imputes each missing
feature at the comparables' mean for it.

Contract (from the kit's starter): the grader runs `python3 -P -s /bot/main.py` in
python:3.13-slim, standard library only, no network. It reads every <id>.in in the input
directory (MAKEFUN_INPUT_DIR, default /input) and writes <id>.out to the output directory
(MAKEFUN_OUTPUT_DIR, default /output): {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]}
for the case's forecast hours (2..24), each q five finite numbers in (0, 1e9], non-decreasing.
Two runs on the same case write the same bytes: no randomness, no clocks, sorted iteration.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; put it back first (starter's line).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
BASE_SIGMA = 0.15  # persistence fallback spread, per root hour (the house baseline's value)

BANDWIDTH = 1.2    # Gaussian kernel bandwidth on standardised feature distance
WEIGHT_FLOOR = 0.10
MIN_POOL = 8       # usable comparables needed before the kernel model is trusted
MIN_DAY_HOURS = 20  # a comparable's release day must be this complete to teach the shape


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (the starter's rounding, kept byte-compatible)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text) -> float | None:
    """An ISO-8601 'Z' timestamp to epoch seconds; None when unreadable."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None


def hour_buckets(points, t0: float) -> dict[int, float]:
    """Mean viewers per hour offset from t0, for points of any spacing.

    A point's timestamp is its `from` (hourly rows) or `at` (history rows); a point covers
    the hour it starts in, so floor division buckets minute reads and hourly points alike.
    """
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for p in points or []:
        if not isinstance(p, dict):
            continue
        ts = parse_ts(p.get("from") or p.get("at"))
        v = p.get("viewers")
        if ts is None or not isinstance(v, (int, float)) or v < 0:
            continue
        off = int((ts - t0) // 3600)
        sums[off] = sums.get(off, 0.0) + float(v)
        counts[off] = counts.get(off, 0) + 1
    return {off: sums[off] / counts[off] for off in sums}


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def case_features(buckets: dict[int, float], p1: float) -> tuple | None:
    """The four checkpoint features from hour buckets and hour 1's print.

    Each feature is None when the history behind it is not there (a short recording, a
    gappy one); the kernel then imputes that feature at the pool's mean, so a partial
    history still informs the forecast instead of discarding the case. The tuple itself
    is None only when there is no usable hour-1 print at all.

    Baseline: median hourly mean over the two weeks before the final day (offsets -336..-25).
    With a shorter recorded history, whatever hours exist before offset -24 are used,
    but at least 6 such hours must be there for f1 to be computed.
    """
    if p1 <= 0:
        return None
    base = [v for off, v in buckets.items() if -336 <= off < -24]
    f1 = math.log(p1 / max(1.0, median(base))) if len(base) >= 6 else None
    pre6 = [buckets[o] for o in range(-6, 0) if o in buckets]
    f2 = math.log(max(sum(pre6) / len(pre6), 1.0) / p1) if pre6 else None
    f3 = math.log(max(buckets[-1], 1.0) / p1) if -1 in buckets else None
    return (f1, f2, f3, math.log(p1))


def comparable_pool(case: dict) -> list[tuple[dict[int, float], float, tuple]]:
    """Usable comparables: (release-day curve {hour: viewers}, its hour-1 print, its features)."""
    pool = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        t0 = parse_ts(comp.get("day_start"))
        if t0 is None:
            continue
        buckets = hour_buckets(comp.get("hourly"), t0)
        day = {off + 1: v for off, v in buckets.items() if 0 <= off < 24 and v > 0}
        if len(day) < MIN_DAY_HOURS or 1 not in day:
            continue
        # A comparable with a partial history still teaches the shape: keep it, with
        # whichever features its history supports (the kernel imputes the rest).
        feats = case_features(buckets, day[1])
        if feats is not None:
            pool.append((day, day[1], feats))
    return pool


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Weighted quantile of (value, weight) pairs, midpoint-interpolated (deterministic)."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    pts = []
    cw = 0.0
    for v, w in pairs:
        cw += w
        pts.append(((cw - 0.5 * w) / total, v))
    if tau <= pts[0][0]:
        return pts[0][1]
    if tau >= pts[-1][0]:
        return pts[-1][1]
    for (p0, v0), (p1_, v1) in zip(pts, pts[1:]):
        if p0 <= tau <= p1_:
            return v0 + (v1 - v0) * (tau - p0) / max(p1_ - p0, 1e-12)
    return pts[-1][1]


def kernel_forecast(p1: float, target_feats: tuple, pool: list, hours: list[int]) -> dict[int, list]:
    """Weighted empirical log-ratio quantiles per hour, anchored on hour 1's print.

    Features are standardised on the pool entries that have them; a missing feature
    (either side) is imputed at the pool mean, i.e. contributes a z-score of 0.
    """
    n_feat = len(target_feats)
    means, sds = [], []
    for i in range(n_feat):
        vals = [f[i] for _, _, f in pool if f[i] is not None]
        if not vals:
            return {}
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        means.append(m)
        sds.append(math.sqrt(var) or 1.0)

    def zscore(feats: tuple, i: int) -> float:
        return 0.0 if feats[i] is None else (feats[i] - means[i]) / sds[i]

    tz = [zscore(target_feats, i) for i in range(n_feat)]
    weights = []
    for _, _, f in pool:
        d2 = sum((zscore(f, i) - tz[i]) ** 2 for i in range(n_feat))
        weights.append(WEIGHT_FLOOR + math.exp(-d2 / (2 * BANDWIDTH * BANDWIDTH)))
    lp = math.log(p1)
    out = {}
    for h in hours:
        pairs = [(math.log(day[h] / cp1), w) for (day, cp1, _), w in zip(pool, weights) if h in day]
        if len(pairs) < MIN_POOL:
            return {}
        out[h] = [lp + weighted_quantile(pairs, tau) for tau in LEVELS]
    return out


def persistence_forecast(p1: float, hours: list[int]) -> dict[int, list]:
    lp = math.log(p1)
    return {h: [lp + z * BASE_SIGMA * math.sqrt(h - 1) for z in Z] for h in hours}


def clean_quantiles(log_qs: list[float]) -> list:
    """Exponentiate, sort, clamp to (0, 1e9], round like the starter."""
    qs = sorted(min(1e9, max(1e-6, math.exp(q))) for q in log_qs)
    return [round2(q) for q in qs]


def forecast_case(case: dict) -> dict:
    hours = []
    for row in case.get("forecast") or []:
        if isinstance(row, dict) and isinstance(row.get("hour"), int):
            hours.append(row["hour"])
    if not hours:
        hours = list(range(2, 25))
    hours = sorted(set(hours))

    p1 = None
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1 and isinstance(row.get("viewers"), (int, float)):
            p1 = float(row["viewers"])
            break
    t0 = parse_ts(case.get("day_start"))
    history = case.get("history")
    if not isinstance(history, list):
        history = case.get("history_hourly")
    buckets = hour_buckets(history, t0) if t0 is not None else {}
    if p1 is None:
        p1 = buckets.get(0, 1.0)
    p1 = max(1.0, p1)

    log_forecast: dict[int, list] = {}
    target_feats = case_features(buckets, p1) if buckets else None
    pool = comparable_pool(case)
    # The kernel needs the pool plus at least one real feature beyond scale itself;
    # anything less and the honest answer is the persistence fallback.
    n_present = sum(1 for v in target_feats if v is not None) if target_feats else 0
    if target_feats is not None and n_present >= 2 and len(pool) >= MIN_POOL:
        log_forecast = kernel_forecast(p1, target_feats, pool, hours)
    if not log_forecast:
        log_forecast = persistence_forecast(p1, hours)

    return {"hours": [{"hour": h, "q": clean_quantiles(log_forecast[h])} for h in hours]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
