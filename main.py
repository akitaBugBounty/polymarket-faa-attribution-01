# polymarket-faa-attribution-01 entry (Python 3.13, standard library only).
#
# Forecast: for each market, the five quantiles (5/25/50/75/95%) of the Yes
# price at each of the 240 minutes after the checkpoint.
#
# Method: Polymarket Yes prices sit flat most minutes, then jump. A raw
# standard deviation is inflated by those rare jumps and makes the spread far
# too wide. So sigma comes from a robust read of the recent 1-minute moves:
# the 90th percentile of |price change| over the last 1440 minutes, divided
# by 1.644854 (the normal 90% point), floored at 0.0005 and capped at 0.02.
# The spread then grows as h**0.4, not sqrt(h): jumps are discrete and partly
# mean-reverting, so the distribution widens more slowly than a random walk.
# The median stays at the last price (no drift: recent drift did not help on
# validation folds). Quantiles are the standard normal's, clamped to [0, 1],
# rounded to 6 decimals.
#
# Validated on the NVIDIA CEO practice folds (the same practice family this
# competition ships): practice-fold mean pinball loss ~0.316 vs house
# baseline ~0.393. No flat-market floor-lift: an earlier 0.25 * biggest-move
# lift inflated sigma up to 40x on flat markets and scored 0.656, so it is
# NOT used here.
#
# Reads $MAKEFUN_INPUT_DIR (else /input/), writes $MAKEFUN_OUTPUT_DIR
# (else /output/). One <id>.out per <id>.in, deterministic, no network.

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
WINDOW = 1440          # minutes of history used for the robust sigma
SIGMA_FLOOR = 0.0005   # per-minute spread floor
SIGMA_CAP = 0.02       # per-minute spread cap
ALPHA = 0.4            # spread grows as h ** ALPHA


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def robust_sigma(prices):
    """Robust per-minute spread from the 90th percentile of |1-min moves|."""
    n = len(prices)
    # Too little history to estimate a spread: fall back to the baseline's.
    if n < 30:
        return 0.0025
    start = max(1, n - WINDOW)
    diffs = sorted(abs(prices[i] - prices[i - 1]) for i in range(start, n))
    if not diffs:
        return SIGMA_FLOOR
    q90 = diffs[min(len(diffs) - 1, int(0.9 * len(diffs)))]
    sig = q90 / 1.644854
    return min(max(sig, SIGMA_FLOOR), SIGMA_CAP)


def forecast(market, horizon):
    """Forecast one market: 5 quantiles for each of the next `horizon` minutes."""
    prices = market["prices"]
    if not prices:
        raise ValueError("market %s has no prices" % market["id"])
    x = prices[-1]
    sig = robust_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        s = sig * (h ** ALPHA)
        row = [price6(x + z * s) for z in Z]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case.get("horizon_minutes", 240))
        forecasts = []
        for m in case["markets"]:
            forecasts.append({"id": m["id"], "q": forecast(m, horizon)})
        out = {"forecasts": forecasts}
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"), sort_keys=True)
            fh.write("\n")


if __name__ == "__main__":
    main()
