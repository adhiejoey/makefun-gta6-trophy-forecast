# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adhiejoey/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adhiejoey/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one, built for
make.fun's **GTA VI trophy rarity forecast** bounty
(`gta6-trophy-rarity-forecast-01`, $250, 20 places paid) —
https://make.fun/bounty/gta6-trophy-rarity-forecast-01.
Deadline: **Sat 21 Nov 2026 11:30 UTC / 18:30 WIB**. No entry fee.

---

<sub>Entered on [make.fun](https://make.fun/?ref=mcpf7r82&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards · Referral code: `mcpf7r82`</sub>

## What it does

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads one case — every read the house made of Exophase's GTA VI PS5 trophy
page up to the checkpoint — and forecasts each of the game's 55 trophies'
earn rate at the case's target as five quantiles (5%, 25%, 50%, 75%, 95%).

Run it:

```sh
python3 main.py            # scans the input dir, writes one .out per .in
# or with explicit dirs:
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Input: JSON `{"schema": "makefun/gta6-trophies-case/v1", "id", "at",
"horizon_minutes", "target", "levels", "game", "trophies", "history",
"notes"}`. Output: JSON `{"trophies": [{"id": 1, "q": [q05, q25, q50,
q75, q95]}, ...]}` — exactly one row per trophy, each id once, each q
five finite numbers from 0 to 100, non-decreasing, rounded to 4 decimals.
Runs in well under 1 s on 1 CPU. Outside the grader's container,
`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` stand in for `/input` and
`/output`.

## Method

An Exophase earn rate is a ratio, `rate = 100 * E / N`, and the two sides
move for different reasons, so the program models them separately and
divides at the target:

- **Players tracked (N)** grows in steps, because Exophase refreshes a
  game's counts on its own schedule (the page sat frozen at 638 players
  all launch night, then jumped to 3,868 at the 04:30Z read on 19 Nov).
  Fitted as `log N = a + b * ln(1 + hours since release)`, anchored at
  the last distinct read, shrunk toward the launch-night prior slope.
- **Earners (E)** only accumulate, so each trophy's earner count is
  fitted the same way, with its slope shrunk toward the pooled slope
  across trophies. Slopes are floored at 0 and capped, and medians are
  clipped to per-class ceilings (platinum / 100% / grind trophies stay
  near zero a week in — the campaign runs ~50 hours, the platinum ~200
  — while the first chapter can reach tens of percent).
- **Flat history** (a single distinct read, as in the practice cases)
  cannot be fitted, so a refresh/dilution fallback applies: N multiplies
  (~6x over a 90-minute horizon, the size of the first refresh the kit
  records) while each trophy's earners grow only by the share of newly
  tracked players who earn it quickly. The kit's totals pin this down:
  across the first refresh players earned 3,333 more trophies, but
  3,230 of them were Gone Fishing, so the other 54 trophies shared only
  ~103 new earners — a refresh cuts most rates to roughly a quarter of
  their launch-night value.
- **Spread:** quantiles are symmetric on the logit scale around the
  median, with sigma combining the house baseline's random-walk term
  (`0.5 * sqrt(horizon in days)`), the OLS prediction error of the two
  log-log fits at the target, and a refresh-step term for the staleness
  of the answering read.

Score: **pinball loss on the logit scale**; lowest wins. A reward needs a
**lower loss than the best house entry** (persistence baseline, fitted
curve baseline, house model forecast).

## Research behind the priors

- GTA VI sold an estimated ~19m copies on day one (Sensor Tower, via
  Reuters/GameSpot), ~75% on PS5, with 28–35m copies projected for week
  one; the campaign runs ~50 hours and the platinum ~200 hours (Insider
  Gaming, unconfirmed by Rockstar).
- GTA V's first story trophy eventually reached ~73% of players
  (PSNProfiles/PocketPSN), so early-story trophies have headroom, but
  week-one rates are held down by the flood of newly tracked players.
- The live Exophase page on 20 Nov 00:34 UTC already showed the pattern:
  Chapter 1 down to 0.73% from 1.25% at launch night while the earliest
  activity trophy stayed near 2% — dilution hits late-story and grind
  trophies hardest.

## Local checks (20 Nov 2026)

- Practice cases `p00`/`p01`: valid output (55 rows, ≤65,536 bytes),
  byte-identical across two runs, ~0.07 s per run.
- Scored against practice-answer proxies rebuilt from the totals the kit
  states (refresh to 3,868 players / 4,276 trophies earned): pinball loss
  ~0.19, vs ~1.40 for the house baseline on the same proxies (the house
  states ~1.51 for the baseline on the true practice answers).
- A synthetic final-like case (338 reads, N 638 → 55k over two days,
  per-class earner growth) runs in ~0.09 s with sane medians (Chapter 1
  rising, platinum ~0.18%, Gone Fishing ~100%).
- Edge cases (empty history, null rates, all-zero rates, zero horizon)
  all produce valid output without crashing.

## Layout

```
main.py                  Python 3.13 entrypoint: the forecaster
README.md                this file
LICENSE                  MIT (SPDX-License-Identifier: MIT)
.makefun/owner.json      payout wallet claim file
```
