"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every GTA VI PlayStation 5 trophy, its earn rate on Exophase
at the case's target, as five quantiles (5/25/50/75/95%), scored by pinball
loss on the logit scale.

Method (counts, not rates)
---------------------------
An Exophase earn rate is a ratio, rate = 100 * E / N, where N is the players
tracked and E the tracked players who have earned the trophy.  Both move for
different reasons, so the model fits them separately and divides at the
target:

* N(t): the tracked-player count.  It grows in steps, because Exophase
  refreshes a game's counts on its own schedule (the house saw the page
  frozen at 638 players from launch night until its 04:30Z read on 19 Nov,
  when it jumped to 3,868).  N is fitted as log N = a + b * ln(1 + hours
  since the release day began), anchored at the last distinct read.
* E_i(t): each trophy's earner count.  Earners only accumulate, so E_i is
  non-decreasing in truth; observed wobble is rounding of the printed rate.
  Each trophy is fitted the same way, log E_i = c_i + d_i * ln(1 + hours),
  with the slope shrunk toward the pooled slope across trophies (and the
  pooled slope shrunk toward a prior from launch night: total trophies
  earned grew 943 -> 4,276 while N grew 638 -> 3,868 over the first 4.5
  hours, i.e. slopes of about 0.9 for E and 1.1 for N in ln(1+hours)).
  Slopes are floored at 0 (earners never shrink) and capped to stop
  explosive extrapolation.

The median is 100 * E_i(target) / N(target), clipped to [0.01, 99.99] and
to a per-class ceiling (see CLASS_CEILINGS): a ~50-hour campaign and a
~200-hour platinum (Insider Gaming's pre-launch estimates, Rockstar
unconfirmed) mean the platinum, 100%-completion gold and the grind
trophies (250 stunts, gold medals everywhere, 100 WAiNK spots, card sets)
are still near zero a week in, while the first chapter can plausibly reach
tens of percent.  The ceilings only bind on runaway fits.

When the history is flat (a single distinct read, as in the practice
cases), no trend can be fitted.  The fallback is a refresh/dilution model:
over the horizon Exophase refreshes and N multiplies (prior median ~6x
for a 90-minute horizon -- the first refresh the kit records multiplied
N 638 -> 3,868 -- growing with the horizon, capped at 30x), while each
trophy's earners grow by (N_mult - 1) * s_i, where s_i is the share of
newly tracked players who earn that trophy quickly.  The kit's totals
pin this down: across that first refresh the tracked players earned
3,333 more trophies, but 3,230 of them were Gone Fishing (the one trophy
whose rate did not fall), so the other 54 trophies shared only ~103 new
earners -- a new player earns an ordinary trophy with probability only
~0.07-0.25, ~1.0 for the opening mission, ~0.25 for the earliest
activities, ~0.20 for chapter trophies.  So a refresh cuts most rates to
roughly a quarter of their launch-night value, which is why the house
baseline (persistence) scores ~1.5 on the practice.

Spread (quantiles)
------------------
Quantiles are symmetric on the logit scale around the median,
q_tau = inv_logit(logit(m) + z_tau * sigma), z = (-1.645, -0.674, 0,
0.674, 1.645).  sigma combines the house baseline's random-walk term
(0.5 * sqrt(horizon in days)), the OLS prediction error of the two
log-log fits at the target, and a refresh-step term for the staleness
of the answering read (the answer is the page as of Exophase's last
refresh before the target).  A flat history gets a wide prior sigma
(0.95), because an unseen refresh can move any rate a long way on the
logit scale.  sigma is floored at 0.30 and capped at 2.0; the opening
mission, pinned near 100%, gets a narrow sigma.

Research baked in (public sources, Nov 2026): GTA VI sold an estimated
~19m copies on day one (Sensor Tower, via Reuters/GameSpot), ~75% on
PS5, with 28-35m copies projected for week one; the campaign runs ~50
hours and the platinum ~200 hours (Insider Gaming); GTA V's first story
trophy eventually reached ~73% of players (PSNProfiles/PocketPSN), so
early-story trophies have headroom but week-one rates are held down by
the flood of newly tracked players.  The live Exophase page on 20 Nov
00:34 UTC already showed the pattern: Chapter 1 down to 0.73% from
1.25% at launch night while the earliest activity trophy stayed near
2%, i.e. dilution hits late-story and grind trophies hardest.

Deterministic: pure arithmetic on the case file, no randomness, no
network, standard library only.  Reads every *.in in the input
directory (MAKEFUN_INPUT_DIR or /input) and writes one *.out per case
to the output directory (MAKEFUN_OUTPUT_DIR or /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release-day start: hours are measured from here (the case's
# game.release_date is 2026-11-19).
RELEASE_START = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Prior log-log slopes (per ln(1 + hours)) from launch night's refresh:
# N 638 -> 3,868 and total trophies earned 943 -> 4,276 over ~4.5 h.
PRIOR_B_N = 1.0   # players-tracked growth
PRIOR_D_E = 0.9   # earner-count growth (pooled across trophies)

# Flat-history fallback: median N multiplier over a 90-minute horizon.
FLAT_N_MULT_90MIN = 6.0
FLAT_N_MULT_CAP = 30.0
FLAT_SIGMA = 0.7

SIGMA_BASE = 0.5       # logit units per sqrt(day), as the house baseline
SIGMA_FLOOR = 0.30
SIGMA_CAP = 2.0
SIGMA_STEP_PER_REFRESH = 0.15   # logit units per sqrt(expected refresh)
REFRESH_HOURS = 5.0             # typical gap between Exophase refreshes


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return max(0.0, (dt - RELEASE_START).total_seconds() / 3600.0)


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, resid_sd, sxx, x_mean, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0, mx, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        sd = math.sqrt(resid / (n - 2))
    else:
        sd = 0.0
    return a, b, sd, sxx, mx, n


def pred_se(sd, n, sxx, x_mean, x_target):
    """OLS prediction standard error at x_target (incl. new-obs noise)."""
    if n < 3 or sxx <= 0:
        return None
    return sd * math.sqrt(1.0 + 1.0 / n + (x_target - x_mean) ** 2 / sxx)


# ---------------------------------------------------------------- classes
def trophy_class(trophy):
    """Coarse class from tier + description; drives ceilings and the
    flat-history new-player earn share.  Returns (kind, ceiling, s)."""
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    tier = trophy.get("tier") or ""
    if tier == "platinum":
        return ("platinum", 0.5, 0.07)
    if "gone fishing" in desc or "gone fishing" in name:
        return ("first", 99.99, 1.0)
    if "100% completion" in desc:
        return ("grind", 0.5, 0.07)
    if ("250 stunts" in desc or "gold medal in all" in desc
            or "100 locations" in desc or "card sets" in desc
            or "50 challenges" in desc):
        return ("grind", 2.0, 0.07)
    if ("20 different races" in desc or "100 vehicles" in desc
            or "bank vaults" in desc or "explore all" in desc
            or "250,000" in desc or "all garage" in desc
            or "mission strands for strangers" in desc):
        return ("grind", 5.0, 0.07)
    if desc.startswith("complete chapter"):
        return ("chapter", 45.0, 0.20)
    if desc.startswith("complete the final mission"):
        return ("story", 35.0, 0.15)
    if desc.startswith("complete "):
        return ("story", 30.0, 0.15)
    return ("other", 25.0, 0.12)


# ---------------------------------------------------------------- snapshots
def distinct_snapshots(case):
    """Collapse the history to one snapshot per distinct page state.

    Exophase refreshes in steps, so long runs of reads are byte-identical;
    each distinct state is kept at the time of its first read.  Earner
    counts are forced non-decreasing per trophy (rounding jitter only).
    Returns a list of (x, N, [E_i]) with x = ln(1 + hours since release).
    """
    n_troph = len(case["trophies"])
    snaps = []
    last_state = None
    earner_floor = [0.0] * n_troph
    for read in case.get("history") or []:
        n_players = read.get("players_tracked")
        rates = read.get("rates") or []
        if not n_players or n_players <= 0:
            continue
        state = (n_players, tuple(rates))
        if state == last_state:
            continue
        last_state = state
        try:
            when = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        x = math.log1p(hours_since_release(when))
        earners = []
        for i in range(n_troph):
            r = rates[i] if i < len(rates) else None
            if r is None:
                e = earner_floor[i]
            else:
                e = max(earner_floor[i], r * n_players / 100.0)
            earner_floor[i] = e
            earners.append(e)
        snaps.append((x, float(n_players), earners))
    return snaps


def last_rates(case):
    """Last non-null rate per trophy, from the history's last read back."""
    n_troph = len(case["trophies"])
    out = [None] * n_troph
    for read in reversed(case.get("history") or []):
        rates = read.get("rates") or []
        for i in range(n_troph):
            if out[i] is None and i < len(rates) and rates[i] is not None:
                out[i] = rates[i]
        if all(v is not None for v in out):
            break
    return [v if v is not None else 0.0 for v in out]


# ---------------------------------------------------------------- forecasts
def fitted_medians(case, snaps, x_target):
    """Median per trophy from the log-log count fits, anchored at the
    last distinct snapshot.  Returns (medians, sigma_fit, n_snaps)."""
    n_troph = len(case["trophies"])
    x_last, n_last, e_last = snaps[-1]
    dx = x_target - x_last

    xs = [s[0] for s in snaps]
    # Players-tracked fit, shrunk toward the launch-night prior.
    _, b_n, sd_n, sxx_n, xm_n, n_n = ols(xs, [math.log(s[1]) for s in snaps])
    w_n = n_n / (n_n + 4.0) if n_n else 0.0
    b_n = w_n * b_n + (1.0 - w_n) * PRIOR_B_N
    b_n = min(2.0, max(0.0, b_n))
    log_n_star = math.log(n_last) + b_n * dx
    se_n = pred_se(sd_n, n_n, sxx_n, xm_n, x_target)

    # Per-trophy earner fits.
    slopes = []
    fits = []
    for i in range(n_troph):
        pts = [(s[0], math.log(s[2][i])) for s in snaps if s[2][i] > 0]
        if len(pts) >= 2 and len({p[0] for p in pts}) >= 2:
            a, b, sd, sxx, xm, n = ols([p[0] for p in pts],
                                       [p[1] for p in pts])
            slopes.append(b)
            fits.append((a, b, sd, sxx, xm, n))
        else:
            fits.append(None)
    slopes.sort()
    d_pool = slopes[len(slopes) // 2] if slopes else PRIOR_D_E
    k = len(slopes)
    d_pool = (k * d_pool + 3.0 * PRIOR_D_E) / (k + 3.0)
    d_pool = min(3.0, max(0.0, d_pool))

    medians = []
    sigma_fits = []
    for i, trophy in enumerate(case["trophies"]):
        kind, ceiling, _ = trophy_class(trophy)
        fit = fits[i]
        if fit is not None:
            _, d_i, sd_i, sxx_i, xm_i, n_i = fit
            w = n_i / (n_i + 3.0)
            d_i = w * d_i + (1.0 - w) * d_pool
            d_i = min(3.0, max(0.0, d_i))
            # Floor at half an earner: a trophy nobody holds yet can
            # still gain its first earners by the target.
            log_e_star = math.log(max(e_last[i], 0.5)) + d_i * dx
            se_i = pred_se(sd_i, n_i, sxx_i, xm_i, x_target)
        else:
            d_i = d_pool
            log_e_star = math.log(max(e_last[i], 0.5)) + d_i * dx
            se_i = None
        log_rate = log_e_star - log_n_star
        m = 100.0 * math.exp(log_rate) if log_rate < 0 else 100.0
        if kind == "first":
            m = min(max(m, 99.0), 99.99)
        else:
            m = clamp(m, 0.01, min(ceiling, 99.99))
        medians.append(m)
        if se_i is None or se_n is None:
            sigma_fits.append(None)
        else:
            sigma_fits.append(math.sqrt(se_i ** 2 + se_n ** 2))
    return medians, sigma_fits, len(snaps)


def prior_medians(case, horizon_minutes):
    """Flat-history fallback: refresh/dilution model (see docstring)."""
    horizon_h = horizon_minutes / 60.0
    n_mult = FLAT_N_MULT_90MIN * math.sqrt(max(horizon_h, 0.0) / 1.5)
    if horizon_h < 1.5:
        n_mult = 1.0 + (FLAT_N_MULT_90MIN - 1.0) * (horizon_h / 1.5)
    n_mult = min(FLAT_N_MULT_CAP, max(1.0, n_mult))
    rates = last_rates(case)
    medians = []
    for i, trophy in enumerate(case["trophies"]):
        kind, ceiling, s = trophy_class(trophy)
        r = rates[i]
        if kind == "first" or r >= 99.99:
            medians.append(99.97)
            continue
        if r >= 1.5:
            s = 0.25
        e_mult = 1.0 + (n_mult - 1.0) * s
        m = r * e_mult / n_mult
        medians.append(clamp(m, 0.01, min(ceiling, 99.99)))
    return medians


def tier_prior_median(trophy):
    """Last-resort median when a case carries no usable history at all."""
    tier = trophy.get("tier") or ""
    kind, _, _ = trophy_class(trophy)
    if kind == "first":
        return 99.97
    return {"platinum": 0.05, "gold": 0.10, "silver": 0.30}.get(tier, 1.0)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    horizon = case["horizon_minutes"]
    d_days = horizon / 1440.0
    try:
        target = parse_time(case["target"])
    except (KeyError, TypeError, ValueError):
        target = None
    if target is not None:
        x_target = math.log1p(hours_since_release(target))
    else:
        at = parse_time(case["at"])
        x_target = math.log1p(hours_since_release(at) + horizon / 60.0)

    snaps = distinct_snapshots(case)
    n_snaps = len(snaps)
    rates = last_rates(case)

    if n_snaps >= 3:
        medians, sigma_fits, _ = fitted_medians(case, snaps, x_target)
        sigmas = []
        for i, trophy in enumerate(case["trophies"]):
            kind, _, _ = trophy_class(trophy)
            if kind == "first":
                sigmas.append(0.35)
                continue
            extra = sigma_fits[i] if sigma_fits[i] is not None else 0.9
            step = SIGMA_STEP_PER_REFRESH * math.sqrt(
                max(horizon / 60.0, 0.0) / REFRESH_HOURS)
            var = (SIGMA_BASE * math.sqrt(d_days)) ** 2 + extra ** 2 + step ** 2
            sigmas.append(min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(var))))
    elif n_snaps == 2:
        fit_med, sigma_fits, _ = fitted_medians(case, snaps, x_target)
        pri_med = prior_medians(case, horizon)
        medians = []
        sigmas = []
        for i, trophy in enumerate(case["trophies"]):
            kind, _, _ = trophy_class(trophy)
            if kind == "first":
                medians.append(99.97)
                sigmas.append(0.35)
                continue
            m = math.exp(0.5 * (logit(fit_med[i]) + logit(pri_med[i])))
            medians.append(clamp(m))
            sigmas.append(0.9)
    elif n_snaps == 1:
        medians = prior_medians(case, horizon)
        sigmas = [0.35 if trophy_class(t)[0] == "first" else FLAT_SIGMA
                  for t in case["trophies"]]
    else:
        medians = [tier_prior_median(t) for t in case["trophies"]]
        sigmas = [1.2] * len(case["trophies"])

    # Trophy pinned at ~100%: keep it there whatever the path said.
    for i, trophy in enumerate(case["trophies"]):
        if trophy_class(trophy)[0] == "first":
            medians[i] = 99.97 if rates[i] >= 99.99 else clamp(medians[i], 99.0, 99.99)
            sigmas[i] = 0.35

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        m = medians[i]
        s = sigmas[i]
        lg = logit(m)
        qs = []
        for z in Z:
            q = inv_logit(lg + z * s)
            qs.append(round4(clamp(q, 0.01, 99.99)))
        # Rounding can only tie neighbours; enforce non-decreasing.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
