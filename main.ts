// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Forecast for every GTA VI (PS5) trophy its earn rate on Exophase at the
// case's target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).
//
// Model, per trophy (written fresh for this entry):
//   * Build the series of DISTINCT reads (Exophase refreshes rarely, so most
//     reads repeat the previous page verbatim; deduping keeps the fit honest).
//   * If the last 24h of distinct reads show movement, fit a weighted least
//     squares line to logit(rate) against ln(1 + hours since release), with
//     recency weights (12h half-life). Extrapolate the line to the target,
//     anchored at the fitted value at the last read. The slope is shrunk
//     toward 0 when the fit is noisy (residual SD), and clamped to [-4, 4]
//     logit per e-fold. The median is capped by a tier prior (platinum is
//     slow: <=3%; gold <=25%; silver <=40%; bronze <=90%) and by how far a
//     rate may plausibly move from its last read.
//   * If history is flat (no movement: launch night, before the first
//     refresh), there is no trend to fit. The next refresh dilutes rates:
//     players_tracked jumps (launch night: 638 -> 3,868) while most new
//     players have not earned anything yet. The median is shifted down by a
//     dilution factor with a tier-dependent new-player earn fraction
//     (bronze 0.35, silver 0.15, gold/platinum 0.05), and the spread is wide
//     so the refresh lands inside the band. The one trophy at 100% stayed
//     100% through the first refresh, so it is forecast near 100% with a
//     downward tail instead of being diluted.
//   * Spread combines the random-walk term (0.5 * sqrt(days), as the house
//     baseline), the fit's residual SD, and a slope-uncertainty term that
//     grows with the extrapolation distance; flat history adds a wide base
//     spread. Quantiles are logistic(median_logit + z_tau * spread).
//
// Deterministic: no randomness, no time dependence, fixed iteration order.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at the levels above. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant: the origin for the log-time trend. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Random-walk spread, logit units per sqrt(day) (house baseline's value). */
const SIGMA_RW = 0.5;
/** Base spread for flat history, logit units: covers an unseen refresh. */
const SIGMA_FLAT = 1.1;
/** Recency half-life for the trend fit, hours. */
const HALF_LIFE_H = 12;
/** Trend window: distinct reads from the last 24h before the checkpoint. */
const WINDOW_H = 24;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clampPct(pct) / (100 - clampPct(pct)));
const logistic = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Hours from release to an instant. */
const hoursSinceRelease = (iso: string): number =>
  (Date.parse(iso) - RELEASE_MS) / 3_600_000;
/** Log-time coordinate; guarded so pre-release reads stay defined. */
const logTime = (tHours: number): number => Math.log(Math.max(0.05, 1 + tHours));

/** Tier ceiling for the forecast median, in percent. */
const tierCap = (tier: string): number =>
  tier === 'platinum' ? 3 : tier === 'gold' ? 25 : tier === 'silver' ? 40 : 90;
/** New-player earn fraction for the dilution shift (flat history). */
const tierEarnFrac = (tier: string): number =>
  tier === 'platinum' || tier === 'gold' ? 0.05 : tier === 'silver' ? 0.15 : 0.35;

/** Weighted least squares y ~ a + b*x. Returns null when degenerate. */
function wls(
  xs: number[],
  ys: number[],
  ws: number[],
): { a: number; b: number; residSd: number } | null {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = ws[i];
    sw += w;
    sx += w * xs[i];
    sy += w * ys[i];
    sxx += w * xs[i] * xs[i];
    sxy += w * xs[i] * ys[i];
  }
  const denom = sw * sxx - sx * sx;
  if (sw <= 0 || Math.abs(denom) < 1e-12) return null;
  const b = (sw * sxy - sx * sy) / denom;
  const a = (sy - b * sx) / sw;
  let rss = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - (a + b * xs[i]);
    rss += ws[i] * r * r;
  }
  return { a, b, residSd: Math.sqrt(rss / sw) };
}

/** One trophy's forecast: median logit and spread in logit units. */
function forecastTrophy(
  tier: string,
  series: Array<{ t: number; n: number; rate: number }>,
  tLast: number,
  tTarget: number,
  horizonDays: number,
): { mu: number; s: number } {
  const last = series[series.length - 1];
  const lastRate = last.rate;
  const lastLogit = logit(lastRate);
  const xLast = logTime(tLast);
  const xTarget = logTime(tTarget);

  // Trophy every tracked player already has: it survived the first refresh
  // at 100%, so forecast it near 100% with a downward tail, not diluted.
  if (lastRate >= 99.9) return { mu: logit(99), s: 1.4 };

  // Distinct reads in the trend window (else all distinct reads).
  const windowed = series.filter((p) => p.t >= tLast - WINDOW_H);
  const pts = windowed.length >= 3 ? windowed : series;
  const distinctRates = new Set(pts.map((p) => p.rate));

  if (pts.length >= 3 && distinctRates.size >= 2) {
    const xs = pts.map((p) => logTime(p.t));
    const ys = pts.map((p) => logit(p.rate));
    const ws = pts.map((p) => Math.pow(0.5, (tLast - p.t) / HALF_LIFE_H));
    const fit = wls(xs, ys, ws);
    if (fit !== null) {
      // Shrink a noisy slope toward 0; clamp what remains.
      const shrink = 1 / (1 + Math.pow(fit.residSd / 0.5, 2));
      const bUse = Math.max(-4, Math.min(4, fit.b * shrink));
      let mu = fit.a + fit.b * xLast + bUse * (xTarget - xLast);
      // Tier ceiling and plausibility guardrails on the move from last.
      mu = Math.min(mu, logit(tierCap(tier)));
      mu = Math.max(mu, lastLogit - 4.5);
      mu = Math.min(mu, lastLogit + 5);
      const sRw = SIGMA_RW * Math.sqrt(horizonDays);
      const sRes = Math.min(1.5, Math.max(0.2, fit.residSd));
      const sSlope = 0.35 * Math.abs(bUse) * (xTarget - xLast);
      const s = Math.min(3, Math.max(0.4, Math.sqrt(sRw * sRw + sRes * sRes + sSlope * sSlope)));
      return { mu, s };
    }
  }

  // Flat history: no trend. Shift the median down for refresh dilution.
  // Estimate the players-tracked growth factor to the target from its own
  // trend when it moved; else a launch-night prior (first refresh ~6x in
  // a few hours, slower later).
  let g = horizonDays <= 0.25 ? 4 : 2;
  const nFirst = series[0].n;
  const nLast = last.n;
  if (nLast > nFirst * 1.05) {
    const xs = series.map((p) => logTime(p.t));
    const ys = series.map((p) => Math.log(Math.max(1, p.n)));
    const ws = series.map((p) => Math.pow(0.5, (tLast - p.t) / HALF_LIFE_H));
    const nFit = wls(xs, ys, ws);
    if (nFit !== null) {
      const nPred = Math.exp(nFit.a + nFit.b * xTarget);
      g = Math.max(1, Math.min(50, nPred / Math.max(1, nLast)));
    }
  }
  const f = tierEarnFrac(tier);
  const diluted = lastRate * ((1 + f * (g - 1)) / g);
  const mu = Math.min(logit(tierCap(tier)), logit(Math.max(0.01, diluted)));
  const sRw = SIGMA_RW * Math.sqrt(horizonDays);
  const s = Math.min(3, Math.max(0.4, Math.sqrt(sRw * sRw + SIGMA_FLAT * SIGMA_FLAT)));
  return { mu, s };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const tTarget = hoursSinceRelease(c.target);
  const horizonDays = c.horizon_minutes / 1440;

  // Distinct snapshots, oldest first: collapse consecutive reads whose
  // rates and players-tracked are unchanged (unrefreshed page).
  const distinct: Read[] = [];
  for (const r of c.history) {
    const prev = distinct[distinct.length - 1];
    if (
      prev !== undefined &&
      prev.players_tracked === r.players_tracked &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      continue;
    }
    distinct.push(r);
  }
  const tLast = hoursSinceRelease(distinct[distinct.length - 1].at);

  const trophies = c.trophies.map((t, i) => {
    // Per-trophy series over the distinct snapshots (skip null rates).
    const series = distinct
      .map((r) => ({
        t: hoursSinceRelease(r.at),
        n: r.players_tracked,
        rate: r.rates[i],
      }))
      .filter((p) => p.rate !== null && Number.isFinite(p.rate)) as Array<{
      t: number;
      n: number;
      rate: number;
    }>;
    let mu: number;
    let s: number;
    if (series.length === 0) {
      // Trophy never observed: wide, low forecast.
      mu = logit(0.5);
      s = 2;
    } else {
      ({ mu, s } = forecastTrophy(t.tier, series, tLast, tTarget, horizonDays));
    }
    // Quantiles, non-decreasing by construction; clamp after rounding.
    let prev = -Infinity;
    const q = Z.map((z) => {
      const v = round4(clampPct(logistic(mu + z * s)));
      const out = Math.max(prev, v);
      prev = out;
      return out;
    });
    return { id: t.id, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
