/* ---------------------------------------------------------------------------
   Just enough model fitting to run in a browser.

   The notebooks fit with L-BFGS through optax. Nothing here needs that: these
   figures fit a handful of simulated participants over a coarse grid, which is
   fast enough to run while a slide loads and makes the same points. Where a
   figure is about the shape of the likelihood rather than the precise estimate,
   a grid is arguably the more honest thing to show anyway.
--------------------------------------------------------------------------- */

import { rlModel, softmax, multiplicativeUtility } from './models.js';

/** Summed log likelihood of a set of choices under one alpha and beta. */
export function logLikelihood(opt1Rewarded, magOpt1, magOpt2, choice1, alpha, beta,
                              startingProb = 0.5) {
  const probOpt1 = rlModel(opt1Rewarded, alpha, startingProb);
  let ll = 0;
  for (let t = 0; t < choice1.length; t++) {
    const p = softmax(multiplicativeUtility(magOpt1[t], probOpt1[t]),
                      multiplicativeUtility(magOpt2[t], 1 - probOpt1[t]), beta);
    ll += choice1[t] === 1 ? Math.log(p) : Math.log(1 - p);
  }
  return ll;
}

const range = (from, to, n) =>
  Array.from({ length: n }, (_, i) => from + ((to - from) * i) / (n - 1));

/**
 * Best fitting alpha and beta by grid search, refined once around the winner
 * so the estimate is not pinned to the coarse grid.
 */
export function fitAlphaBeta(data, { alphaN = 24, betaN = 24 } = {}) {
  const { opt1Rewarded, magOpt1, magOpt2, choice1 } = data;
  const search = (aLo, aHi, bLo, bHi) => {
    let best = { ll: -Infinity, alpha: 0, beta: 0 };
    for (const alpha of range(aLo, aHi, alphaN)) {
      for (const beta of range(bLo, bHi, betaN)) {
        const ll = logLikelihood(opt1Rewarded, magOpt1, magOpt2, choice1, alpha, beta);
        if (ll > best.ll) best = { ll, alpha, beta };
      }
    }
    return best;
  };
  const coarse = search(0.01, 1, 0.005, 0.5);
  const da = 1 / alphaN;
  const db = 0.5 / betaN;
  return search(Math.max(0.005, coarse.alpha - da), Math.min(1, coarse.alpha + da),
                Math.max(0.002, coarse.beta - db), Math.min(0.6, coarse.beta + db));
}

/** Best fitting alpha alone, with beta held at a known value. */
export function fitAlpha(data, beta, n = 60) {
  const { opt1Rewarded, magOpt1, magOpt2, choice1 } = data;
  let best = { ll: -Infinity, alpha: 0 };
  for (const alpha of range(0.01, 1, n)) {
    const ll = logLikelihood(opt1Rewarded, magOpt1, magOpt2, choice1, alpha, beta);
    if (ll > best.ll) best = { ll, alpha };
  }
  return best.alpha;
}

/**
 * Walk uphill on the log likelihood from a starting point, by finite
 * differences. Returns the path, which is the thing worth drawing.
 *
 * Two details keep the picture honest. Beta is rescaled so that a step means
 * the same thing along both axes -- without that the walk crabs sideways,
 * because the likelihood changes far faster with beta than with alpha. And
 * each step is halved until it actually improves the likelihood, so every step
 * drawn really does go uphill, while steps stay long on steep ground and
 * shorten as the surface flattens.
 */
export function gradientAscent(data, start, { steps = 80, rate = 0.0012, maxStep = 0.045,
                                              betaScale = 0.4 } = {}) {
  const { opt1Rewarded, magOpt1, magOpt2, choice1 } = data;
  const ll = (a, b) => logLikelihood(opt1Rewarded, magOpt1, magOpt2, choice1, a, b);
  const clampA = (v) => Math.min(0.99, Math.max(0.01, v));
  const clampB = (v) => Math.min(0.5, Math.max(0.005, v));
  const h = 0.004;

  let alpha = start.alpha;
  let beta = start.beta;
  let current = ll(alpha, beta);
  const path = [{ alpha, beta }];

  for (let i = 0; i < steps; i++) {
    const dA = (ll(clampA(alpha + h), beta) - ll(clampA(alpha - h), beta)) / (2 * h);
    const dB = (ll(alpha, clampB(beta + h)) - ll(alpha, clampB(beta - h))) / (2 * h) * betaScale;
    const length = Math.hypot(dA, dB);
    if (length < 1e-6) break;

    let t = rate * length > maxStep ? maxStep / length : rate;
    let moved = false;
    for (let k = 0; k < 10; k++) {
      const next = { alpha: clampA(alpha + t * dA), beta: clampB(beta + t * dB * betaScale) };
      const value = ll(next.alpha, next.beta);
      if (value > current + 1e-6) {
        ({ alpha, beta } = next);
        current = value;
        path.push({ alpha, beta });
        moved = true;
        break;
      }
      t /= 2;
    }
    if (!moved) break;
  }
  return path;
}
