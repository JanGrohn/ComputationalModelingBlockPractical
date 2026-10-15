/* ---------------------------------------------------------------------------
   The models from the Colab notebooks, in JavaScript.

   These are deliberately line-for-line equivalents of the Python in
   sessionN/fitting.py and the notebooks, so that a figure on a slide and the
   same figure in the notebook cannot drift apart. Option 1 is the green
   stimulus, option 2 the orange one.
--------------------------------------------------------------------------- */

/** Seeded RNG (mulberry32), so a slide looks the same every time it is shown. */
export function makeRng(seed = 12345) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 1 if option 1 is rewarded on a trial, 0 if option 2 is. */
export function generateSchedule(trueProbability, rng) {
  return Array.from(trueProbability, (p) => (rng() < p ? 1 : 0));
}

/** Reward points between 1 and 100, drawn independently of the probabilities. */
export function generateMagnitudes(nTrials, rng) {
  return Array.from({ length: nTrials }, () => 1 + Math.floor(rng() * 100));
}

/**
 * Equations 1 and 2: the model's running estimate of how likely option 1 is
 * to be rewarded.
 *     delta_t   = o_t - p_t
 *     p_{t+1}   = p_t + alpha * delta_t
 */
export function rlModel(opt1Rewarded, alpha, startingProb = 0.5) {
  const n = opt1Rewarded.length;
  const probOpt1 = new Float64Array(n);
  probOpt1[0] = startingProb;
  for (let t = 0; t < n - 1; t++) {
    const delta = opt1Rewarded[t] - probOpt1[t];
    probOpt1[t + 1] = probOpt1[t] + alpha * delta;
  }
  return probOpt1;
}

/** Equation 3: utility as the product of magnitude and probability. */
export function multiplicativeUtility(mag, prob) {
  return mag * prob;
}

/** Equation 4: utility as a weighted sum, with magnitude rescaled to 0-1. */
export function additiveUtility(mag, prob, omega) {
  return (omega * (mag / 100) + (1 - omega) * prob) * 100;
}

/** Equation 5: the softmax, written in its numerically stable form. */
export function softmax(utility1, utility2, beta) {
  return 1 / (1 + Math.exp(-beta * (utility1 - utility2)));
}

/**
 * The full simulation: learning, utility, choice probability, and a choice.
 * Returns the trial-by-trial latents the notebook plots.
 */
export function simulateRlModel(opt1Rewarded, magOpt1, magOpt2, alpha, beta, {
  startingProb = 0.5,
  utility = multiplicativeUtility,
  omega = null,
  rng = null,
} = {}) {
  const n = opt1Rewarded.length;
  const probOpt1 = rlModel(opt1Rewarded, alpha, startingProb);
  const utility1 = new Float64Array(n);
  const utility2 = new Float64Array(n);
  const choiceProb1 = new Float64Array(n);
  const choice1 = rng ? new Int8Array(n) : null;

  for (let t = 0; t < n; t++) {
    utility1[t] = omega === null ? utility(magOpt1[t], probOpt1[t])
                                 : utility(magOpt1[t], probOpt1[t], omega);
    utility2[t] = omega === null ? utility(magOpt2[t], 1 - probOpt1[t])
                                 : utility(magOpt2[t], 1 - probOpt1[t], omega);
    choiceProb1[t] = softmax(utility1[t], utility2[t], beta);
    if (choice1) choice1[t] = choiceProb1[t] > rng() ? 1 : 0;
  }
  return { probOpt1, utility1, utility2, choiceProb1, choice1 };
}

/**
 * Equation 6: how much the outcome t trials ago still weighs on the current
 * estimate. Written with t counted back from the present, so index 0 is the
 * most recent trial and carries weight alpha.
 */
export function rlWeights(alpha, nBack = 20) {
  return Array.from({ length: nBack }, (_, k) => alpha * Math.pow(1 - alpha, k));
}

/** A stable block followed by a volatile one, as in the task. */
export function stableThenVolatile() {
  return [
    ...Array(100).fill(0.75),
    ...Array(25).fill(0.25),
    ...Array(25).fill(0.75),
    ...Array(25).fill(0.25),
    ...Array(25).fill(0.75),
  ];
}

/** The reversal schedule used in session 1. */
export function reversalSchedule() {
  return [
    ...Array(25).fill(0.25),
    ...Array(25).fill(0.75),
    ...Array(25).fill(0.25),
    ...Array(25).fill(0.75),
    ...Array(100).fill(0.25),
  ];
}
