/* The live figures for session 3.

   The plots recapping weeks 1 and 2 are drawn by ../assets/figures-shared.js,
   and the ones about the real participants by ../assets/figures-fits.js, since
   session 4 shows both sets again. What is here is which figure goes on which
   slide, at what size, and what its caption says in the PDF. */

import {
  registerAll, drawChoices, choicesNote, drawCondition, drawMagnitudes, SCHEDULE_MAGNITUDES,
  drawWeights, weightsNote, drawGridSearch, gridSearchNote, drawRecovery, recoveryNote,
  drawFalsification, drawAscent, ascentNote, ASCENT_STEPS, drawRegression, regressionNote,
  drawErrorTerm, errorTermNote, drawLikelihoodSpike, likelihoodSpikeNote, drawFitComplexity,
  fitComplexityNote, drawMiniLearning, drawMiniSoftmax,
} from '../assets/figures-shared.js';
import {
  alphaDistributions, alphaDifference, drawChiSquare, chiSquareNote, drawBic, bicNote,
  parameterMatrix, MUL_LABELS, ADD_LABELS, drawModelFrequencies, modelFrequenciesNote,
} from '../assets/figures-fits.js';
import { FITS, SIMULATIONS, STATS, REGRESSION } from '../assets/participant-fits.js';

const N = STATS.nParticipants;

/* --- recaps of weeks 1 and 2 ---------------------------------------------- */

const recaps = {
  'fig-choices': {
    params: {
      alpha: { min: 0.01, max: 1, step: 0.01, value: 0.3, label: '&alpha;' },
      beta:  { min: 0.01, max: 0.5, step: 0.01, value: 0.05, label: '&beta;' },
    },
    draw: (el, v) => drawChoices(el, v, { width: 1120, height: 300 }),
    note: choicesNote,
  },
  'fig-magnitudes': {
    params: {},
    draw: (el) => drawMagnitudes(el, { width: 1120, height: 140 }),
    note: () => 'Points on offer, redrawn at random every trial, in both conditions alike.',
  },
  // the magnitudes of the same schedule the choices plot above simulates
  'fig-magnitudes-2': {
    params: {},
    draw: (el) => drawMagnitudes(el, { width: 1120, height: 140, magnitudes: SCHEDULE_MAGNITUDES }),
    note: () => 'The magnitudes on offer in the same schedule, redrawn at random every trial.',
  },
  'fig-weights': {
    // starts at the learning rate the regression beside it was simulated with
    params: { alpha: { min: 0.02, max: 1, step: 0.01, value: REGRESSION.alpha, label: '&alpha;' } },
    draw: (el, v) => drawWeights(el, v, { width: 560, height: 270 }),
    note: weightsNote,
  },
  'fig-regression': {
    params: {},
    draw: (el) => drawRegression(el, REGRESSION, { width: 560, height: 270 }),
    note: () => regressionNote(REGRESSION),
  },
  'fig-error-term': {
    params: {},
    draw: (el) => drawErrorTerm(el, { width: 540, height: 260 }),
    note: () => errorTermNote,
  },
  'fig-likelihood-spike': {
    params: {},
    draw: (el) => drawLikelihoodSpike(el, { width: 420, height: 320 }),
    note: () => likelihoodSpikeNote,
  },
  'fig-fit-complexity': {
    params: {},
    draw: (el) => drawFitComplexity(el, { width: 1100, height: 220 }),
    note: () => fitComplexityNote,
  },
  'fig-mini-learning': { params: {}, draw: (el) => drawMiniLearning(el), note: () => '' },
  'fig-mini-softmax': { params: {}, draw: (el) => drawMiniSoftmax(el), note: () => '' },
  'fig-recovery-mini': {
    params: {},
    draw: (el) => drawRecovery(el, 'alpha', 'fitAlpha', '\u03b1', 0.72, { width: 340, height: 290 }),
    note: () => recoveryNote('Learning rate'),
  },
  'fig-grid': {
    params: { points: { min: 4, max: 24, step: 1, value: 10, label: 'grid',
                        format: (v) => `${v}×${v}` } },
    draw: (el, v) => drawGridSearch(el, v, { width: 600, height: 340 }),
    note: gridSearchNote,
  },
  'fig-ascent': {
    params: { step: { min: 1, max: ASCENT_STEPS, step: 1, value: ASCENT_STEPS,
                      label: 'step', format: (v) => String(Math.round(v)) } },
    draw: (el, v) => drawAscent(el, v, { width: 690, height: 320 }),
    note: ascentNote,
    opts: { play: { param: 'step', fps: 14 } },
  },
  'fig-recovery-alpha': {
    params: {},
    draw: (el) => drawRecovery(el, 'alpha', 'fitAlpha', 'α', 0.72, { width: 560, height: 360 }),
    note: () => recoveryNote('Learning rate'),
  },
  'fig-falsify-same': {
    params: {},
    draw: (el) => drawFalsification(el, 'same', { width: 520, height: 215 }),
    note: () => 'Simulated with one learning rate for both blocks: the two recovered distributions '
                + 'sit on top of each other.',
  },
  'fig-falsify-diff': {
    params: {},
    draw: (el) => drawFalsification(el, 'different', { width: 520, height: 215 }),
    note: () => 'Simulated with a higher learning rate in the volatile block: the difference comes '
                + 'back out.',
  },
};

Object.assign(recaps, Object.fromEntries([1, 2].map((condition) => [`fig-condition-${condition}`, {
  params: {},
  draw: (el) => drawCondition(el, condition, { width: 530, height: 195 }),
  note: () => (condition === 1
    ? 'Stable: P(option 1 rewarded) holds at 0.75 throughout.'
    : 'Volatile: it starts at 0.75 and reverses every 25 trials.'),
}])));

/* --- the fits to the real participants ------------------------------------ */

const realFits = {
  'fig-alpha-dist': {
    params: {},
    draw: (el) => alphaDistributions(el, FITS.mul2, { width: 560, height: 280 }),
    note: () => `Learning rates fitted to all ${N} participants under multiplicative utility. The `
                + 'volatile distribution sits to the right, but this plot compares the two groups '
                + 'rather than each participant with themselves.',
  },
  'fig-alpha-diff': {
    params: {},
    draw: (el) => alphaDifference(el, FITS.mul2, { width: 560, height: 280 }),
    note: () => {
      const t = STATS.tTest.mul;
      return 'The same fits as a within participant difference, which is what the hypothesis is '
             + `about. A one sample t test against zero gives t(${t.df}) = ${t.t}, `
             + `p = ${t.p.toPrecision(2)}.`;
    },
  },
  'fig-chi2': { params: {}, draw: (el) => drawChiSquare(el, { width: 1020, height: 200 }), note: chiSquareNote },
  'fig-bic': { params: {}, draw: drawBic, note: bicNote },
  'fig-params-mul': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.mul2, MUL_LABELS, { width: 600, height: 470 }),
    note: () => 'Every fitted parameter against every other, under multiplicative utility. Points '
                + 'pinned at 0 or 1 are the ones to look at: a learning rate of 1 means only the '
                + 'last outcome counted, and a β of 0 means choices were random.',
  },
  'fig-params-add': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.add2, ADD_LABELS),
    note: () => 'All 75 participants, fitted with two learning rates under additive utility.',
  },
  'fig-model-frequencies': {
    params: {},
    draw: (el) => drawModelFrequencies(el, { width: 700, height: 290 }),
    note: modelFrequenciesNote,
  },
};

/* --- the two control simulations ------------------------------------------ */

const simulations = Object.fromEntries(Object.entries(SIMULATIONS).map(([key, sim]) => [
  `fig-sim-${key}`, {
    params: {},
    draw: (el) => alphaDifference(el, sim, { width: 540, height: 230 }),
    note: () => 'Recovered differences when the data were simulated '
                + (key === 'same'
                  ? 'with the same learning rate in both blocks. '
                  : 'with the same learning rates, but shuffled between participants in the '
                    + 'volatile block. ')
                + `The likelihood ratio test gives χ²(${sim.test.df}) = `
                + `${sim.test.lambda}, p = ${sim.test.p.toPrecision(2)}.`,
  }]));

registerAll(recaps, realFits, simulations);
