/* The live figures for session 4.

   Most of the session revisits week 3, so most of these are the same figures
   under new ids, drawn by ../assets/figures-shared.js and ../assets/figures-fits.js.
   The one new figure is the parameter recovery the session 4 notebook runs. */

import { layout, CONFIG, COL } from '../assets/deck.js';
import {
  plotly, axis, registerAll, drawChoices, choicesNote, drawCondition, conditionNote,
  drawMagnitudes, magnitudesNote, drawLikelihood, likelihoodNote, drawGridSearch,
  gridSearchNote, drawAscent, ascentNote,
} from '../assets/figures-shared.js';
import {
  alphaDistributions, alphaDifference, drawChiSquare, chiSquareNote, drawBic, bicNote,
  parameterMatrix, MUL_LABELS, ADD_LABELS, drawModelRecovery, modelRecoveryNote,
} from '../assets/figures-fits.js';
import { FITS, SIMULATIONS, STATS, PARAMETER_RECOVERY } from '../assets/participant-fits.js';

const N = STATS.nParticipants;

/* --- the recap, straight from week 3 -------------------------------------- */

const recap = {
  'fig-likelihood': {
    params: {},
    draw: (el) => drawLikelihood(el, { width: 700, height: 255 }),
    note: () => likelihoodNote,
  },
  'fig-grid': {
    params: { points: { min: 4, max: 24, step: 1, value: 10, label: 'grid',
                        format: (v) => `${v}×${v}` } },
    draw: (el, v) => drawGridSearch(el, v, { width: 600, height: 340 }),
    note: gridSearchNote,
  },
  'fig-ascent': { params: {}, draw: drawAscent, note: () => ascentNote },
  'fig-alpha-dist': {
    params: {},
    draw: (el) => alphaDistributions(el, FITS.mul2),
    note: () => `Learning rates fitted to all ${N} participants under multiplicative utility, `
                + 'block by block.',
  },
  'fig-alpha-diff': {
    params: {},
    draw: (el) => alphaDifference(el, FITS.mul2),
    note: () => {
      const t = STATS.tTest.mul;
      return `The same fits as a within participant difference: t(${t.df}) = ${t.t}, `
             + `p = ${t.p.toPrecision(2)}.`;
    },
  },
  'fig-chi2': { params: {}, draw: drawChiSquare, note: chiSquareNote },
  'fig-bic': { params: {}, draw: drawBic, note: bicNote },
  'fig-model-recovery': { params: {}, draw: drawModelRecovery, note: modelRecoveryNote },
  'fig-params-mul': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.mul2, MUL_LABELS, { width: 1060, height: 330 }),
    note: () => 'Every fitted parameter against every other, under multiplicative utility. Points '
                + 'pinned at 0 or 1 are the ones to look at.',
  },
  'fig-params-add': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.add2, ADD_LABELS),
    note: () => 'The same under additive utility, with ω as a fourth parameter.',
  },
};

const simulations = Object.fromEntries(Object.entries(SIMULATIONS).map(([key, sim]) => [
  `fig-sim-${key}`, {
    params: {},
    draw: (el) => alphaDifference(el, sim, { width: 540, height: 290 }),
    note: () => `Simulated ${key === 'same' ? 'unshuffled' : 'with the volatile learning rates '
      + 'shuffled between participants'}: χ²(${sim.test.df}) = ${sim.test.lambda}, `
      + `p = ${sim.test.p.toPrecision(2)}.`,
  }]));

/* --- plots worth putting in the report ------------------------------------ */

const plotsForTheReport = {
  'fig-choices': {
    params: {
      alpha: { min: 0.01, max: 1, step: 0.01, value: 0.3, label: '&alpha;' },
      beta:  { min: 0.01, max: 0.5, step: 0.01, value: 0.05, label: '&beta;' },
    },
    draw: (el, v) => drawChoices(el, v, { width: 1080, height: 330 }),
    note: choicesNote,
  },
  'fig-magnitudes': {
    params: {},
    draw: (el) => drawMagnitudes(el, { width: 1080, height: 170 }),
    note: () => magnitudesNote,
  },
};

Object.assign(plotsForTheReport,
  Object.fromEntries([1, 2].map((condition) => [`fig-condition-${condition}`, {
    params: {},
    draw: (el) => drawCondition(el, condition, { width: 530, height: 215 }),
    note: () => conditionNote(condition),
  }])));

/* The same three plots again, larger, on the slides about what to put in the
   report: there each one is the whole slide rather than half of it. */
Object.assign(plotsForTheReport, {
  'fig-alpha-dist-2': {
    params: {},
    draw: (el) => alphaDistributions(el, FITS.mul2, { width: 900, height: 390 }),
    note: () => `Fitted learning rates for all ${N} participants, by block.`,
  },
  'fig-alpha-diff-2': {
    params: {},
    draw: (el) => alphaDifference(el, FITS.mul2, { width: 900, height: 360 }),
    note: () => {
      const t = STATS.tTest.mul;
      return `The within participant difference: t(${t.df}) = ${t.t}, p = ${t.p.toPrecision(2)}.`;
    },
  },
  'fig-params-mul-2': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.mul2, MUL_LABELS, { width: 1000, height: 430 }),
    note: () => 'Every fitted parameter against every other, under multiplicative utility.',
  },
});

/* --- parameter recovery, from the session 4 notebook's simulations -------- */

/** Slope and intercept of the least squares line through (x, y). */
function leastSquares(x, y) {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
  }
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}

function recoveryPanel(el, key, label, hi, { width = 520, height = 330 } = {}) {
  const { simulated, recovered } = PARAMETER_RECOVERY[key];
  const { slope, intercept } = leastSquares(simulated, recovered);
  plotly().react(el, [
    { x: [0, hi], y: [0, hi], mode: 'lines', name: 'simulated = recovered',
      line: { color: COL.ink, width: 1.2, dash: 'dash' }, hoverinfo: 'skip' },
    { x: [0, hi], y: [intercept, intercept + slope * hi], mode: 'lines', name: 'best fit',
      line: { color: COL.opt2, width: 2 }, hoverinfo: 'skip' },
    { x: simulated, y: recovered, mode: 'markers',
      marker: { size: 5, color: COL.opt1, opacity: 0.45 },
      hovertemplate: 'simulated %{x:.3f}<br>recovered %{y:.3f}<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis(`simulated ${label}`, { range: [0, hi], constrain: 'domain' }),
    yaxis: axis(`recovered ${label}`, { range: [0, hi], scaleanchor: 'x', scaleratio: 1,
                                        constrain: 'domain' }),
    annotations: [
      { x: 0.04, y: 0.94, xref: 'paper', yref: 'paper', xanchor: 'left', showarrow: false,
        text: `slope ${slope.toFixed(2)}`, font: { size: 15, color: COL.opt2 } },
    ],
  }), CONFIG);
}

const RECOVERY_PANELS = {
  'fig-recovery-alpha': { key: 'alphaVolatile', label: 'α volatile', hi: 1 },
  'fig-recovery-beta': { key: 'beta', label: 'β', hi: 0.6 },
};

const recovery = Object.fromEntries(Object.entries(RECOVERY_PANELS).map(([id, p]) => [id, {
  params: {},
  draw: (el) => recoveryPanel(el, p.key, p.label, p.hi),
  note: () => `Simulated against recovered ${p.label}, for four simulated datasets of ${N} `
              + 'participants each. The dashed line is simulated = recovered; the orange line is '
              + 'the least squares fit through the points. The closer its slope is to 1, and the '
              + 'tighter the scatter around it, the better this parameter recovers.',
}]));

registerAll(recap, simulations, plotsForTheReport, recovery);
