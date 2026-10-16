/* The live figures for session 4.

   Most of the session revisits week 3, so most of these are the same figures
   under new ids, drawn by ../assets/figures-shared.js and ../assets/figures-fits.js.
   The new ones are what the session 4 notebook adds: one real participant's
   run through the task, and the parameter recovery. */

import { layout, CONFIG, COL } from '../assets/deck.js';
import {
  plotly, axis, registerAll, drawGridSearch, gridSearchNote, drawAscent, ascentNote,
  ASCENT_STEPS, drawErrorTerm, errorTermNote, drawFitComplexity, fitComplexityNote, correlation,
} from '../assets/figures-shared.js';
import {
  alphaDistributions, alphaDifference, drawChiSquare, chiSquareNote, drawBic, bicNote,
  parameterMatrix, MUL_LABELS, ADD_LABELS, drawModelRecovery, modelRecoveryNote,
  drawModelFrequencies, modelFrequenciesNote, drawParticipant, participantNote,
} from '../assets/figures-fits.js';
import { FITS, SIMULATIONS, STATS, PARAMETER_RECOVERY } from '../assets/participant-fits.js';

const N = STATS.nParticipants;

/* --- the recap, straight from week 3 -------------------------------------- */

const recap = {
  'fig-error-term': { params: {}, draw: (el) => drawErrorTerm(el, { width: 540, height: 260 }),
                      note: () => errorTermNote },
  'fig-fit-complexity': { params: {}, draw: (el) => drawFitComplexity(el, { width: 1100, height: 220 }),
                          note: () => fitComplexityNote },
  'fig-grid': {
    params: { points: { min: 4, max: 24, step: 1, value: 10, label: 'grid',
                        format: (v) => `${v}×${v}` } },
    draw: (el, v) => drawGridSearch(el, v, { width: 600, height: 340 }),
    note: gridSearchNote,
  },
  'fig-ascent': {
    params: { step: { min: 1, max: ASCENT_STEPS, step: 1, value: ASCENT_STEPS,
                      label: 'step', format: (v) => String(Math.round(v)) } },
    draw: (el, v) => drawAscent(el, v),
    note: ascentNote,
    opts: { play: { param: 'step', fps: 14 } },
  },
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
  'fig-model-frequencies': {
    params: {},
    draw: (el) => drawModelFrequencies(el, { width: 700, height: 450 }),
    note: modelFrequenciesNote,
  },
  'fig-model-recovery': {
    params: {},
    draw: (el) => drawModelRecovery(el, { width: 700, height: 360 }),
    note: modelRecoveryNote,
  },
  'fig-params-mul': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.mul2, MUL_LABELS, { width: 600, height: 470 }),
    note: () => 'Every fitted parameter against every other, under multiplicative utility. Points '
                + 'pinned at 0 or 1 are the ones to look at.',
  },
  'fig-params-add': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.add2, ADD_LABELS),
    note: () => 'All 75 participants, fitted with two learning rates under additive utility.',
  },
};

const simulations = Object.fromEntries(Object.entries(SIMULATIONS).map(([key, sim]) => [
  `fig-sim-${key}`, {
    params: {},
    draw: (el) => alphaDifference(el, sim, { width: 540, height: 240 }),
    note: () => `Simulated ${key === 'same' ? 'unshuffled' : 'with the volatile learning rates '
      + 'shuffled between participants'}: χ²(${sim.test.df}) = ${sim.test.lambda}, `
      + `p = ${sim.test.p.toPrecision(2)}.`,
  }]));

/* --- plots worth putting in the report ------------------------------------ */

const plotsForTheReport = {
  'fig-participant-schedule': {
    params: {},
    draw: (el) => drawParticipant(el, { withModel: false, width: 1120, height: 470 }),
    note: () => participantNote(false),
  },
  'fig-participant-model': {
    params: {},
    draw: (el) => drawParticipant(el, { withModel: true, width: 1120, height: 520 }),
    note: () => participantNote(true),
  },
};



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
    draw: (el) => parameterMatrix(el, FITS.mul2, MUL_LABELS, { width: 600, height: 470 }),
    note: () => 'All 75 participants, fitted with two learning rates under multiplicative utility.',
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

function recoveryPanel(el, key, label, hi, { width = 370, height = 330 } = {}) {
  const { simulated, recovered } = PARAMETER_RECOVERY[key];
  const { slope, intercept } = leastSquares(simulated, recovered);
  plotly().react(el, [
    { x: [0, hi], y: [0, hi], mode: 'lines', name: 'simulated = recovered',
      line: { color: COL.truth, width: 1.2, dash: 'dash' }, hoverinfo: 'skip' },
    { x: simulated, y: recovered, mode: 'markers',
      marker: { size: 4, color: COL.neutral, opacity: 0.3 },
      hovertemplate: 'simulated %{x:.3f}<br>recovered %{y:.3f}<extra></extra>' },
    { x: [0, hi], y: [intercept, intercept + slope * hi], mode: 'lines', name: 'best fit',
      line: { color: COL.estimate, width: 2.5 }, hoverinfo: 'skip' },
  ], layout({
    width, height,
    xaxis: axis(`simulated ${label}`, { range: [0, hi], constrain: 'domain' }),
    yaxis: axis(`recovered ${label}`, { range: [0, hi], scaleanchor: 'x', scaleratio: 1,
                                        constrain: 'domain' }),
    annotations: [
      { x: 0.04, y: 0.97, xref: 'x domain', yref: 'y domain', xanchor: 'left', yanchor: 'top',
        showarrow: false, align: 'left', font: { size: 15, color: COL.ink },
        text: `r = ${correlation(simulated, recovered).toFixed(2)}<br>`
          + `<span style="color:${COL.estimate}">slope ${slope.toFixed(2)}</span>` },
    ],
  }), CONFIG);
}

const RECOVERY_PANELS = {
  'fig-recovery-alpha-stable': { key: 'alphaStable', label: 'α stable', hi: 1 },
  'fig-recovery-alpha': { key: 'alphaVolatile', label: 'α volatile', hi: 1 },
  'fig-recovery-beta': { key: 'beta', label: 'β', hi: 0.6 },
};

const recovery = Object.fromEntries(Object.entries(RECOVERY_PANELS).map(([id, p]) => [id, {
  params: {},
  draw: (el) => recoveryPanel(el, p.key, p.label, p.hi),
  note: () => `Simulated against recovered ${p.label}, ${PARAMETER_RECOVERY.nDatasets} simulated `
              + `datasets of ${N} participants; dashed, simulated = recovered; purple, the least `
              + 'squares fit.',
}]));

registerAll(recap, simulations, plotsForTheReport, recovery);
