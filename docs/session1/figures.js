/* The live figures for session 1. Each one mirrors a cell of session1.ipynb. */

import {
  makeRng, generateSchedule, rlModel, rlWeights,
  multiplicativeUtility, additiveUtility, reversalSchedule,
} from '../assets/models.js';
import { interactiveFigure, layout, CONFIG, COL } from '../assets/deck.js';

/** Resolved per call: a module-level capture would run before Plotly loads. */
function plotly() {
  if (!window.Plotly) {
    throw new Error('Plotly failed to load, so the live figures cannot be drawn. '
                    + 'Check the connection to cdn.plot.ly.');
  }
  return window.Plotly;
}
const trials = (n) => Array.from({ length: n }, (_, i) => i);

/* --- 1. a fixed 80% schedule, and what the learner makes of it ------------ */

function scheduleTrace(opt1Rewarded, trueProbability, probOpt1) {
  const x = trials(opt1Rewarded.length);
  return [
    { x, y: opt1Rewarded, mode: 'markers', name: 'outcome',
      marker: { size: 5, color: COL.faint } },
    { x, y: trueProbability, mode: 'lines', name: 'true probability',
      line: { color: COL.ink, dash: 'dash', width: 1.5 } },
    { x, y: Array.from(probOpt1), mode: 'lines', name: 'model estimate',
      line: { color: COL.opt1, width: 2.5 } },
  ];
}

const fixedSchedule = {
  'fig-schedule': {
    params: {
      alpha:        { min: 0.01, max: 1, step: 0.01, value: 0.05, label: '&alpha;' },
      startingProb: { min: 0, max: 1, step: 0.01, value: 0.5, label: '<i>p</i><sub>1</sub>' },
      trueProb:     { min: 0.5, max: 1, step: 0.05, value: 0.8, label: 'true <i>p</i>' },
    },
    draw(el, v) {
      const n = 200;
      const rng = makeRng(v.__seed ?? 12345);
      const trueProbability = Array(n).fill(v.trueProb);
      const opt1Rewarded = generateSchedule(trueProbability, rng);
      const probOpt1 = rlModel(opt1Rewarded, v.alpha, v.startingProb);
      plotly().react(el, scheduleTrace(opt1Rewarded, trueProbability, probOpt1), layout({
        width: 1080, height: 330,
        xaxis: Object.assign(layout().xaxis, { title: 'trial', range: [0, n] }),
        yaxis: Object.assign(layout().yaxis, { title: 'P(option 1 rewarded)', range: [-0.05, 1.05] }),
      }), CONFIG);
    },
    note: (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)}, a starting estimate of `
                 + `${v.startingProb.toFixed(2)} and a true reward probability of ${v.trueProb.toFixed(2)}. `
                 + `Grey dots are the trial outcomes, the dashed line the true probability, `
                 + `the green line the model's running estimate.`,
  },
};

/* --- 2. the same learner on a reversing schedule ------------------------- */

const reversal = {
  'fig-reversal': {
    params: {
      alpha: { min: 0.01, max: 1, step: 0.01, value: 0.15, label: '&alpha;' },
    },
    draw(el, v) {
      const rng = makeRng(v.__seed ?? 7);
      const trueProbability = reversalSchedule();
      const opt1Rewarded = generateSchedule(trueProbability, rng);
      const probOpt1 = rlModel(opt1Rewarded, v.alpha, 0.5);
      plotly().react(el, scheduleTrace(opt1Rewarded, trueProbability, probOpt1), layout({
        width: 1080, height: 330,
        xaxis: Object.assign(layout().xaxis, { title: 'trial', range: [0, trueProbability.length] }),
        yaxis: Object.assign(layout().yaxis, { title: 'P(option 1 rewarded)', range: [-0.05, 1.05] }),
        shapes: [{ type: 'rect', xref: 'x', yref: 'paper', x0: 0, x1: 100, y0: 0, y1: 1,
                   fillcolor: '#000', opacity: 0.035, line: { width: 0 }, layer: 'below' }],
        annotations: [
          { x: 50,  y: 1.06, xref: 'x', yref: 'paper', text: 'reversals every 25 trials', showarrow: false,
            font: { size: 12, color: COL.muted } },
          { x: 150, y: 1.06, xref: 'x', yref: 'paper', text: 'stable', showarrow: false,
            font: { size: 12, color: COL.muted } },
        ],
      }), CONFIG);
    },
    note: (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)}. The first 100 trials reverse `
                 + `every 25 trials; the last 100 are stable.`,
  },
};

/* --- 3 & 4. the two utility functions ------------------------------------ */

const SAMPLES = 60;
const magRange  = Array.from({ length: SAMPLES }, (_, i) => 1 + (99 * i) / (SAMPLES - 1));
const probRange = Array.from({ length: SAMPLES }, (_, i) => i / (SAMPLES - 1));

/**
 * Drawn as a contour map rather than a 3D surface. It needs no WebGL, so it
 * survives any lecture-room machine and prints to PDF, and the contrast the
 * slide is making reads more directly: multiplicative utility has curved
 * contours pinned to both axes, additive utility has straight parallel ones.
 */
function contour(el, z, label) {
  plotly().react(el, [{
    type: 'contour', z, x: probRange, y: magRange,
    colorscale: [[0, '#f4f7f4'], [0.5, '#9cc49e'], [1, COL.opt1]],
    showscale: false,
    contours: { coloring: 'heatmap', start: 0, end: 100, size: 10,
                showlabels: true, labelfont: { size: 10, color: '#2a2f33' } },
    line: { color: 'rgba(255,255,255,0.75)', width: 1 },
    hovertemplate: 'probability %{x:.2f}<br>magnitude %{y:.0f}<br>utility %{z:.1f}<extra></extra>',
  }], layout({
    width: 540, height: 310,
    margin: { l: 52, r: 10, t: 10, b: 42 },
    xaxis: Object.assign(layout().xaxis, { title: 'probability', dtick: 0.25 }),
    yaxis: Object.assign(layout().yaxis, { title: 'magnitude', showgrid: false }),
  }), CONFIG);
  el.setAttribute('aria-label', label);
}

const utilities = {
  'fig-utility-mult': {
    params: {},
    draw(el) {
      const z = magRange.map((m) => probRange.map((p) => multiplicativeUtility(m, p)));
      contour(el, z, 'contour map of multiplicative utility');
    },
    note: () => 'Contours of utility over probability and magnitude. They curve away from '
                + 'both axes: utility is near zero whenever either term is near zero, so a '
                + 'large magnitude cannot rescue a near-impossible reward.',
  },
  'fig-utility-add': {
    params: { omega: { min: 0, max: 1, step: 0.01, value: 0.5, label: '&omega;' } },
    draw(el, v) {
      const z = magRange.map((m) => probRange.map((p) => additiveUtility(m, p, v.omega)));
      contour(el, z, 'contour map of additive utility');
    },
    note: (v) => `Shown with a magnitude weight of ${v.omega.toFixed(2)}. The contours are `
                 + `straight and parallel, so a shortfall on one dimension can always be `
                 + `traded against a gain on the other. The slope of the contours is what `
                 + `\u03c9 controls.`,
  },
};

/* --- 5. the learning rate as a weighting of past outcomes ---------------- */

const weights = {
  'fig-weights': {
    params: { alpha: { min: 0.02, max: 1, step: 0.01, value: 0.15, label: '&alpha;' } },
    draw(el, v) {
      const nBack = 20;
      const w = rlWeights(v.alpha, nBack);
      const x = Array.from({ length: nBack }, (_, k) => -(k + 1));
      plotly().react(el, [{
        x, y: w, type: 'bar', marker: { color: COL.opt1 },
        hovertemplate: '%{x} trials back<br>weight %{y:.3f}<extra></extra>',
      }], layout({
        width: 740, height: 300,
        bargap: 0.35,
        xaxis: Object.assign(layout().xaxis, { title: 'trials into the past', dtick: 5 }),
        yaxis: Object.assign(layout().yaxis, { title: 'weight', range: [0, 1.02] }),
      }), CONFIG);
    },
    note: (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)}: the most recent outcome carries `
                 + `weight ${v.alpha.toFixed(2)}, and each earlier one is discounted by a further factor `
                 + `of ${(1 - v.alpha).toFixed(2)}.`,
  },
};

/* --- wire them all up ---------------------------------------------------- */

for (const group of [fixedSchedule, reversal, utilities, weights]) {
  for (const [id, spec] of Object.entries(group)) {
    interactiveFigure(id, spec.params, spec.draw, spec.note);
  }
}
