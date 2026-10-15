/* The live figures for session 2. Each one mirrors a cell of session2.ipynb. */

import {
  makeRng, generateSchedule, generateMagnitudes, rlModel, softmax,
  multiplicativeUtility, additiveUtility, simulateRlModel, stableThenVolatile,
} from '../assets/models.js';
import { interactiveFigure, layout, CONFIG, COL } from '../assets/deck.js';

function plotly() {
  if (!window.Plotly) {
    throw new Error('Plotly failed to load, so the live figures cannot be drawn. '
                    + 'Check the connection to cdn.plot.ly.');
  }
  return window.Plotly;
}
const trials = (n) => Array.from({ length: n }, (_, i) => i + 1);

/* --- the softmax ---------------------------------------------------------- */

const softmaxFigure = {
  'fig-softmax': {
    params: { beta: { min: 0.01, max: 1, step: 0.01, value: 0.2, label: '&beta;' } },
    draw(el, v) {
      const d = Array.from({ length: 201 }, (_, i) => -100 + i);
      plotly().react(el, [
        { x: d, y: d.map((u) => (u > 0 ? 1 : u < 0 ? 0 : 0.5)), mode: 'lines',
          name: 'deterministic', line: { color: COL.muted, dash: 'dot', width: 2 },
          hovertemplate: 'deterministic<extra></extra>' },
        { x: d, y: d.map((u) => softmax(u, 0, v.beta)), mode: 'lines',
          name: 'softmax', line: { color: COL.opt1, width: 3 },
          hovertemplate: 'utility difference %{x}<br>P(choose option 1) %{y:.2f}<extra></extra>' },
      ], layout({
        width: 1080, height: 390,
        xaxis: Object.assign(layout().xaxis, {
          title: { text: 'utility of option 1 − utility of option 2', standoff: 12 },
          automargin: true, zeroline: true, zerolinecolor: '#e2e6ea' }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'P(choose option 1)', standoff: 12 }, automargin: true,
          range: [-0.04, 1.2], tickvals: [0, 0.25, 0.5, 0.75, 1] }),
        annotations: [
          { x: -98, y: 1.13, xanchor: 'left', showarrow: false, text: '··· deterministic rule',
            font: { size: 16, color: COL.muted } },
          { x: -30, y: 1.13, xanchor: 'left', showarrow: false, text: '— softmax',
            font: { size: 16, color: COL.opt1 } },
        ],
      }), CONFIG);
    },
    note: (v) => `Shown with an inverse temperature of ${v.beta.toFixed(2)}. The dotted line is `
                 + `the deterministic rule, which always takes the better option; the green curve `
                 + `is the softmax, which takes it most of the time. A larger β makes the curve `
                 + `steeper, so behaviour approaches the deterministic rule.`,
  },
};

/* --- the model's choice probabilities, trial by trial --------------------- */

const schedule = stableThenVolatile();
const baseRng = makeRng(2024);
const SCHEDULE = {
  trueProbability: schedule,
  opt1Rewarded: generateSchedule(schedule, baseRng),
  magOpt1: generateMagnitudes(schedule.length, baseRng),
  magOpt2: generateMagnitudes(schedule.length, baseRng),
};

const choiceFigure = {
  'fig-choices': {
    params: {
      alpha: { min: 0.01, max: 1, step: 0.01, value: 0.3, label: '&alpha;' },
      beta:  { min: 0.01, max: 0.5, step: 0.01, value: 0.05, label: '&beta;' },
    },
    draw(el, v) {
      const { trueProbability, opt1Rewarded, magOpt1, magOpt2 } = SCHEDULE;
      const rng = makeRng(77);
      const { probOpt1, choiceProb1, choice1 } =
        simulateRlModel(opt1Rewarded, magOpt1, magOpt2, v.alpha, v.beta, { rng });
      const x = trials(trueProbability.length);
      plotly().react(el, [
        { x, y: trueProbability, mode: 'lines', name: 'true probability',
          line: { color: COL.ink, dash: 'dash', width: 1.6 },
          hovertemplate: 'true probability %{y:.2f}<extra></extra>' },
        { x, y: Array.from(probOpt1), mode: 'lines', name: 'learnt probability',
          line: { color: COL.opt1, width: 2.6 },
          hovertemplate: 'trial %{x}<br>learnt probability %{y:.2f}<extra></extra>' },
        { x, y: Array.from(choiceProb1), mode: 'markers', name: 'choice probability',
          marker: { size: 4, color: '#1f4b73', symbol: 'x', opacity: 0.8 },
          hovertemplate: 'trial %{x}<br>P(choose option 1) %{y:.2f}<extra></extra>' },
        { x, y: Array.from(choice1), mode: 'markers', name: 'simulated choice',
          marker: { size: 4, color: COL.opt2, opacity: 0.55 },
          hovertemplate: 'trial %{x}<br>chose option %{y}<extra></extra>' },
      ], layout({
        width: 1120, height: 345,
        xaxis: Object.assign(layout().xaxis, { title: { text: 'trial', standoff: 14 }, range: [0.5, x.length + 0.5] }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'probability', standoff: 12 }, automargin: true,
          tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1], range: [-0.06, 1.44] }),
        annotations: [
          { x: 1, y: 1.3, xanchor: 'left', showarrow: false, text: '– – true probability',
            font: { size: 16, color: COL.ink } },
          { x: 42, y: 1.3, xanchor: 'left', showarrow: false, text: '— learnt probability',
            font: { size: 16, color: COL.opt1 } },
          { x: 92, y: 1.3, xanchor: 'left', showarrow: false, text: '× choice probability',
            font: { size: 16, color: '#1f4b73' } },
          { x: 143, y: 1.3, xanchor: 'left', showarrow: false, text: '● simulated choice',
            font: { size: 16, color: COL.opt2 } },
        ],
      }), CONFIG);
    },
    note: (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)} and an inverse temperature of `
                 + `${v.beta.toFixed(2)}. The learnt probability tracks the true one; the choice `
                 + `probability also reflects the points on offer, which is why it reaches further `
                 + `than the learnt probability does.`,
  },
};

/* --- the likelihood landscape -------------------------------------------- */

/** The summed log likelihood of a set of choices under one alpha and beta. */
function logLikelihood(opt1Rewarded, magOpt1, magOpt2, choice1, alpha, beta) {
  const probOpt1 = rlModel(opt1Rewarded, alpha, 0.5);
  let ll = 0;
  for (let t = 0; t < choice1.length; t++) {
    const p = softmax(multiplicativeUtility(magOpt1[t], probOpt1[t]),
                      multiplicativeUtility(magOpt2[t], 1 - probOpt1[t]), beta);
    ll += choice1[t] === 1 ? Math.log(p) : Math.log(1 - p);
  }
  return ll;
}

const TRUE_ALPHA = 0.3;
const TRUE_BETA = 0.2;

const landscape = {
  'fig-likelihood': {
    params: {},
    draw(el) {
      const rng = makeRng(5150);
      const tp = stableThenVolatile();
      const opt1Rewarded = generateSchedule(tp, rng);
      const magOpt1 = generateMagnitudes(tp.length, rng);
      const magOpt2 = generateMagnitudes(tp.length, rng);
      const { choice1 } = simulateRlModel(opt1Rewarded, magOpt1, magOpt2, TRUE_ALPHA, TRUE_BETA,
                                          { rng: makeRng(99) });

      const alphas = Array.from({ length: 60 }, (_, i) => 0.01 + (0.79 * i) / 59);
      const betas = Array.from({ length: 60 }, (_, i) => 0.01 + (0.49 * i) / 59);
      const z = alphas.map((a) => betas.map((b) =>
        logLikelihood(opt1Rewarded, magOpt1, magOpt2, choice1, a, b)));

      // the peak, which is what the fitting algorithm is looking for
      let best = { ll: -Infinity, a: 0, b: 0 };
      z.forEach((row, i) => row.forEach((ll, j) => {
        if (ll > best.ll) best = { ll, a: alphas[i], b: betas[j] };
      }));

      plotly().react(el, [
        { type: 'contour', z, x: betas, y: alphas,
          colorscale: [[0, '#f4f7f4'], [0.55, '#9cc49e'], [1, COL.opt1]],
          showscale: true,
          colorbar: { title: { text: 'log likelihood', side: 'right' }, thickness: 11,
                      len: 0.92, outlinewidth: 0, tickfont: { size: 13 }, titlefont: { size: 14 } },
          contours: { coloring: 'heatmap' },
          line: { color: 'rgba(255,255,255,0.6)', width: 1 },
          hovertemplate: 'α %{y:.2f}<br>β %{x:.2f}<br>log likelihood %{z:.1f}<extra></extra>' },
        { x: [TRUE_BETA], y: [TRUE_ALPHA], mode: 'markers', name: 'simulated',
          marker: { size: 13, color: '#fff', symbol: 'circle', line: { color: COL.ink, width: 2 } },
          hovertemplate: 'simulated with α 0.30, β 0.20<extra></extra>' },
        { x: [best.b], y: [best.a], mode: 'markers', name: 'peak',
          marker: { size: 13, color: COL.opt2, symbol: 'x', line: { width: 3 } },
          hovertemplate: `peak at α ${best.a.toFixed(2)}, β ${best.b.toFixed(2)}<extra></extra>` },
      ], layout({
        width: 980, height: 430,
        xaxis: Object.assign(layout().xaxis, {
          title: { text: 'inverse temperature β', standoff: 12 }, automargin: true, showgrid: false }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'learning rate α', standoff: 12 }, automargin: true, showgrid: false }),
        annotations: [
          { x: TRUE_BETA, y: TRUE_ALPHA, ax: 36, ay: -30, text: 'simulated with',
            font: { size: 14, color: COL.ink }, arrowcolor: COL.ink, arrowhead: 2,
            arrowsize: 0.8, arrowwidth: 1, xanchor: 'left' },
          { x: best.b, y: best.a, ax: 30, ay: 34, text: 'best fit',
            font: { size: 14, color: COL.opt2 }, arrowcolor: COL.opt2, arrowhead: 2,
            arrowsize: 0.8, arrowwidth: 1, xanchor: 'left' },
        ],
      }), CONFIG);
    },
    note: () => `The log likelihood of one simulated participant's choices, over every pairing of `
                + `a learning rate and an inverse temperature. The data were simulated with `
                + `α = 0.30 and β = 0.20; the peak of the landscape is what fitting looks `
                + `for. The ridge running across it is why α and β trade off against each other.`,
  },
};

/* --- wire them up --------------------------------------------------------- */

for (const group of [softmaxFigure, choiceFigure, landscape]) {
  for (const [id, spec] of Object.entries(group)) {
    interactiveFigure(id, spec.params, spec.draw, spec.note);
  }
}
