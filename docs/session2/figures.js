/* The live figures for session 2. Each one mirrors a cell of session2.ipynb. */

import {
  makeRng, generateSchedule, generateMagnitudes, rlModel, softmax,
  multiplicativeUtility, additiveUtility, simulateRlModel, stableThenVolatile,
} from '../assets/models.js';
import { interactiveFigure, layout, CONFIG, COL } from '../assets/deck.js';
import { logLikelihood } from '../assets/fitting.js';
import { RECOVERY, FALSIFICATION, ASCENT } from './precomputed.js';

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

      // choices carry the option colours used throughout: green when option 1
      // was taken, orange when option 2 was
      const chose = (want) => {
        const xs = [], ys = [];
        choice1.forEach((c, i) => { if (c === want) { xs.push(x[i]); ys.push(c); } });
        return { x: xs, y: ys };
      };

      plotly().react(el, [
        { x, y: trueProbability, mode: 'lines', name: 'true probability',
          line: { color: COL.ink, dash: 'dash', width: 1.8 },
          hovertemplate: 'true probability %{y:.2f}<extra></extra>' },
        { x, y: Array.from(probOpt1), mode: 'lines', name: 'learnt probability',
          line: { color: COL.opt1, width: 3 },
          hovertemplate: 'trial %{x}<br>learnt probability %{y:.2f}<extra></extra>' },
        { x, y: Array.from(choiceProb1), mode: 'lines', name: 'choice probability',
          line: { color: COL.opt1, width: 1.6, dash: 'dot' },
          hovertemplate: 'trial %{x}<br>P(choose option 1) %{y:.2f}<extra></extra>' },
        { ...chose(1), mode: 'markers', name: 'chose option 1',
          marker: { size: 6, color: COL.opt1 },
          hovertemplate: 'trial %{x}<br>chose option 1<extra></extra>' },
        { ...chose(0), mode: 'markers', name: 'chose option 2',
          marker: { size: 6, color: COL.opt2 },
          hovertemplate: 'trial %{x}<br>chose option 2<extra></extra>' },
      ], layout({
        width: 1120, height: 345,
        xaxis: Object.assign(layout().xaxis, {
          title: { text: 'trial', standoff: 14 }, range: [0.5, x.length + 0.5] }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'probability', standoff: 14 },
          tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1], range: [-0.06, 1.46] }),
        annotations: [
          { x: 1, y: 1.33, xanchor: 'left', showarrow: false,
            text: '\u2013\u2009\u2013\u2009true probability', font: { size: 16, color: COL.ink } },
          { x: 42, y: 1.33, xanchor: 'left', showarrow: false,
            text: '\u2014\u2009learnt probability', font: { size: 16, color: COL.opt1 } },
          { x: 92, y: 1.33, xanchor: 'left', showarrow: false,
            text: '\u00b7\u00b7\u2009choice probability', font: { size: 16, color: COL.opt1 } },
          { x: 1, y: 1.13, xanchor: 'left', showarrow: false,
            text: '\u25cf\u2009chose option 1', font: { size: 16, color: COL.opt1 } },
          { x: 42, y: 1.13, xanchor: 'left', showarrow: false,
            text: '\u25cf\u2009chose option 2', font: { size: 16, color: COL.opt2 } },
        ],
      }), CONFIG);
    },
    note: (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)} and an inverse temperature of `
                 + `${v.beta.toFixed(2)}. The learnt probability tracks the true one; the dotted `
                 + `choice probability also reflects the points on offer, which is why it reaches `
                 + `further. Dots at 1 and 0 are the simulated choices.`,
  },
};

/* --- the two conditions, and the points on offer ------------------------- */

const CONDITIONS = { 'fig-condition-1': 1, 'fig-condition-2': 2 };

function conditionSchedule(condition) {
  const n = 102;
  const pOption1 = condition === 1
    ? new Array(n).fill(0.75)
    : [...new Array(28).fill(0.75), ...new Array(25).fill(0.25),
       ...new Array(25).fill(0.75), ...new Array(25).fill(0.25)].slice(0, n);
  const rng = makeRng(condition === 1 ? 404 : 909);
  return { pOption1, opt1Rewarded: generateSchedule(pOption1, rng) };
}

const conditions = Object.fromEntries(Object.entries(CONDITIONS).map(([id, condition]) => [id, {
  params: {},
  draw(el) {
    const { pOption1, opt1Rewarded } = conditionSchedule(condition);
    const x = trials(pOption1.length);
    const pick = (want) => {
      const xs = [], ys = [];
      opt1Rewarded.forEach((o, i) => { if (o === want) { xs.push(x[i]); ys.push(o); } });
      return { x: xs, y: ys };
    };
    plotly().react(el, [
      { ...pick(1), mode: 'markers', name: 'option 1 rewarded',
        marker: { size: 5, color: COL.opt1 },
        hovertemplate: 'trial %{x}<br>option 1 rewarded<extra></extra>' },
      { ...pick(0), mode: 'markers', name: 'option 2 rewarded',
        marker: { size: 5, color: COL.opt2 },
        hovertemplate: 'trial %{x}<br>option 2 rewarded<extra></extra>' },
      { x, y: pOption1, mode: 'lines', name: 'true probability',
        line: { color: COL.ink, dash: 'dash', width: 2 },
        hovertemplate: 'P(option 1 rewarded) %{y:.2f}<extra></extra>' },
    ], layout({
      width: 530, height: 215,
      xaxis: Object.assign(layout().xaxis, {
        title: { text: 'trial', standoff: 14 }, range: [0.5, pOption1.length + 0.5] }),
      yaxis: Object.assign(layout().yaxis, {
        title: { text: 'P(option 1 rewarded)', standoff: 14 }, range: [-0.08, 1.08], dtick: 0.25 }),
    }), CONFIG);
  },
  note: () => condition === 1
    ? 'P(option 1 rewarded) holds at 0.75 for all 102 trials: the stable block.'
    : 'P(option 1 rewarded) starts at 0.75 and reverses every 25 trials: the volatile block.',
}]));

const magnitudes = {
  'fig-magnitudes': {
    params: {},
    draw(el) {
      const n = 102;
      const rng = makeRng(31415);
      const x = trials(n);
      const mag = () => Array.from({ length: n }, () => 1 + Math.floor(rng() * 99));
      plotly().react(el, [
        { x, y: mag(), mode: 'lines+markers', name: 'option 1',
          line: { color: COL.opt1, width: 1.6 }, marker: { size: 4, color: COL.opt1 },
          hovertemplate: 'trial %{x}<br>option 1: %{y} points<extra></extra>' },
        { x, y: mag(), mode: 'lines+markers', name: 'option 2',
          line: { color: COL.opt2, width: 1.6, dash: 'dot' }, marker: { size: 4, color: COL.opt2 },
          hovertemplate: 'trial %{x}<br>option 2: %{y} points<extra></extra>' },
      ], layout({
        width: 1120, height: 165,
        xaxis: Object.assign(layout().xaxis, {
          title: { text: 'trial', standoff: 14 }, range: [0.5, n + 0.5] }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'points on offer', standoff: 14 }, range: [0, 104], dtick: 50 }),
        annotations: [
          { x: 1, y: 1.24, xref: 'x', yref: 'paper', xanchor: 'left', showarrow: false,
            text: '\u2014\u2009option 1', font: { size: 16, color: COL.opt1 } },
          { x: 18, y: 1.24, xref: 'x', yref: 'paper', xanchor: 'left', showarrow: false,
            text: '\u00b7\u00b7\u00b7\u2009option 2', font: { size: 16, color: COL.opt2 } },
        ],
      }), CONFIG);
    },
    note: () => 'Points on offer, redrawn at random every trial and independently of which option '
                + 'is rewarded. Identical in both conditions.',
  },
};

/* --- the likelihood landscape -------------------------------------------- */


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

/* --- parameter recovery (fitted offline, see precompute-session2.mjs) ----- */

function recoveryPanel(el, key, fitKey, label, hi) {
  plotly().react(el, [
    { x: [0, hi], y: [0, hi], mode: 'lines', line: { color: COL.ink, width: 1.2, dash: 'dash' },
      hoverinfo: 'skip' },
    { x: RECOVERY.map((r) => r[key]), y: RECOVERY.map((r) => r[fitKey]), mode: 'markers',
      marker: { size: 9, color: COL.opt1, opacity: 0.75 },
      hovertemplate: 'simulated %{x:.3f}<br>recovered %{y:.3f}<extra></extra>' },
  ], layout({
    width: 500, height: 340,
    xaxis: Object.assign(layout().xaxis, {
      title: { text: `simulated ${label}`, standoff: 14 }, range: [0, hi],
      constrain: 'domain' }),
    yaxis: Object.assign(layout().yaxis, {
      title: { text: `recovered ${label}`, standoff: 14 }, range: [0, hi],
      scaleanchor: 'x', scaleratio: 1, constrain: 'domain' }),
  }), CONFIG);
}

const recovery = {
  'fig-recovery-alpha-2': {
    params: {},
    draw: (el) => recoveryPanel(el, 'alpha', 'fitAlpha', '\u03b1', 0.72),
    note: () => 'Simulated against recovered learning rate, for 36 simulated participants.',
  },
  'fig-recovery-alpha': {
    params: {},
    draw: (el) => recoveryPanel(el, 'alpha', 'fitAlpha', '\u03b1', 0.72),
    note: () => 'Learning rate: 36 simulated participants, each fitted back with the same model. '
                + 'The dashed line is simulated = recovered.',
  },
  'fig-recovery-beta': {
    params: {},
    draw: (el) => recoveryPanel(el, 'beta', 'fitBeta', '\u03b2', 0.32),
    note: () => 'Inverse temperature, the same 36 participants. It recovers less tightly than the '
                + 'learning rate, which is what the ridge in the likelihood landscape predicts.',
  },
};

/* --- falsification -------------------------------------------------------- */

function falsificationPanel(el, which) {
  const { stable, volatile } = FALSIFICATION[which];
  const bins = { start: 0, end: 0.8, size: 0.05 };
  plotly().react(el, [
    { x: stable, type: 'histogram', xbins: bins,
      marker: { color: '#1f4b73', opacity: 0.55 },
      hovertemplate: 'stable block<br>%{y} participants<extra></extra>' },
    { x: volatile, type: 'histogram', xbins: bins,
      marker: { color: COL.opt2, opacity: 0.55 },
      hovertemplate: 'volatile block<br>%{y} participants<extra></extra>' },
  ], layout({
    width: 520, height: 300,
    barmode: 'overlay',
    xaxis: Object.assign(layout().xaxis, {
      title: { text: 'recovered learning rate', standoff: 14 }, range: [0, 0.8] }),
    yaxis: Object.assign(layout().yaxis, { title: { text: 'participants', standoff: 14 } }),
    annotations: [
      { x: 0, y: 1.18, xref: 'paper', yref: 'paper', xanchor: 'left', showarrow: false,
        text: '\u25a0\u2009stable block', font: { size: 16, color: '#1f4b73' } },
      { x: 0.42, y: 1.18, xref: 'paper', yref: 'paper', xanchor: 'left', showarrow: false,
        text: '\u25a0\u2009volatile block', font: { size: 16, color: COL.opt2 } },
    ],
  }), CONFIG);
}

const falsification = {
  'fig-falsify-same-2': {
    params: {}, draw: (el) => falsificationPanel(el, 'same'),
    note: () => 'Simulated with one learning rate for both blocks.',
  },
  'fig-falsify-diff-2': {
    params: {}, draw: (el) => falsificationPanel(el, 'different'),
    note: () => 'Simulated with a higher learning rate in the volatile block.',
  },
  'fig-falsify-same': {
    params: {},
    draw: (el) => falsificationPanel(el, 'same'),
    note: () => 'Simulated with one learning rate for both blocks: the two recovered distributions '
                + 'sit on top of each other, as they should.',
  },
  'fig-falsify-diff': {
    params: {},
    draw: (el) => falsificationPanel(el, 'different'),
    note: () => 'Simulated with a higher learning rate in the volatile block: the difference comes '
                + 'back out, so the task can find it when it is really there.',
  },
};

/* --- gradient ascent on the landscape ------------------------------------- */

const ascent = {
  'fig-ascent': {
    params: {},
    draw(el) {
      const { data, truth, paths } = ASCENT;
      const alphas = Array.from({ length: 46 }, (_, i) => 0.01 + (0.88 * i) / 45);
      const betas = Array.from({ length: 46 }, (_, i) => 0.01 + (0.39 * i) / 45);
      const z = alphas.map((a) => betas.map((b) =>
        logLikelihood(data.opt1Rewarded, data.magOpt1, data.magOpt2, data.choice1, a, b)));
      const top = Math.max(...z.flat());
      plotly().react(el, [
        { type: 'contour', z, x: betas, y: alphas,
          zmin: top - 70, zmax: top,
          colorscale: [[0, '#ffffff'], [0.45, '#dcebdc'], [0.8, '#a8cbaa'], [1, COL.opt1]],
          showscale: false, contours: { coloring: 'heatmap', start: top - 70, end: top, size: 7 },
          line: { color: 'rgba(255,255,255,0.7)', width: 1 },
          hovertemplate: '\u03b1 %{y:.2f}<br>\u03b2 %{x:.2f}<br>log likelihood %{z:.0f}<extra></extra>' },
        ...paths.map((path) => ({
          x: path.map((p) => p.beta), y: path.map((p) => p.alpha), mode: 'lines+markers',
          line: { color: COL.ink, width: 2 }, marker: { size: 5, color: COL.ink },
          hovertemplate: 'step %{pointNumber}<br>\u03b1 %{y:.2f}, \u03b2 %{x:.2f}<extra></extra>',
        })),
        { x: [truth.beta], y: [truth.alpha], mode: 'markers',
          marker: { size: 14, color: '#fff', symbol: 'circle', line: { color: COL.ink, width: 2 } },
          hovertemplate: 'simulated with these values<extra></extra>' },
      ], layout({
        width: 900, height: 390,
        xaxis: Object.assign(layout().xaxis, {
          title: { text: 'inverse temperature \u03b2', standoff: 14 }, showgrid: false }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'learning rate \u03b1', standoff: 14 }, showgrid: false }),
        annotations: paths.map((path, i) => ({
          x: path[0].beta, y: path[0].alpha, ax: i === 0 ? -30 : 44, ay: i === 0 ? 30 : -4,
          text: `start ${i + 1}`, font: { size: 15, color: COL.ink }, arrowcolor: COL.ink,
          arrowhead: 2, arrowsize: 0.8, arrowwidth: 1,
        })).concat([{ x: truth.beta, y: truth.alpha, ax: 40, ay: -34, text: 'simulated with',
          font: { size: 15, color: COL.ink }, arrowcolor: COL.ink, arrowhead: 2,
          arrowsize: 0.8, arrowwidth: 1, xanchor: 'left' }]),
      }), CONFIG);
    },
    note: () => 'Two runs of gradient ascent on one likelihood surface, from two different starting '
                + 'points. Every step goes uphill; the steps are long where the surface is steep and '
                + 'shorten as it flattens. Both end near the values the data were simulated with.',
  },
};

/* --- wire them up --------------------------------------------------------- */

for (const group of [softmaxFigure, choiceFigure, conditions, magnitudes, landscape,
                     recovery, falsification, ascent]) {
  for (const [id, spec] of Object.entries(group)) {
    interactiveFigure(id, spec.params, spec.draw, spec.note);
  }
}
