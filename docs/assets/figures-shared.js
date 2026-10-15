/* ---------------------------------------------------------------------------
   The figures more than one session shows.

   Sessions 3 and 4 spend much of their time recapping sessions 1 and 2, so
   they show the same plots again. Each session gives the figure its own
   element id and its own size; what is drawn lives here, once, so a change to
   a colour or an axis reaches every deck that shows it.
--------------------------------------------------------------------------- */

import {
  makeRng, generateSchedule, generateMagnitudes, rlWeights,
  simulateRlModel, stableThenVolatile,
} from './models.js';
import { interactiveFigure, layout, CONFIG, COL, CONTINUOUS } from './deck.js';
import { logLikelihood } from './fitting.js';
import { RECOVERY, FALSIFICATION, ASCENT } from './simulated-fits.js';

/** Resolved per call: a module-level capture would run before Plotly loads. */
export function plotly() {
  if (!window.Plotly) {
    throw new Error('Plotly failed to load, so the live figures cannot be drawn. '
                    + 'Check the connection to cdn.plot.ly.');
  }
  return window.Plotly;
}

/** Trial numbers starting at 1, matching the equations in the notebooks. */
export const trials = (n) => Array.from({ length: n }, (_, i) => i + 1);

/** Register every { id: { params, draw, note, opts } } entry in each group given. */
export function registerAll(...groups) {
  for (const group of groups) {
    for (const [id, spec] of Object.entries(group)) {
      interactiveFigure(id, spec.params, spec.draw, spec.note, spec.opts);
    }
  }
}

/** An axis with a title that cannot collide with its own tick labels. */
export const axis = (text, extra = {}) =>
  Object.assign(layout().xaxis, { title: { text, standoff: 14 } }, extra);

/** Split xs into the ones whose matching value equals `want`, and their ys. */
const where = (xs, values, want) => {
  const out = { x: [], y: [] };
  values.forEach((v, i) => { if (v === want) { out.x.push(xs[i]); out.y.push(v); } });
  return out;
};

/* --- the model's choice probabilities, trial by trial --------------------- */

const baseRng = makeRng(2024);
const trueProbability = stableThenVolatile();

/** One simulated participant's schedule, shared so every deck shows the same one. */
export const SCHEDULE = {
  trueProbability,
  opt1Rewarded: generateSchedule(trueProbability, baseRng),
  magOpt1: generateMagnitudes(trueProbability.length, baseRng),
  magOpt2: generateMagnitudes(trueProbability.length, baseRng),
};

export function drawChoices(el, v, { width = 1120, height = 345 } = {}) {
  const { opt1Rewarded, magOpt1, magOpt2 } = SCHEDULE;
  const { probOpt1, choiceProb1, choice1 } =
    simulateRlModel(opt1Rewarded, magOpt1, magOpt2, v.alpha, v.beta, { rng: makeRng(77) });
  const x = trials(trueProbability.length);

  plotly().react(el, [
    { x, y: trueProbability, mode: 'lines', name: 'true probability',
      line: { color: COL.truth, dash: 'dash', width: 1.8 },
      hovertemplate: 'true probability %{y:.2f}<extra></extra>' },
    { x, y: Array.from(probOpt1), mode: 'lines', name: 'RL model probability',
      line: { color: COL.option1, width: 3 },
      hovertemplate: 'trial %{x}<br>RL model probability %{y:.2f}<extra></extra>' },
    { x, y: Array.from(choiceProb1), mode: 'lines', name: 'choice probability',
      line: { color: COL.option1, width: 1.6, dash: 'dot' },
      hovertemplate: 'trial %{x}<br>P(choose option 1) %{y:.2f}<extra></extra>' },
    { ...where(x, Array.from(choice1), 1), mode: 'markers', name: 'simulated choice, option 1',
      marker: { size: 6, color: COL.option1 },
      hovertemplate: 'trial %{x}<br>simulated choice: option 1<extra></extra>' },
    { ...where(x, Array.from(choice1), 0), mode: 'markers', name: 'simulated choice, option 2',
      marker: { size: 6, color: COL.option2 },
      hovertemplate: 'trial %{x}<br>simulated choice: option 2<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis('trial number', { range: [0.5, x.length + 0.5] }),
    yaxis: axis('option 1 rewarded?', { tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1], range: [-0.06, 1.46] }),
    annotations: [
      { x: 1, y: 1.33, text: '– – true probability', color: COL.truth },
      { x: 42, y: 1.33, text: '— RL model probability', color: COL.option1 },
      { x: 92, y: 1.33, text: '·· choice probability', color: COL.option1 },
      { x: 1, y: 1.13, text: '● simulated choice, option 1', color: COL.option1 },
      { x: 70, y: 1.13, text: '● simulated choice, option 2', color: COL.option2 },
    ].map(({ x: ax, y, text, color }) =>
      ({ x: ax, y, xanchor: 'left', showarrow: false, text, font: { size: 16, color } })),
  }), CONFIG);
}

export const choicesNote = (v) =>
  `Shown with a learning rate of ${v.alpha.toFixed(2)} and an inverse temperature of `
  + `${v.beta.toFixed(2)}. The RL model probability tracks the true one; the dotted choice `
  + `probability also reflects the magnitudes on offer, which is why it reaches further. Dots at `
  + `1 and 0 are the simulated choices.`;

/* --- the two conditions, and the magnitudes on offer ---------------------- */

function conditionSchedule(condition) {
  const n = 102;
  const pOption1 = condition === 1
    ? new Array(n).fill(0.75)
    : [...new Array(28).fill(0.75), ...new Array(25).fill(0.25),
       ...new Array(25).fill(0.75), ...new Array(25).fill(0.25)].slice(0, n);
  return { pOption1, opt1Rewarded: generateSchedule(pOption1, makeRng(condition === 1 ? 404 : 909)) };
}

export function drawCondition(el, condition, { width = 530, height = 215 } = {}) {
  const { pOption1, opt1Rewarded } = conditionSchedule(condition);
  const x = trials(pOption1.length);
  plotly().react(el, [
    { ...where(x, Array.from(opt1Rewarded), 1), mode: 'markers', name: 'trial outcomes, option 1 rewarded',
      marker: { size: 5, color: COL.option1 },
      hovertemplate: 'trial %{x}<br>option 1 rewarded<extra></extra>' },
    { ...where(x, Array.from(opt1Rewarded), 0), mode: 'markers', name: 'trial outcomes, option 2 rewarded',
      marker: { size: 5, color: COL.option2 },
      hovertemplate: 'trial %{x}<br>option 2 rewarded<extra></extra>' },
    { x, y: pOption1, mode: 'lines', name: 'true probability',
      line: { color: COL.truth, dash: 'dash', width: 2 },
      hovertemplate: 'P(option 1 rewarded) %{y:.2f}<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis('trial number', { range: [0.5, pOption1.length + 0.5] }),
    yaxis: axis('option 1 rewarded?', { range: [-0.08, 1.08], dtick: 0.25 }),
  }), CONFIG);
}

export const conditionNote = (condition) => (condition === 1
  ? 'P(option 1 rewarded) holds at 0.75 for all 102 trials: the stable block.'
  : 'P(option 1 rewarded) starts at 0.75 and reverses every 25 trials: the volatile block.');

export function drawMagnitudes(el, { width = 1120, height = 165 } = {}) {
  const n = 102;
  const rng = makeRng(31415);
  const x = trials(n);
  const mag = () => Array.from({ length: n }, () => 1 + Math.floor(rng() * 99));
  plotly().react(el, [
    { x, y: mag(), mode: 'lines+markers', name: 'magnitude option 1',
      line: { color: COL.option1, width: 1.6 }, marker: { size: 4, color: COL.option1 },
      hovertemplate: 'trial %{x}<br>magnitude option 1: %{y}<extra></extra>' },
    { x, y: mag(), mode: 'lines+markers', name: 'magnitude option 2',
      line: { color: COL.option2, width: 1.6, dash: 'dot' }, marker: { size: 4, color: COL.option2 },
      hovertemplate: 'trial %{x}<br>magnitude option 2: %{y}<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis('trial number', { range: [0.5, n + 0.5] }),
    yaxis: axis('magnitude', { range: [0, 104], dtick: 50 }),
    annotations: [
      { x: 1, text: '— magnitude option 1', color: COL.option1 },
      { x: 30, text: '··· magnitude option 2', color: COL.option2 },
    ].map(({ x: ax, text, color }) => ({ x: ax, y: 1.24, xref: 'x', yref: 'paper',
      xanchor: 'left', showarrow: false, text, font: { size: 16, color } })),
  }), CONFIG);
}

export const magnitudesNote = 'Points on offer, redrawn at random every trial and independently '
  + 'of which option is rewarded. Identical in both conditions.';

/* --- the likelihood landscape --------------------------------------------- */

const TRUE_ALPHA = 0.3;
const TRUE_BETA = 0.2;

/** One simulated participant, and the log likelihood of their choices on a grid. */
function landscapeGrid() {
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

  let best = { ll: -Infinity, a: 0, b: 0 };
  z.forEach((row, i) => row.forEach((ll, j) => {
    if (ll > best.ll) best = { ll, a: alphas[i], b: betas[j] };
  }));
  return { alphas, betas, z, best };
}

export function drawLikelihood(el, { width = 980, height = 430 } = {}) {
  const { alphas, betas, z, best } = landscapeGrid();
  plotly().react(el, [
    { type: 'contour', z, x: betas, y: alphas,
      colorscale: CONTINUOUS,
      showscale: true,
      colorbar: { title: { text: 'log likelihood', side: 'right' }, thickness: 11,
                  len: 0.92, outlinewidth: 0, tickfont: { size: 13 }, titlefont: { size: 14 } },
      contours: { coloring: 'heatmap' },
      line: { color: 'rgba(255,255,255,0.6)', width: 1 },
      hovertemplate: 'α %{y:.2f}<br>β %{x:.2f}<br>log likelihood %{z:.1f}<extra></extra>' },
    { x: [TRUE_BETA], y: [TRUE_ALPHA], mode: 'markers', name: 'simulated',
      marker: { size: 13, color: '#fff', symbol: 'circle', line: { color: COL.truth, width: 2 } },
      hovertemplate: 'simulated with α 0.30, β 0.20<extra></extra>' },
    { x: [best.b], y: [best.a], mode: 'markers', name: 'peak',
      marker: { size: 13, color: COL.estimate, symbol: 'x', line: { width: 3 } },
      hovertemplate: `peak at α ${best.a.toFixed(2)}, β ${best.b.toFixed(2)}<extra></extra>` },
  ], layout({
    width, height,
    xaxis: axis('inverse temperature β', { showgrid: false }),
    yaxis: axis('learning rate α', { showgrid: false }),
    annotations: [
      { x: TRUE_BETA, y: TRUE_ALPHA, ax: 36, ay: -30, text: 'simulated with',
        font: { size: 14, color: COL.truth }, arrowcolor: COL.truth, arrowhead: 2,
        arrowsize: 0.8, arrowwidth: 1, xanchor: 'left' },
      { x: best.b, y: best.a, ax: 30, ay: 34, text: 'best fit',
        font: { size: 14, color: COL.estimate }, arrowcolor: COL.estimate, arrowhead: 2,
        arrowsize: 0.8, arrowwidth: 1, xanchor: 'left' },
    ],
  }), CONFIG);
}

export const likelihoodNote = "The log likelihood of one simulated participant's choices, over "
  + 'every pairing of a learning rate and an inverse temperature. The data were simulated with '
  + 'α = 0.30 and β = 0.20; the peak of the landscape is what fitting looks for. The ridge '
  + 'running across it is why α and β trade off against each other.';

/**
 * The same landscape with a grid search drawn over it, for the slide that asks
 * why we do not simply evaluate every combination.
 */
export function drawGridSearch(el, v, { width = 920, height = 400 } = {}) {
  const { alphas, betas, z, best } = landscapeGrid();
  const n = v.points;
  const gridA = Array.from({ length: n }, (_, i) => 0.02 + (0.76 * i) / (n - 1));
  const gridB = Array.from({ length: n }, (_, i) => 0.02 + (0.46 * i) / (n - 1));
  const points = { x: [], y: [] };
  for (const a of gridA) for (const b of gridB) { points.x.push(b); points.y.push(a); }

  plotly().react(el, [
    { type: 'contour', z, x: betas, y: alphas,
      colorscale: CONTINUOUS,
      showscale: false, contours: { coloring: 'heatmap' },
      line: { color: 'rgba(255,255,255,0.6)', width: 1 }, hoverinfo: 'skip' },
    { ...points, mode: 'markers', name: 'evaluated',
      marker: { size: 4, color: COL.neutral, opacity: 0.55 },
      hovertemplate: 'α %{y:.2f}, β %{x:.2f}<extra></extra>' },
    { x: [best.b], y: [best.a], mode: 'markers', name: 'peak',
      marker: { size: 13, color: COL.estimate, symbol: 'x', line: { width: 3 } },
      hovertemplate: 'the peak we are looking for<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis('inverse temperature β', { showgrid: false }),
    yaxis: axis('learning rate α', { showgrid: false }),
  }), CONFIG);
}

export const gridSearchNote = (v) => {
  const total = v.points ** 2;
  return `A ${v.points} by ${v.points} grid over two parameters is ${total} evaluations, almost `
    + `all of them far from the peak. With four parameters the same grid is `
    + `${total ** 2} evaluations.`;
};

/* --- parameter recovery, from simulated participants ---------------------- */

export function drawRecovery(el, key, fitKey, label, hi, { width = 500, height = 340 } = {}) {
  plotly().react(el, [
    { x: [0, hi], y: [0, hi], mode: 'lines', line: { color: COL.truth, width: 1.2, dash: 'dash' },
      hoverinfo: 'skip' },
    { x: RECOVERY.map((r) => r[key]), y: RECOVERY.map((r) => r[fitKey]), mode: 'markers',
      marker: { size: 9, color: COL.neutral, opacity: 0.75 },
      hovertemplate: 'simulated %{x:.3f}<br>recovered %{y:.3f}<extra></extra>' },
  ], layout({
    width, height,
    // constrain the domain rather than the range, so the box comes out square
    // without Plotly widening the axes past the values they are meant to show
    xaxis: axis(`simulated ${label}`, { range: [0, hi], constrain: 'domain' }),
    yaxis: axis(`recovered ${label}`, { range: [0, hi], scaleanchor: 'x', scaleratio: 1,
                                        constrain: 'domain' }),
  }), CONFIG);
}

/* --- model falsification -------------------------------------------------- */

export function drawFalsification(el, which, { width = 520, height = 300 } = {}) {
  const { stable, volatile } = FALSIFICATION[which];
  const xbins = { start: 0, end: 0.8, size: 0.05 };
  plotly().react(el, [
    { x: stable, type: 'histogram', xbins, marker: { color: COL.stable, opacity: 0.55 },
      hovertemplate: 'stable block<br>%{y} participants<extra></extra>' },
    { x: volatile, type: 'histogram', xbins, marker: { color: COL.volatile, opacity: 0.55 },
      hovertemplate: 'volatile block<br>%{y} participants<extra></extra>' },
  ], layout({
    width, height, barmode: 'overlay',
    xaxis: axis('recovered learning rate', { range: [0, 0.8] }),
    yaxis: axis('participants'),
    annotations: blockKey(),
  }), CONFIG);
}

/** The stable/volatile colour key, placed above a plot. */
export function blockKey(stableAt = 0, volatileAt = 0.42) {
  return [
    { x: stableAt, text: '■ stable block', color: COL.stable },
    { x: volatileAt, text: '■ volatile block', color: COL.volatile },
  ].map(({ x, text, color }) => ({ x, y: 1.18, xref: 'paper', yref: 'paper', xanchor: 'left',
    showarrow: false, text, font: { size: 16, color } }));
}

/* --- gradient ascent ------------------------------------------------------ */

/** How many steps the longer of the two ascent runs takes. */
export const ASCENT_STEPS = Math.max(...ASCENT.paths.map((p) => p.length));

/**
 * Two runs of gradient ascent on one likelihood surface.
 *
 * `v.step` is how far along to draw, so the slide can play the climb rather
 * than present it finished. At the last step both runs are complete, which is
 * what the default and the PDF export show.
 */
export function drawAscent(el, v = {}, { width = 900, height = 390 } = {}) {
  const { data, truth, paths } = ASCENT;
  const step = Math.round(v.step ?? ASCENT_STEPS);
  const alphas = Array.from({ length: 46 }, (_, i) => 0.01 + (0.88 * i) / 45);
  const betas = Array.from({ length: 46 }, (_, i) => 0.01 + (0.39 * i) / 45);
  const z = alphas.map((a) => betas.map((b) =>
    logLikelihood(data.opt1Rewarded, data.magOpt1, data.magOpt2, data.choice1, a, b)));
  const top = Math.max(...z.flat());
  const walked = paths.map((path) => path.slice(0, Math.max(1, Math.min(step, path.length))));

  plotly().react(el, [
    // lightened, so the climb stays visible where it starts, over the dark
    // low end of the scale
    { type: 'contour', z, x: betas, y: alphas, zmin: top - 70, zmax: top, opacity: 0.6,
      colorscale: CONTINUOUS,
      showscale: false, contours: { coloring: 'heatmap', start: top - 70, end: top, size: 7 },
      line: { color: 'rgba(255,255,255,0.7)', width: 1 },
      hovertemplate: 'α %{y:.2f}<br>β %{x:.2f}<br>log likelihood %{z:.0f}<extra></extra>' },
    ...walked.map((path) => ({
      x: path.map((p) => p.beta), y: path.map((p) => p.alpha), mode: 'lines+markers',
      line: { color: COL.ink, width: 2 }, marker: { size: 5, color: COL.ink },
      hovertemplate: 'step %{pointNumber}<br>α %{y:.2f}, β %{x:.2f}<extra></extra>',
    })),
    // where each run has got to, so the eye can follow the two climbs
    { x: walked.map((p) => p[p.length - 1].beta), y: walked.map((p) => p[p.length - 1].alpha),
      mode: 'markers', marker: { size: 11, color: COL.estimate, line: { color: '#fff', width: 1.5 } },
      hovertemplate: 'step %{pointNumber}<extra></extra>' },
    { x: [truth.beta], y: [truth.alpha], mode: 'markers',
      marker: { size: 14, color: '#fff', symbol: 'circle', line: { color: COL.truth, width: 2 } },
      hovertemplate: 'simulated with these values<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis('inverse temperature β', { showgrid: false }),
    yaxis: axis('learning rate α', { showgrid: false }),
    annotations: paths.map((path, i) => ({
      x: path[0].beta, y: path[0].alpha, ax: i === 0 ? 34 : 44, ay: i === 0 ? 22 : -4,
      text: `start ${i + 1}`, font: { size: 15, color: COL.ink }, arrowcolor: COL.ink,
      arrowhead: 2, arrowsize: 0.8, arrowwidth: 1, xanchor: 'left',
    })).concat([{ x: truth.beta, y: truth.alpha, ax: 40, ay: -34, text: 'simulated with',
      font: { size: 15, color: COL.truth }, arrowcolor: COL.truth, arrowhead: 2,
      arrowsize: 0.8, arrowwidth: 1, xanchor: 'left' }]),
  }), CONFIG);
}

export const ascentNote = (v = {}) => {
  const step = Math.round(v.step ?? ASCENT_STEPS);
  const done = step >= ASCENT_STEPS;
  return 'Two runs of gradient ascent on one likelihood surface, from two different starting '
    + 'points. Every step goes uphill; the steps are long where the surface is steep and shorten '
    + 'as it flattens. '
    + (done ? 'Both end near the values the data were simulated with.'
            : `Shown after ${step} steps; the purple dots are where each run has got to.`);
};

/* --- the learning rate as a weighting of past outcomes -------------------- */

export function drawWeights(el, v, { width = 1000, height = 340 } = {}) {
  const nBack = 20;
  const w = rlWeights(v.alpha, nBack);
  plotly().react(el, [{
    x: Array.from({ length: nBack }, (_, k) => -(k + 1)), y: w, type: 'bar',
    marker: { color: COL.neutral },
    hovertemplate: '%{x} trials back<br>weight %{y:.3f}<extra></extra>',
  }], layout({
    width, height, bargap: 0.35,
    // placed in data coordinates and clamped, so the label stays inside the
    // plot even when alpha is 1 and the first bar fills the whole height
    annotations: [{
      x: -1, y: w[0], ax: -3.4, ay: Math.min(w[0] + 0.24, 0.93), axref: 'x', ayref: 'y',
      text: 'the last trial carries weight α',
      font: { size: 12, color: COL.ink }, arrowcolor: COL.muted,
      arrowhead: 2, arrowsize: 0.8, arrowwidth: 1, xanchor: 'right',
    }],
    xaxis: axis('delay (trials)', { dtick: 5, range: [-20.6, -0.4] }),
    yaxis: axis('weight', { range: [0, 1.08] }),
  }), CONFIG);
}

export const weightsNote = (v) =>
  `Shown with a learning rate of ${v.alpha.toFixed(2)}: the most recent outcome carries weight `
  + `${v.alpha.toFixed(2)}, and each earlier one is discounted by a further factor of `
  + `${(1 - v.alpha).toFixed(2)}.`;
