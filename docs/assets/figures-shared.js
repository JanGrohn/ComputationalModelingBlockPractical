/* ---------------------------------------------------------------------------
   The figures more than one session shows.

   Sessions 3 and 4 spend much of their time recapping sessions 1 and 2, so
   they show the same plots again. Each session gives the figure its own
   element id and its own size; what is drawn lives here, once, so a change to
   a colour or an axis reaches every deck that shows it.
--------------------------------------------------------------------------- */

import {
  makeRng, generateSchedule, generateMagnitudes, rlModel, rlWeights, softmax,
  simulateRlModel, stableThenVolatile,
} from './models.js';
import { interactiveFigure, layout, CONFIG, COL, CONTINUOUS, HEADROOM } from './deck.js';
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

/** The magnitudes the task offers: 102 trials of them, drawn once. */
const TASK_MAGNITUDES = (() => {
  const rng = makeRng(31415);
  const draw = () => Array.from({ length: 102 }, () => 1 + Math.floor(rng() * 99));
  return [draw(), draw()];
})();

/**
 * The magnitudes on offer, trial by trial. By default the task's own; a slide
 * that pairs this with the choices plot passes that schedule's magnitudes, so
 * the two panels describe the same participant.
 */
export function drawMagnitudes(el, { width = 1120, height = 165,
                                     magnitudes = TASK_MAGNITUDES } = {}) {
  const [mag1, mag2] = magnitudes.map((m) => Array.from(m));
  const n = mag1.length;
  const x = trials(n);
  const keyAt = (f) => 1 + f * n;
  plotly().react(el, [
    { x, y: mag1, mode: 'lines+markers', name: 'magnitude option 1',
      line: { color: COL.option1, width: 1.6 }, marker: { size: 4, color: COL.option1 },
      hovertemplate: 'trial %{x}<br>magnitude option 1: %{y}<extra></extra>' },
    { x, y: mag2, mode: 'lines+markers', name: 'magnitude option 2',
      line: { color: COL.option2, width: 1.6, dash: 'dot' }, marker: { size: 4, color: COL.option2 },
      hovertemplate: 'trial %{x}<br>magnitude option 2: %{y}<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis('trial number', { range: [0.5, n + 0.5] }),
    yaxis: axis('magnitude', { range: [0, 104], dtick: 50 }),
    annotations: [
      { x: keyAt(0), text: '— magnitude option 1', color: COL.option1 },
      { x: keyAt(0.29), text: '··· magnitude option 2', color: COL.option2 },
    ].map(({ x: ax, text, color }) => ({ x: ax, y: 1.24, xref: 'x', yref: 'paper',
      xanchor: 'left', showarrow: false, text, font: { size: 16, color } })),
  }), CONFIG);
}

/** The magnitudes of the schedule the choices plot simulates. */
export const SCHEDULE_MAGNITUDES = [SCHEDULE.magOpt1, SCHEDULE.magOpt2];

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

/** Pearson's correlation. */
export function correlation(a, b) {
  const n = a.length;
  const ma = a.reduce((x, y) => x + y, 0) / n;
  const mb = b.reduce((x, y) => x + y, 0) / n;
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < n; i++) {
    ab += (a[i] - ma) * (b[i] - mb); aa += (a[i] - ma) ** 2; bb += (b[i] - mb) ** 2;
  }
  return ab / Math.sqrt(aa * bb);
}

/**
 * Simulated against recovered values for one parameter. The points are
 * coloured by the inverse temperature each participant was simulated with,
 * since that is what decides how much their choices can reveal: someone with
 * a β near zero chooses almost at random.
 */
export function drawRecovery(el, key, fitKey, label, hi, { width = 500, height = 340 } = {}) {
  const simulated = RECOVERY.map((r) => r[key]);
  const recovered = RECOVERY.map((r) => r[fitKey]);
  plotly().react(el, [
    { x: [0, hi], y: [0, hi], mode: 'lines', line: { color: COL.truth, width: 1.2, dash: 'dash' },
      hoverinfo: 'skip' },
    { x: simulated, y: recovered, mode: 'markers',
      marker: { size: 10, color: RECOVERY.map((r) => r.beta), colorscale: CONTINUOUS,
                cmin: 0, cmax: 0.3, line: { color: '#fff', width: 0.5 },
                colorbar: { title: { text: 'simulated β', side: 'right' }, thickness: 10, len: 0.8,
                            outlinewidth: 0, tickfont: { size: 12 }, titlefont: { size: 13 } } },
      customdata: RECOVERY.map((r) => r.beta),
      hovertemplate: 'simulated %{x:.3f}<br>recovered %{y:.3f}<br>simulated β %{customdata:.3f}'
                     + '<extra></extra>' },
  ], layout({
    width, height,
    // constrain the domain rather than the range, so the box comes out square
    // without Plotly widening the axes past the values they are meant to show
    xaxis: axis(`simulated ${label}`, { range: [0, hi], constrain: 'domain' }),
    yaxis: axis(`recovered ${label}`, { range: [0, hi], scaleanchor: 'x', scaleratio: 1,
                                        constrain: 'domain' }),
    annotations: [{ x: 0.04, y: 0.96, xref: 'x domain', yref: 'y domain', xanchor: 'left',
      yanchor: 'top', showarrow: false, font: { size: 16, color: COL.ink },
      text: `r = ${correlation(simulated, recovered).toFixed(2)}` }],
  }), CONFIG);
}

export const recoveryNote = (label) => `${label}: ${RECOVERY.length} simulated participants, each `
  + 'fitted back with the same model, coloured by the inverse temperature they were simulated '
  + 'with. The dashed line is simulated = recovered.';

/**
 * Box plots in a strip above a histogram, one per series, as the notebooks'
 * histograms draw them. Returns the extra traces and the two axes to use.
 */
export function boxesAbove(series) {
  return {
    traces: series.map(({ x, color, name }) => ({
      x, type: 'box', yaxis: 'y2', name, orientation: 'h', boxpoints: 'outliers',
      marker: { color, size: 4 }, line: { color, width: 1.5 }, fillcolor: '#ffffff',
      hovertemplate: `${name}<br>%{x:.2f}<extra></extra>`,
    })),
    axes: {
      yaxis: { domain: [0, 0.74] },
      yaxis2: { domain: [0.8, 1], showticklabels: false, showgrid: false, zeroline: false,
                fixedrange: true },
    },
  };
}

/* --- model falsification -------------------------------------------------- */

export function drawFalsification(el, which, { width = 520, height = 300 } = {}) {
  const { stable, volatile } = FALSIFICATION[which];
  const xbins = { start: 0, end: 0.8, size: 0.05 };
  const boxes = boxesAbove([{ x: volatile, color: COL.volatile, name: 'volatile block' },
                            { x: stable, color: COL.stable, name: 'stable block' }]);
  plotly().react(el, [
    { x: stable, type: 'histogram', xbins, marker: { color: COL.stable, opacity: 0.55 },
      hovertemplate: 'stable block<br>%{y} participants<extra></extra>' },
    { x: volatile, type: 'histogram', xbins, marker: { color: COL.volatile, opacity: 0.55 },
      hovertemplate: 'volatile block<br>%{y} participants<extra></extra>' },
    ...boxes.traces,
  ], layout({
    margin: HEADROOM,
    width, height, barmode: 'overlay', showlegend: false,
    xaxis: axis('recovered learning rate', { range: [0, 0.8] }),
    yaxis: Object.assign(axis('participants'), boxes.axes.yaxis),
    yaxis2: boxes.axes.yaxis2,
    annotations: blockKey(),
  }), CONFIG);
}

/** The stable/volatile colour key, placed above a plot. */
export function blockKey(stableAt = 0, volatileAt = 0.42) {
  return [
    { x: stableAt, text: '■ stable block', color: COL.stable },
    { x: volatileAt, text: '■ volatile block', color: COL.volatile },
  ].map(({ x, text, color }) => ({ x, y: 1.01, xref: 'paper', yref: 'paper', xanchor: 'left',
    yanchor: 'bottom', showarrow: false, text, font: { size: 16, color } }));
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
  const { truth, paths } = ASCENT;
  const step = Math.round(v.step ?? ASCENT_STEPS);
  const { alphas, betas, z, top } = ascentSurface();
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

/* --- figures the original decks had, redrawn ------------------------------ */

/**
 * The model written as a logistic regression: the weights fitted to the last
 * five outcomes, normalised to sum to one, against the weights the learning
 * model predicts, with the inverse temperature read off their sum. Fitted with
 * the session 2 notebook's own simulation code by task-source/precompute-fits.py.
 */
export function drawRegression(el, REGRESSION, { width = 1100, height = 290 } = {}) {
  const lags = REGRESSION.normalised.map((_, k) => `lag ${k + 1}`);
  const predicted = REGRESSION.normalised.map((_, k) => REGRESSION.alpha * (1 - REGRESSION.alpha) ** k);
  plotly().react(el, [
    { x: lags, y: REGRESSION.normalised, type: 'bar', name: 'fitted weight / sum of weights',
      marker: { color: COL.estimate },
      hovertemplate: '%{x}: %{y:.3f}<extra>fitted</extra>' },
    { x: lags, y: predicted, mode: 'markers', name: 'α(1 − α)<sup>lag − 1</sup>',
      marker: { symbol: 'line-ew', size: 46, line: { color: COL.truth, width: 3 } },
      hovertemplate: '%{x}: %{y:.3f}<extra>predicted from α</extra>' },
  ], layout({
    width, height, bargap: 0.45,
    xaxis: axis(''),
    yaxis: axis('weight', { range: [0, 0.75], dtick: 0.2 }),
    annotations: [
      { x: 0.98, y: 0.98, xref: 'paper', yref: 'paper', xanchor: 'right', yanchor: 'top',
        showarrow: false, align: 'right', font: { size: 16, color: COL.ink },
        text: `<span style="color:${COL.estimate}">■ fitted weight ÷ sum of weights</span><br>`
          + `▬ α(1 − α)<sup>lag − 1</sup>, the model's weights for α = ${REGRESSION.alpha}<br><br>`
          + `<span style="color:${COL.estimate}">sum of weights ÷ 100 = ${REGRESSION.betaFromWeights}</span><br>`
          + `the β simulated with: ${REGRESSION.beta}` },
    ],
  }), CONFIG);
}

export const regressionNote = (REGRESSION) =>
  `A logistic regression of ${REGRESSION.nTrials.toLocaleString('en-GB')} simulated choices on the `
  + `last five outcomes, as at the end of the session 2 notebook. Normalised, the weights fall `
  + `on the learning model's α(1 − α)^(lag − 1) with α = ${REGRESSION.alpha}, and their sum `
  + `divided by 100 gives back the inverse temperature, ${REGRESSION.betaFromWeights} against `
  + `the ${REGRESSION.beta} simulated.`;

/**
 * What one trial adds to the error: minus the log of the probability the
 * model gave the choice that was made. It is small for choices the model
 * expected and grows without limit for choices it thought unlikely, which is
 * which trials dominate the sum.
 */
export function drawErrorTerm(el, { width = 560, height = 300 } = {}) {
  const p = Array.from({ length: 200 }, (_, i) => 0.005 + (0.995 * i) / 199);
  plotly().react(el, [{
    x: p, y: p.map((q) => -Math.log(q)), mode: 'lines', line: { color: COL.neutral, width: 3 },
    hovertemplate: 'P = %{x:.2f}<br>error %{y:.2f}<extra></extra>',
  }], layout({
    width, height,
    xaxis: axis('probability the model gave the observed choice', { range: [0, 1], dtick: 0.25 }),
    yaxis: axis('error, −log P', { range: [0, 5.4] }),
    annotations: [
      { x: 0.04, y: -Math.log(0.04), ax: 60, ay: -10, text: 'a choice the model thought unlikely',
        xanchor: 'left', font: { size: 14, color: COL.ink }, arrowcolor: COL.muted, arrowhead: 2,
        arrowsize: 0.8, arrowwidth: 1 },
      { x: 0.92, y: -Math.log(0.92), ax: -30, ay: -46, text: 'a choice it expected',
        xanchor: 'right', font: { size: 14, color: COL.ink }, arrowcolor: COL.muted, arrowhead: 2,
        arrowsize: 0.8, arrowwidth: 1 },
    ],
  }), CONFIG);
}

export const errorTermNote = 'What a single trial adds to the negative log likelihood: close to '
  + 'nothing for a choice the model expected, and without limit for one it thought unlikely.';

/** Least squares polynomial of the given degree, by the normal equations. */
function polyfit(x, y, degree) {
  const n = degree + 1;
  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) =>
    x.reduce((s, xi) => s + xi ** (i + j), 0)));
  const b = Array.from({ length: n }, (_, i) => x.reduce((s, xi, k) => s + xi ** i * y[k], 0));
  for (let c = 0; c < n; c++) {            // Gaussian elimination with pivoting
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const coef = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    coef[r] = (b[r] - A[r].slice(r + 1).reduce((s, a, k) => s + a * coef[r + 1 + k], 0)) / A[r][r];
  }
  return (t) => coef.reduce((s, c, i) => s + c * t ** i, 0);
}

/**
 * The same points fitted by a model with too few parameters, about the right
 * number, and too many: the last fits these points best and would predict new
 * ones worst. As in the original slide, drawn rather than described.
 */
export function drawFitComplexity(el, { width = 1100, height = 270 } = {}) {
  const rng = makeRng(808);
  const x = Array.from({ length: 10 }, (_, i) => -1 + (2 * i) / 9 + (rng() - 0.5) * 0.08);
  const y = x.map((t) => t * t + (rng() - 0.5) * 0.35);
  const grid = Array.from({ length: 200 }, (_, i) => -1.05 + (2.1 * i) / 199);
  const fits = [[0, 'too simple'], [2, 'about right'], [9, 'too complex']];
  const axes = {};
  const traces = [];
  fits.forEach(([degree, label], i) => {
    const suffix = i === 0 ? '' : i + 1;
    const f = polyfit(x, y, degree);
    axes[`xaxis${suffix}`] = axis('', { showticklabels: false, range: [-1.1, 1.1] });
    axes[`yaxis${suffix}`] = axis('', { showticklabels: false, range: [-0.6, 1.6] });
    traces.push(
      { x: grid, y: grid.map(f), mode: 'lines', xaxis: `x${suffix}`, yaxis: `y${suffix}`,
        line: { color: COL.estimate, width: 3 }, hoverinfo: 'skip' },
      { x, y, mode: 'markers', xaxis: `x${suffix}`, yaxis: `y${suffix}`,
        marker: { size: 9, color: COL.neutral }, hoverinfo: 'skip' });
  });
  plotly().react(el, traces, layout({
    margin: HEADROOM,
    width, height, showlegend: false,
    grid: { rows: 1, columns: 3, pattern: 'independent', xgap: 0.08 },
    ...axes,
    annotations: fits.map(([degree, label], i) => ({
      text: `${label}: ${degree + 1} parameter${degree ? 's' : ''}`, x: 0.5, y: 1.03,
      xref: `x${i === 0 ? '' : i + 1} domain`, yref: 'paper', yanchor: 'bottom', showarrow: false,
      font: { size: 17, color: COL.ink },
    })),
  }), CONFIG);
}

export const fitComplexityNote = 'The same ten points fitted with one, three and ten free '
  + 'parameters. The fit only ever improves as parameters are added, and with ten it passes '
  + 'through every point, but it would predict a new point worst of the three.';

/** The softmax at a low and a high inverse temperature, on one plot. */
export function drawSoftmaxPair(el, { low = 0.02, high = 0.2, width = 1000, height = 380 } = {}) {
  const d = Array.from({ length: 201 }, (_, i) => -100 + i);
  const curve = (beta) => d.map((u) => softmax(u, 0, beta));
  plotly().react(el, [
    { x: d, y: curve(high), mode: 'lines', line: { color: COL.option1, width: 3 },
      hovertemplate: `β = ${high}<br>u<sub>1</sub> − u<sub>2</sub> = %{x}: P(c<sub>1</sub>) = %{y:.2f}<extra></extra>` },
    { x: d, y: curve(low), mode: 'lines', line: { color: COL.option1, width: 3, dash: 'dash' },
      hovertemplate: `β = ${low}<br>u<sub>1</sub> − u<sub>2</sub> = %{x}: P(c<sub>1</sub>) = %{y:.2f}<extra></extra>` },
  ], layout({
    width, height,
    xaxis: Object.assign(axis('utility 1 − utility 2  (u<sub>1</sub> − u<sub>2</sub>)'),
                         { zeroline: true, zerolinecolor: '#e2e6ea', range: [-100, 100] }),
    yaxis: axis('probability of choosing option 1', { range: [-0.04, 1.04], tickvals: [0, 0.25, 0.5, 0.75, 1] }),
    annotations: [
      { x: 12, y: softmax(12, 0, high), ax: -40, ay: -26, text: `β = ${high}: steep, rarely the worse option`,
        xanchor: 'right', font: { size: 15, color: COL.ink }, arrowcolor: COL.muted, arrowhead: 2,
        arrowsize: 0.8, arrowwidth: 1 },
      { x: -55, y: softmax(-55, 0, low), ax: 30, ay: -60, text: `β = ${low}: shallow, often the worse option`,
        xanchor: 'left', font: { size: 15, color: COL.ink }, arrowcolor: COL.muted, arrowhead: 2,
        arrowsize: 0.8, arrowwidth: 1 },
    ],
  }), CONFIG);
}

export const softmaxPairNote = (low = 0.02, high = 0.2) => `The softmax at inverse temperatures `
  + `of ${low} (dashed) and ${high} (solid). The higher it is, the more the choice follows the `
  + `difference in utility.`;

/** The likelihood surface the gradient ascent climbs, as a likelihood. */
function ascentSurface() {
  const { data } = ASCENT;
  const alphas = Array.from({ length: 46 }, (_, i) => 0.01 + (0.88 * i) / 45);
  const betas = Array.from({ length: 46 }, (_, i) => 0.01 + (0.39 * i) / 45);
  const z = alphas.map((a) => betas.map((b) =>
    logLikelihood(data.opt1Rewarded, data.magOpt1, data.magOpt2, data.choice1, a, b)));
  return { alphas, betas, z, top: Math.max(...z.flat()) };
}

/**
 * The plain likelihood of the same choices gradient ascent climbs: a single
 * narrow spike, flat almost everywhere else, so from most starting points
 * there is no slope to follow. Shown beside the climb, it is why fitting
 * works with the log.
 */
export function drawLikelihoodSpike(el, { width = 420, height = 360 } = {}) {
  const { alphas, betas, z, top } = ascentSurface();
  plotly().react(el, [{
    type: 'contour', z: z.map((row) => row.map((ll) => Math.exp(ll - top))), x: betas, y: alphas,
    colorscale: CONTINUOUS, showscale: false, contours: { coloring: 'heatmap' },
    line: { color: 'rgba(255,255,255,0.5)', width: 0.8 },
    hovertemplate: 'α %{y:.2f}<br>β %{x:.2f}<br>likelihood ÷ its peak %{z:.3f}<extra></extra>',
  }], layout({
    width, height,
    xaxis: axis('inverse temperature β', { showgrid: false }),
    yaxis: axis('learning rate α', { showgrid: false }),
  }), CONFIG);
}

export const likelihoodSpikeNote = 'The likelihood of the same choices, rather than its log: one '
  + 'narrow peak and almost nothing anywhere else, so there is no slope to climb from most '
  + 'starting points. The log keeps the peak where it is and slopes towards it from everywhere.';

/** A small plot of the learning model at work, for a recap beside its equations. */
export function drawMiniLearning(el, { width = 330, height = 150 } = {}) {
  const { opt1Rewarded } = SCHEDULE;
  const probOpt1 = Array.from(rlModel(opt1Rewarded, 0.3));
  const x = trials(trueProbability.length);
  plotly().react(el, [
    { x, y: trueProbability, mode: 'lines', line: { color: COL.truth, dash: 'dash', width: 1.4 },
      hoverinfo: 'skip' },
    { x, y: probOpt1, mode: 'lines', line: { color: COL.option1, width: 2 },
      hovertemplate: 'trial %{x}: %{y:.2f}<extra>RL model probability</extra>' },
  ], layout({
    width, height, margin: { l: 8, r: 8, t: 6, b: 6 },
    xaxis: axis('', { showticklabels: false }), yaxis: axis('', { range: [0, 1], dtick: 0.5 }),
  }), CONFIG);
}

/** A small softmax beside the deterministic rule, for the same recap. */
export function drawMiniSoftmax(el, { width = 330, height = 150 } = {}) {
  const d = Array.from({ length: 201 }, (_, i) => -100 + i);
  plotly().react(el, [
    { x: d, y: d.map((u) => (u > 0 ? 1 : u < 0 ? 0 : 0.5)), mode: 'lines',
      line: { color: COL.option1, dash: 'dot', width: 1.4 }, hoverinfo: 'skip' },
    { x: d, y: d.map((u) => softmax(u, 0, 0.08)), mode: 'lines', line: { color: COL.option1, width: 2 },
      hovertemplate: 'u₁ − u₂ = %{x}: P(c₁) = %{y:.2f}<extra></extra>' },
  ], layout({
    width, height, margin: { l: 8, r: 8, t: 6, b: 6 },
    xaxis: axis('', { showticklabels: false, zeroline: true, zerolinecolor: '#e2e6ea' }),
    yaxis: axis('', { range: [0, 1], dtick: 0.5 }),
  }), CONFIG);
}

/**
 * Landscapes for a model with a learning rate per block: one simulated
 * participant who learns faster in the volatile block, each block's log
 * likelihood drawn over its own learning rate and the shared β.
 */
export function drawBlockLandscapes(el, { width = 1120, height = 330, alphaS = 0.12, alphaV = 0.45,
                                          beta = 0.15 } = {}) {
  const rng = makeRng(6060);
  const tp = stableThenVolatile();
  const opt1Rewarded = generateSchedule(tp, rng);
  const magOpt1 = generateMagnitudes(tp.length, rng);
  const magOpt2 = generateMagnitudes(tp.length, rng);
  const blocks = [[0, 100, alphaS, 'stable block'], [100, 200, alphaV, 'volatile block']];
  const alphas = Array.from({ length: 50 }, (_, i) => 0.01 + (0.89 * i) / 49);
  const betas = Array.from({ length: 50 }, (_, i) => 0.005 + (0.395 * i) / 49);
  const traces = [];
  const axes = {};
  blocks.forEach(([from, to, alpha, label], i) => {
    const suffix = i === 0 ? '' : '2';
    const sl = (a) => Array.from(a).slice(from, to);
    const { choice1 } = simulateRlModel(sl(opt1Rewarded), sl(magOpt1), sl(magOpt2), alpha, beta,
                                        { rng: makeRng(70 + i) });
    const z = alphas.map((a) => betas.map((b) =>
      logLikelihood(sl(opt1Rewarded), sl(magOpt1), sl(magOpt2), choice1, a, b)));
    const top = Math.max(...z.flat());
    traces.push(
      { type: 'contour', z, x: betas, y: alphas, xaxis: `x${suffix}`, yaxis: `y${suffix}`,
        zmin: top - 40, zmax: top, colorscale: CONTINUOUS, showscale: false,
        contours: { coloring: 'heatmap', start: top - 40, end: top, size: 4 },
        line: { color: 'rgba(255,255,255,0.5)', width: 0.8 },
        hovertemplate: 'α %{y:.2f}<br>β %{x:.2f}<br>log likelihood %{z:.1f}<extra></extra>' },
      { x: [beta], y: [alpha], mode: 'markers', xaxis: `x${suffix}`, yaxis: `y${suffix}`,
        marker: { size: 13, color: '#fff', line: { color: COL.truth, width: 2 } },
        hovertemplate: `simulated with α ${alpha}<extra></extra>` });
    axes[`xaxis${suffix}`] = axis('inverse temperature β', { showgrid: false });
    axes[`yaxis${suffix}`] = axis(i === 0 ? 'learning rate α' : '', { showgrid: false });
  });
  plotly().react(el, traces, layout({
    margin: HEADROOM,
    width, height, showlegend: false,
    grid: { rows: 1, columns: 2, pattern: 'independent', xgap: 0.2 },
    ...axes,
    annotations: blocks.map(([, , , label], i) => ({
      text: label, x: 0.5, y: 1.03, xref: `x${i ? 2 : ''} domain`, yref: 'paper', yanchor: 'bottom',
      showarrow: false, font: { size: 17, color: i ? COL.volatile : COL.stable } })),
  }), CONFIG);
}

export const blockLandscapesNote = 'With a learning rate for each block, each block has a landscape '
  + 'of its own. One simulated participant, with α = 0.12 in the stable block and 0.45 in the '
  + 'volatile one; the circles mark the values simulated with.';
