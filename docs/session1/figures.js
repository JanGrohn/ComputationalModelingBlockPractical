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

/**
 * Traces for a schedule plot.
 *
 * Trials are numbered from 1, matching the equations: the notebooks call the
 * starting estimate p_1, so the first trial has to be trial 1. The outcome
 * dots take the colour of whichever option was rewarded, green for option 1
 * and orange for option 2, and sit at 1 and 0 as before, so the colour is a
 * second cue rather than the only one.
 */
function scheduleTrace(opt1Rewarded, trueProbability, probOpt1) {
  const x = trials(opt1Rewarded.length).map((t) => t + 1);
  const pick = (want) => {
    const xs = [], ys = [];
    opt1Rewarded.forEach((o, i) => { if (o === want) { xs.push(x[i]); ys.push(o); } });
    return { x: xs, y: ys };
  };
  const one = pick(1);
  const two = pick(0);
  return [
    { ...one, mode: 'markers', name: 'option 1 rewarded',
      marker: { size: 6, color: COL.opt1 },
      hovertemplate: 'trial %{x}<br>option 1 rewarded<extra></extra>' },
    { ...two, mode: 'markers', name: 'option 2 rewarded',
      marker: { size: 6, color: COL.opt2 },
      hovertemplate: 'trial %{x}<br>option 2 rewarded<extra></extra>' },
    { x, y: trueProbability, mode: 'lines', name: 'true probability',
      line: { color: COL.ink, dash: 'dash', width: 2 },
      hovertemplate: 'true probability %{y:.2f}<extra></extra>' },
    { x, y: Array.from(probOpt1), mode: 'lines', name: 'model estimate',
      line: { color: COL.opt1, width: 3 },
      hovertemplate: 'trial %{x}<br>model estimate %{y:.2f}<extra></extra>' },
  ];
}

/** Series names, in the headroom above the plot, large enough to read. */
function scheduleLabels(n) {
  const label = (x, text, color) => ({
    x, y: 1.16, xanchor: 'left', showarrow: false, text,
    font: { size: 17, color },
  });
  return [
    label(1, '\u25cf\u2009option 1 rewarded', COL.opt1),
    label(n * 0.24, '\u25cf\u2009option 2 rewarded', COL.opt2),
    label(n * 0.51, '\u2013\u2009\u2013\u2009true probability', COL.ink),
    label(n * 0.77, '\u2014\u2009model\u2019s estimate', COL.opt1),
  ];
}

const fixedSchedule = {
  'fig-schedule': {
    params: {
      alpha:        { min: 0.01, max: 1, step: 0.01, value: 0.05, label: '&alpha;' },
      startingProb: { min: 0, max: 1, step: 0.01, value: 0.5, label: '<i>p</i><sub>1</sub>' },
      trueProb:     { min: 0, max: 1, step: 0.05, value: 0.8, label: 'true <i>p</i>' },
    },
    draw(el, v) {
      const n = 200;
      const rng = makeRng(v.__seed ?? 12345);
      const trueProbability = Array(n).fill(v.trueProb);
      const opt1Rewarded = generateSchedule(trueProbability, rng);
      const probOpt1 = rlModel(opt1Rewarded, v.alpha, v.startingProb);
      plotly().react(el, scheduleTrace(opt1Rewarded, trueProbability, probOpt1), layout({
        width: 1120, height: 430,
        margin: { l: 10, r: 16, t: 10, b: 46 },
        xaxis: Object.assign(layout().xaxis, { title: 'trial', range: [0.5, n + 0.5] }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'P(option 1 rewarded)', standoff: 12 }, automargin: true,
          // the range runs past 1 only to make room for the labels above the
          // plot, so the ticks stop where the probability does
          tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
          range: [-0.06, 1.44] }),
        annotations: scheduleLabels(n),
      }), CONFIG);
    },
    note: (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)}, a starting estimate of `
                 + `${v.startingProb.toFixed(2)} and a true reward probability of ${v.trueProb.toFixed(2)}. `
                 + `Dots along the top and bottom are the trial outcomes, the dashed line the `
                 + `true probability, the green line the model's running estimate.`,
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
        width: 1120, height: 360,
        margin: { l: 10, r: 16, t: 10, b: 44 },
        xaxis: Object.assign(layout().xaxis, { title: 'trial', range: [0.5, trueProbability.length + 0.5] }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'P(option 1 rewarded)', standoff: 12 }, automargin: true,
          // the range runs past 1 only to make room for the labels above the
          // plot, so the ticks stop where the probability does
          tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1],
          range: [-0.06, 1.44] }),
        shapes: [{ type: 'rect', xref: 'x', yref: 'paper', x0: 0.5, x1: 100.5, y0: 0, y1: 1,
                   fillcolor: '#000', opacity: 0.045, line: { width: 0 }, layer: 'below' }],
        annotations: scheduleLabels(trueProbability.length).concat([
          // the block names head their halves of the plot, well clear of the
          // outcome dots that sit on the line at 1
          { x: 50,  y: 1.36, text: 'volatile', showarrow: false,
            font: { size: 21, color: COL.ink } },
          { x: 150, y: 1.36, text: 'stable', showarrow: false,
            font: { size: 21, color: COL.ink } },
        ]),
      }), CONFIG);
    },
    note: (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)}. The reward probability `
                 + `reverses every 25 trials in the volatile block (shaded) and holds steady in `
                 + `the stable block.`,
  },
};

/* --- the two conditions, side by side ----------------------------------- */

const CONDITIONS = {
  'fig-condition-1': { condition: 1, title: 'Condition 1' },
  'fig-condition-2': { condition: 2, title: 'Condition 2' },
};

/**
 * The reward schedule a participant in each condition actually meets. These
 * are the schedules the task itself generates: condition 1 holds the
 * contingency steady, condition 2 reverses it every 25 trials.
 */
function conditionSchedule(condition) {
  const n = 102;
  const pOption1 = condition === 1
    ? new Array(n).fill(0.75)
    : [...new Array(28).fill(0.75), ...new Array(25).fill(0.25),
       ...new Array(25).fill(0.75), ...new Array(25).fill(0.25)].slice(0, n);
  const rng = makeRng(condition === 1 ? 404 : 909);
  return { pOption1, opt1Rewarded: generateSchedule(pOption1, rng) };
}

const conditions = Object.fromEntries(Object.entries(CONDITIONS).map(([id, spec]) => [id, {
  params: {},
  draw(el) {
    const { pOption1, opt1Rewarded } = conditionSchedule(spec.condition);
    const x = trials(pOption1.length).map((t) => t + 1);
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
      width: 530, height: 205,
      margin: { l: 10, r: 12, t: 26, b: 40 },
      xaxis: Object.assign(layout().xaxis, { title: 'trial', range: [0.5, pOption1.length + 0.5] }),
      // automargin keeps the axis title clear of its own tick labels
      yaxis: Object.assign(layout().yaxis, {
        title: { text: 'P(option 1 rewarded)', standoff: 10 },
        automargin: true, range: [-0.08, 1.08], dtick: 0.25 }),
    }), CONFIG);
  },
  note: () => spec.condition === 1
    ? 'P(option 1 rewarded) holds at 0.75 for all 102 trials.'
    : 'P(option 1 rewarded) starts at 0.75 and reverses every 25 trials from trial 29.',
}]));

/* --- the reward magnitudes ----------------------------------------------- */

const magnitudes = {
  'fig-magnitudes': {
    params: {},
    draw(el) {
      const n = 102;
      const rng = makeRng(31415);
      const x = trials(n).map((t) => t + 1);
      const mag = () => Array.from({ length: n }, () => 1 + Math.floor(rng() * 99));
      plotly().react(el, [
        { x, y: mag(), mode: 'lines+markers', name: 'option 1',
          line: { color: COL.opt1, width: 1.6 }, marker: { size: 4, color: COL.opt1 },
          hovertemplate: 'trial %{x}<br>option 1: %{y} points<extra></extra>' },
        { x, y: mag(), mode: 'lines+markers', name: 'option 2',
          line: { color: COL.opt2, width: 1.6, dash: 'dot' }, marker: { size: 4, color: COL.opt2 },
          hovertemplate: 'trial %{x}<br>option 2: %{y} points<extra></extra>' },
      ], layout({
        width: 1120, height: 155,
        margin: { l: 10, r: 16, t: 24, b: 40 },
        xaxis: Object.assign(layout().xaxis, { title: 'trial', range: [0.5, n + 0.5] }),
        yaxis: Object.assign(layout().yaxis, {
          title: { text: 'points on offer', standoff: 12 }, automargin: true,
          range: [0, 104], dtick: 50 }),
        annotations: [
          { x: 1, y: 1.22, xref: 'x', yref: 'paper', xanchor: 'left', showarrow: false,
            text: '\u2014\u2009option 1', font: { size: 16, color: COL.opt1 } },
          { x: 18, y: 1.22, xref: 'x', yref: 'paper', xanchor: 'left', showarrow: false,
            text: '\u00b7\u00b7\u00b7\u2009option 2', font: { size: 16, color: COL.opt2 } },
        ],
      }), CONFIG);
    },
    note: () => 'Points on offer, redrawn at random every trial and independently of which '
                + 'option is rewarded. Identical in both conditions.',
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
    // the shading is the quantity the slide is about, so it needs saying
    showscale: true,
    colorbar: { title: { text: 'utility', side: 'right' }, thickness: 11,
                len: 0.92, tickvals: [0, 50, 100], outlinewidth: 0,
                tickfont: { size: 13 }, titlefont: { size: 14 } },
    contours: { coloring: 'heatmap', start: 0, end: 100, size: 10,
                showlabels: true, labelfont: { size: 10, color: '#2a2f33' } },
    line: { color: 'rgba(255,255,255,0.75)', width: 1 },
    hovertemplate: 'probability %{x:.2f}<br>magnitude %{y:.0f}<br>utility %{z:.1f}<extra></extra>',
  }], layout({
    width: 540, height: 310,
    margin: { l: 52, r: 10, t: 10, b: 42 },
    xaxis: Object.assign(layout().xaxis, {
      title: { text: 'probability', standoff: 12 }, automargin: true, dtick: 0.25 }),
    yaxis: Object.assign(layout().yaxis, {
      title: { text: 'magnitude', standoff: 12 }, automargin: true, showgrid: false }),
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
        width: 700, height: 250,
        bargap: 0.35,
        // the label is placed in data coordinates and clamped, so that it
        // stays inside the plot even when alpha is 1 and the first bar fills
        // the whole height
        annotations: [{
          x: -1, y: w[0],
          ax: -3.4, ay: Math.min(w[0] + 0.24, 0.93),
          axref: 'x', ayref: 'y',
          text: 'the last trial carries weight \u03b1',
          font: { size: 12, color: COL.ink }, arrowcolor: COL.muted,
          arrowhead: 2, arrowsize: 0.8, arrowwidth: 1, xanchor: 'right',
        }],
        xaxis: Object.assign(layout().xaxis, { title: 'trials into the past', dtick: 5, range: [-20.6, -0.4] }),
        yaxis: Object.assign(layout().yaxis, { title: 'weight', range: [0, 1.08] }),
      }), CONFIG);
    },
    note: (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)}: the most recent outcome carries `
                 + `weight ${v.alpha.toFixed(2)}, and each earlier one is discounted by a further factor `
                 + `of ${(1 - v.alpha).toFixed(2)}.`,
  },
};

/* --- wire them all up ---------------------------------------------------- */

for (const group of [fixedSchedule, reversal, conditions, magnitudes, utilities, weights]) {
  for (const [id, spec] of Object.entries(group)) {
    interactiveFigure(id, spec.params, spec.draw, spec.note);
  }
}
