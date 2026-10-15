/* ---------------------------------------------------------------------------
   The figures drawn from the fits to real participants.

   Sessions 3 and 4 both show these: the learning rates fitted to the Blain and
   Rutledge data, the tests run on them, and the model recovery that says
   whether those tests can be trusted. Every number comes from
   ./participant-fits.js, which task-source/precompute-fits.py writes using the
   practical's own fitting code, so a slide and the notebook cannot disagree.
--------------------------------------------------------------------------- */

import { layout, CONFIG, COL } from './deck.js';
import { plotly, axis, blockKey } from './figures-shared.js';
import {
  SIMULATIONS, STATS, CHI2, MODEL_RECOVERY, MODEL_FREQUENCIES,
} from './participant-fits.js';

const N = STATS.nParticipants;

/* --- learning rates fitted to the real participants ----------------------- */

/**
 * Both blocks' fitted learning rates, overlaid. This is the plot the session
 * asks students to be sceptical of: it shows the two distributions, but the
 * hypothesis is about each participant's own difference, which it hides.
 */
export function alphaDistributions(el, fits, { width = 560, height = 300 } = {}) {
  const xbins = { start: 0, end: 1, size: 0.05 };
  plotly().react(el, [
    { x: fits.alphaStable, type: 'histogram', xbins,
      marker: { color: COL.stable, opacity: 0.55 },
      hovertemplate: 'stable block<br>%{y} participants<extra></extra>' },
    { x: fits.alphaVolatile, type: 'histogram', xbins,
      marker: { color: COL.opt2, opacity: 0.55 },
      hovertemplate: 'volatile block<br>%{y} participants<extra></extra>' },
  ], layout({
    width, height, barmode: 'overlay',
    xaxis: axis('fitted learning rate', { range: [0, 1] }),
    yaxis: axis('participants'),
    annotations: blockKey(),
  }), CONFIG);
}

/** The within participant difference, which is what the t test actually uses. */
export function alphaDifference(el, fits, { width = 560, height = 300 } = {}) {
  const difference = fits.alphaVolatile.map((v, i) => v - fits.alphaStable[i]);
  const mean = difference.reduce((a, b) => a + b, 0) / difference.length;
  plotly().react(el, [
    { x: difference, type: 'histogram', xbins: { start: -1, end: 1, size: 0.1 },
      marker: { color: COL.opt1, opacity: 0.7 },
      hovertemplate: 'difference %{x}<br>%{y} participants<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis('volatile α − stable α', { range: [-1, 1], zeroline: false }),
    yaxis: axis('participants'),
    shapes: [
      { type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1,
        line: { color: COL.muted, width: 1.5, dash: 'dash' } },
      { type: 'line', x0: mean, x1: mean, yref: 'paper', y0: 0, y1: 0.86,
        line: { color: COL.ink, width: 2 } },
    ],
    annotations: [
      { x: mean, y: 0.95, yref: 'paper', xanchor: 'left', showarrow: false,
        text: `mean ${mean.toFixed(2)}`, font: { size: 15, color: COL.ink } },
    ],
  }), CONFIG);
}

/* --- the likelihood ratio test, against its null distribution ------------- */

export function drawChiSquare(el, { width = 1020, height = 215 } = {}) {
  const marks = [
    { value: SIMULATIONS.same.test.lambda, label: 'simulated with one learning rate',
      color: COL.stable },
    { value: SIMULATIONS.shuffled.test.lambda, label: 'simulated, volatile shuffled',
      color: COL.muted },
    { value: STATS.likelihoodRatio.mul.lambda, label: 'the real participants', color: COL.opt2 },
  ];
  const peak = Math.max(...CHI2.density);

  plotly().react(el, [
    { x: CHI2.x, y: CHI2.density, mode: 'lines', fill: 'tozeroy',
      line: { color: COL.opt1, width: 2 }, fillcolor: 'rgba(46,125,50,0.16)',
      hovertemplate: 'λ %{x}<extra></extra>' },
  ], layout({
    width, height,
    xaxis: axis(`λ, against χ² with ${CHI2.df} degrees of freedom`,
                { range: [30, 430] }),
    yaxis: Object.assign(axis('density'), { range: [0, peak * 1.75], showticklabels: false }),
    shapes: [
      { type: 'line', x0: CHI2.critical, x1: CHI2.critical, y0: 0, y1: peak * 1.1,
        line: { color: COL.faint, width: 1.5, dash: 'dash' } },
      ...marks.map((m) => ({ type: 'line', x0: m.value, x1: m.value, y0: 0, y1: peak * 1.22,
        line: { color: m.color, width: 2.5 } })),
    ],
    annotations: [
      { x: CHI2.critical, y: peak * 1.12, yanchor: 'bottom', xanchor: 'left', showarrow: false,
        text: `p = 0.05 at ${CHI2.critical}`, font: { size: 14, color: COL.faint } },
      // past the middle of the axis a left anchored label would run off the
      // edge, so those are hung to the left of their line instead
      ...marks.map((m, i) => {
        const left = m.value < 240;
        return { x: m.value, y: peak * (1.62 - i * 0.2), xanchor: left ? 'left' : 'right',
          showarrow: false,
          text: left ? `  ${m.label}: ${m.value}` : `${m.label}: ${m.value}  `,
          font: { size: 15, color: m.color } };
      }),
    ],
  }), CONFIG);
}

export const chiSquareNote = () =>
  `The null distribution of the statistic, with ${CHI2.df} degrees of freedom, one per `
  + `participant. Beyond ${CHI2.critical} it is significant at p < 0.05.`;

/* --- BIC, and what it says about the four models -------------------------- */

const MODEL_LABELS = {
  mul1: '1 α<br>multiplicative', mul2: '2 α<br>multiplicative',
  add1: '1 α<br>additive', add2: '2 α<br>additive',
};

export function drawBic(el, { width = 600, height = 320 } = {}) {
  const keys = Object.keys(MODEL_LABELS);
  const values = keys.map((k) => STATS.bic[k]);
  const best = Math.min(...values);
  plotly().react(el, [{
    x: keys.map((k) => MODEL_LABELS[k]), y: values, type: 'bar',
    marker: { color: values.map((v) => (v === best ? COL.opt1 : COL.faint)) },
    text: values.map((v) => v.toFixed(0)), textposition: 'outside', textfont: { size: 16 },
    hovertemplate: '%{x}<br>BIC %{y:.0f}<extra></extra>',
  }], layout({
    width, height, bargap: 0.45,
    xaxis: axis(''),
    // the differences matter, not the absolute values, so start the axis where
    // the bars differ rather than at zero
    yaxis: axis('summed BIC, lower is better', { range: [9500, 11000] }),
  }), CONFIG);
}

export const bicNote = () =>
  `BIC summed over all ${N} participants. Additive utility with two learning rates wins at `
  + `${STATS.bic.add2}, against ${STATS.bic.mul2} for multiplicative utility with two. The two `
  + `utility functions are not nested, so this comparison cannot be a likelihood ratio test.`;

/* --- the parameters fitted to each participant ---------------------------- */

export const MUL_LABELS = {
  alphaStable: 'α stable', alphaVolatile: 'α volatile', beta: 'β',
};
export const ADD_LABELS = { ...MUL_LABELS, omega: 'ω' };

/**
 * Every parameter against every other, for spotting participants at the
 * extremes. Plotly's own splom gives no control over where it puts the axis
 * titles, and at this size it stacks them on top of the tick labels, so the
 * lower triangle is built here as an ordinary grid of subplots instead.
 */
export function parameterMatrix(el, fits, labels, { width = 1060, height = 400 } = {}) {
  const keys = Object.keys(labels);
  const side = keys.length - 1;
  const traces = [];
  const axes = {};

  for (let row = 0; row < side; row++) {
    for (let col = 0; col <= row; col++) {
      const n = row * side + col + 1;
      const suffix = n === 1 ? '' : n;
      const xKey = keys[col];
      const yKey = keys[row + 1];
      traces.push({
        x: fits[xKey], y: fits[yKey], mode: 'markers', type: 'scatter',
        xaxis: `x${suffix}`, yaxis: `y${suffix}`,
        marker: { size: 6, color: COL.opt1, opacity: 0.7 },
        text: fits[xKey].map((_, i) => `participant ${i}`),
        hovertemplate: `%{text}<br>${labels[xKey]} %{x:.2f}<br>${labels[yKey]} %{y:.2f}`
                       + '<extra></extra>',
      });
      // only the outer edges are labelled, so the inner subplots stay clear
      axes[`xaxis${suffix}`] = axis(row === side - 1 ? labels[xKey] : '', { dtick: 0.5 });
      axes[`yaxis${suffix}`] = axis(col === 0 ? labels[yKey] : '', { dtick: 0.5 });
    }
  }

  plotly().react(el, traces, layout({
    width, height,
    grid: { rows: side, columns: side, pattern: 'independent' },
    ...axes,
  }), CONFIG);
}

/* --- model recovery ------------------------------------------------------- */

export function drawModelRecovery(el, { width = 700, height = 420 } = {}) {
  const short = MODEL_RECOVERY.labels.map((l) => l.replace(', ', '<br>'));
  plotly().react(el, [{
    type: 'heatmap', z: MODEL_RECOVERY.matrix, x: short, y: short,
    xgap: 2, ygap: 2, zmin: 0, zmax: 1,
    colorscale: [[0, '#ffffff'], [1, COL.opt1]],
    colorbar: { title: { text: 'proportion', side: 'right' }, thickness: 11, len: 0.9,
                outlinewidth: 0, tickfont: { size: 13 }, titlefont: { size: 14 } },
    hovertemplate: 'simulated with %{y}<br>picked %{x}<br>%{z:.0%} of datasets<extra></extra>',
  }], layout({
    width, height,
    xaxis: Object.assign(axis('picked by the BIC'), { showgrid: false }),
    yaxis: Object.assign(axis('simulated with'), { showgrid: false, autorange: 'reversed' }),
  }), CONFIG);
}

export const modelRecoveryNote = () =>
  `Over ${MODEL_RECOVERY.nDatasets} simulated datasets, how often the BIC picks each model out. `
  + 'The diagonal is every time it got the right answer: here it is complete, so a BIC comparison '
  + 'on these data can be trusted to tell the four models apart.';

/* --- random effects: which model does each participant use? --------------- */

/**
 * Model frequencies with their exceedance probabilities, as section 3 of the
 * session 3 notebook computes them.
 *
 * This is the random effects comparison: every participant is allowed their own
 * model, and the frequency is the proportion of participants each model
 * accounts for. The exceedance probability on top is the chance that model is
 * the most common one of the four. It is worth noticing that this disagrees
 * with the fixed effects BIC comparison on the same data, which is the point
 * the notebook asks students about.
 */
export function drawModelFrequencies(el, { width = 940, height = 400 } = {}) {
  const { labels, estimated, sd, exceedance } = MODEL_FREQUENCIES;
  const names = labels.map((l) => l.replace(', ', '<br>'));
  const best = Math.max(...exceedance);
  plotly().react(el, [{
    x: names, y: estimated, type: 'bar',
    error_y: { type: 'data', array: sd, color: COL.ink, thickness: 1.5, width: 6 },
    marker: { color: exceedance.map((x) => (x === best ? COL.opt1 : COL.faint)) },
    hovertemplate: '%{x}<br>frequency %{y:.2f}<extra></extra>',
  }], layout({
    width, height, bargap: 0.45,
    xaxis: axis(''),
    yaxis: axis('frequency', { range: [0, 0.72], dtick: 0.2 }),
    annotations: estimated.map((f, i) => ({
      x: names[i], y: f + sd[i] + 0.04, xanchor: 'center', yanchor: 'bottom', showarrow: false,
      text: `exceedance ${exceedance[i].toFixed(3)}`,
      font: { size: 14, color: exceedance[i] === best ? COL.opt1 : COL.muted },
    })),
  }), CONFIG);
}

export const modelFrequenciesNote = () => {
  const { labels, estimated, exceedance } = MODEL_FREQUENCIES;
  const winner = exceedance.indexOf(Math.max(...exceedance));
  return `${labels[winner]} wins, with an exceedance probability of `
    + `${exceedance[winner].toFixed(3)} on a frequency of ${estimated[winner].toFixed(2)} `
    + '— a different model from the one the summed BIC picked. Error bars are the standard '
    + 'deviation of the Dirichlet estimate.';
};
