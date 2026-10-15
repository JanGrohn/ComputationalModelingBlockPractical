/* The live figures for session 3. The plots recapping earlier weeks are drawn
   by ../assets/figures-shared.js; the ones here are the new material, all of
   it computed by task-source/precompute-fits.py with the practical's own code. */

import { layout, CONFIG, COL } from '../assets/deck.js';
import {
  plotly, axis, registerAll, blockKey, drawChoices, choicesNote, drawCondition, conditionNote,
  drawMagnitudes, magnitudesNote, drawWeights, weightsNote, drawLikelihood, likelihoodNote,
  drawGridSearch, gridSearchNote, drawRecovery, drawFalsification, drawAscent, ascentNote,
} from '../assets/figures-shared.js';
import { FITS, SIMULATIONS, STATS, CHI2, MODEL_RECOVERY } from '../assets/participant-fits.js';

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
    draw: (el) => drawMagnitudes(el, { width: 1120, height: 150 }),
    note: () => 'Points on offer, redrawn at random every trial, in both conditions alike.',
  },
  'fig-magnitudes-2': {
    params: {},
    draw: (el) => drawMagnitudes(el, { width: 1120, height: 140 }),
    note: () => magnitudesNote,
  },
  'fig-weights': {
    params: { alpha: { min: 0.02, max: 1, step: 0.01, value: 0.15, label: '&alpha;' } },
    draw: (el, v) => drawWeights(el, v, { width: 1000, height: 290 }),
    note: weightsNote,
  },
  'fig-likelihood': { params: {}, draw: drawLikelihood, note: () => likelihoodNote },
  'fig-grid': {
    params: { points: { min: 4, max: 24, step: 1, value: 10, label: 'grid',
                        format: (v) => `${v}×${v}` } },
    draw: (el, v) => drawGridSearch(el, v, { width: 600, height: 340 }),
    note: gridSearchNote,
  },
  'fig-ascent': { params: {}, draw: drawAscent, note: () => ascentNote },
  'fig-recovery-alpha': {
    params: {},
    draw: (el) => drawRecovery(el, 'alpha', 'fitAlpha', 'α', 0.72, { width: 560, height: 360 }),
    note: () => 'Simulated against recovered learning rate, for 36 simulated participants. The '
                + 'dashed line is simulated = recovered.',
  },
  'fig-falsify-same': {
    params: {}, draw: (el) => drawFalsification(el, 'same', { width: 520, height: 270 }),
    note: () => 'Simulated with one learning rate for both blocks: the two recovered distributions '
                + 'sit on top of each other.',
  },
  'fig-falsify-diff': {
    params: {}, draw: (el) => drawFalsification(el, 'different', { width: 520, height: 270 }),
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

/* --- learning rates fitted to the real participants ----------------------- */

/**
 * Both blocks' fitted learning rates, overlaid. This is the plot the session
 * asks students to be sceptical of: it shows the two distributions, but the
 * hypothesis is about each participant's own difference, which it hides.
 */
function alphaDistributions(el, fits, { width = 560, height = 300 } = {}) {
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
function alphaDifference(el, fits, { width = 560, height = 300 } = {}) {
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

const realFits = {
  'fig-alpha-dist': {
    params: {},
    draw: (el) => alphaDistributions(el, FITS.mul2),
    note: () => `Learning rates fitted to all ${N} participants under multiplicative utility. The `
                + 'volatile distribution sits to the right, but this plot compares the two groups '
                + 'rather than each participant with themselves.',
  },
  'fig-alpha-diff': {
    params: {},
    draw: (el) => alphaDifference(el, FITS.mul2),
    note: () => {
      const t = STATS.tTest.mul;
      return `The same fits as a within participant difference, which is what the hypothesis is `
             + `about. A one sample t test against zero gives t(${t.df}) = ${t.t}, `
             + `p = ${t.p.toPrecision(2)}.`;
    },
  },
  'fig-alpha-dist-add': {
    params: {},
    draw: (el) => alphaDistributions(el, FITS.add2),
    note: () => 'The same participants fitted with additive utility instead.',
  },
  'fig-alpha-diff-add': {
    params: {},
    draw: (el) => alphaDifference(el, FITS.add2),
    note: () => {
      const t = STATS.tTest.add;
      return `Under additive utility the difference survives: t(${t.df}) = ${t.t}, `
             + `p = ${t.p.toPrecision(2)}.`;
    },
  },
};

/* --- the likelihood ratio test -------------------------------------------- */

const chiSquare = {
  'fig-chi2': {
    params: {},
    draw(el) {
      const marks = [
        { value: SIMULATIONS.same.test.lambda, label: 'simulated with one learning rate',
          color: COL.stable },
        { value: SIMULATIONS.shuffled.test.lambda, label: 'simulated, volatile shuffled',
          color: COL.muted },
        { value: STATS.likelihoodRatio.mul.lambda, label: 'the real participants',
          color: COL.opt2 },
      ];
      const peak = Math.max(...CHI2.density);
      plotly().react(el, [
        { x: CHI2.x, y: CHI2.density, mode: 'lines', fill: 'tozeroy',
          line: { color: COL.opt1, width: 2 }, fillcolor: 'rgba(46,125,50,0.16)',
          hovertemplate: 'λ %{x}<extra></extra>' },
      ], layout({
        width: 1020, height: 215,
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
          { x: CHI2.critical, y: peak * 1.12, yanchor: 'bottom', xanchor: 'left',
            showarrow: false, text: `p = 0.05 at ${CHI2.critical}`,
            font: { size: 14, color: COL.faint } },
          // past the middle of the axis a left anchored label would run off
          // the edge, so those are hung to the left of their line instead
          ...marks.map((m, i) => {
            const left = m.value < 240;
            return { x: m.value, y: peak * (1.62 - i * 0.2),
              xanchor: left ? 'left' : 'right', showarrow: false,
              text: left ? `  ${m.label}: ${m.value}` : `${m.label}: ${m.value}  `,
              font: { size: 15, color: m.color } };
          }),
        ],
      }), CONFIG);
    },
    note: () => `The null distribution of the statistic, with ${CHI2.df} degrees of freedom, one `
                + `per participant. Beyond ${CHI2.critical} it is significant at p < 0.05.`,
  },
};

/* --- the two control simulations ------------------------------------------ */

const simulations = Object.fromEntries(Object.entries(SIMULATIONS).map(([key, sim]) => [
  `fig-sim-${key}`, {
    params: {},
    draw: (el) => alphaDifference(el, sim, { width: 540, height: 290 }),
    note: () => `Recovered differences when the data were simulated `
                + (key === 'same'
                  ? 'with the same learning rate in both blocks. '
                  : 'with the same learning rates, but shuffled between participants in the '
                    + 'volatile block. ')
                + `The likelihood ratio test gives χ²(${sim.test.df}) = `
                + `${sim.test.lambda}, p = ${sim.test.p.toPrecision(2)}.`,
  }]));

/* --- BIC, and what it says about the four models -------------------------- */

const MODEL_LABELS = {
  mul1: '1 α<br>multiplicative', mul2: '2 α<br>multiplicative',
  add1: '1 α<br>additive', add2: '2 α<br>additive',
};

const bic = {
  'fig-bic': {
    params: {},
    draw(el) {
      const keys = Object.keys(MODEL_LABELS);
      const values = keys.map((k) => STATS.bic[k]);
      const best = Math.min(...values);
      plotly().react(el, [{
        x: keys.map((k) => MODEL_LABELS[k]), y: values, type: 'bar',
        marker: { color: values.map((v) => (v === best ? COL.opt1 : COL.faint)) },
        text: values.map((v) => v.toFixed(0)), textposition: 'outside',
        textfont: { size: 16 },
        hovertemplate: '%{x}<br>BIC %{y:.0f}<extra></extra>',
      }], layout({
        width: 600, height: 320, bargap: 0.45,
        xaxis: axis(''),
        // the differences matter, not the absolute values, so start the axis
        // where the bars differ rather than at zero
        yaxis: axis('summed BIC, lower is better', { range: [9500, 11000] }),
      }), CONFIG);
    },
    note: () => `BIC summed over all ${N} participants. Additive utility with two learning rates `
                + `wins at ${STATS.bic.add2}, against ${STATS.bic.mul2} for multiplicative `
                + `utility with two. The two utility functions are not nested, so this comparison `
                + `cannot be a likelihood ratio test.`,
  },
};

/* --- the parameters fitted to each participant ---------------------------- */

/**
 * Every parameter against every other, for spotting participants at the
 * extremes. Plotly's own splom gives no control over where it puts the axis
 * titles, and at this size it stacks them on top of the tick labels, so the
 * lower triangle is built here as an ordinary grid of subplots instead.
 */
function parameterMatrix(el, fits, labels, { width = 1060, height = 400 } = {}) {
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

const MUL_LABELS = { alphaStable: 'α stable', alphaVolatile: 'α volatile', beta: 'β' };
const ADD_LABELS = { ...MUL_LABELS, omega: 'ω' };

const parameters = {
  'fig-params-mul': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.mul2, MUL_LABELS, { width: 1060, height: 330 }),
    note: () => 'Every fitted parameter against every other, under multiplicative utility. Points '
                + 'pinned at 0 or 1 are the ones to look at: a learning rate of 1 means only the '
                + 'last outcome counted, and a β of 0 means choices were random.',
  },
  'fig-params-add': {
    params: {},
    draw: (el) => parameterMatrix(el, FITS.add2, ADD_LABELS),
    note: () => 'The same under additive utility, with the weight ω on magnitude as a fourth '
                + 'parameter. An ω at 0 or 1 means the participant used only one of the two '
                + 'dimensions.',
  },
};

/* --- model recovery ------------------------------------------------------- */

const modelRecovery = {
  'fig-model-recovery': {
    params: {},
    draw(el) {
      const short = MODEL_RECOVERY.labels.map((l) => l.replace(', ', '<br>'));
      plotly().react(el, [{
        type: 'heatmap', z: MODEL_RECOVERY.matrix, x: short, y: short,
        xgap: 2, ygap: 2, zmin: 0, zmax: 1,
        colorscale: [[0, '#ffffff'], [1, COL.opt1]],
        colorbar: { title: { text: 'proportion', side: 'right' }, thickness: 11, len: 0.9,
                    outlinewidth: 0, tickfont: { size: 13 }, titlefont: { size: 14 } },
        hovertemplate: 'simulated with %{y}<br>picked %{x}<br>%{z:.0%} of datasets<extra></extra>',
      }], layout({
        width: 700, height: 420,
        xaxis: Object.assign(axis('picked by the BIC'), { showgrid: false }),
        yaxis: Object.assign(axis('simulated with'), { showgrid: false, autorange: 'reversed' }),
      }), CONFIG);
    },
    note: () => `Over ${MODEL_RECOVERY.nDatasets} simulated datasets, how often the BIC picks each `
                + `model out. The diagonal is every time it got the right answer: here it is `
                + `complete, so a BIC comparison on these data can be trusted to tell the four `
                + `models apart.`,
  },
};

registerAll(recaps, realFits, chiSquare, simulations, bic, parameters, modelRecovery);
