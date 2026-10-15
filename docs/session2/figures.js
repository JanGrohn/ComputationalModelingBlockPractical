/* The live figures for session 2. Each one mirrors a cell of session2.ipynb.
   The plots sessions 3 and 4 show again are drawn by ../assets/figures-shared.js. */

import { softmax } from '../assets/models.js';
import { layout, CONFIG, COL } from '../assets/deck.js';
import {
  plotly, axis, registerAll, drawChoices, choicesNote, drawCondition,
  drawMagnitudes, drawLikelihood, likelihoodNote, drawRecovery, recoveryNote,
  drawFalsification, drawAscent, ascentNote, ASCENT_STEPS, drawWeights, weightsNote,
  drawGridSearch, gridSearchNote, drawErrorTerm, errorTermNote, drawBlockLandscapes,
  blockLandscapesNote, drawRegression, regressionNote, drawSoftmaxPair, softmaxPairNote,
} from '../assets/figures-shared.js';
import { REGRESSION } from '../assets/participant-fits.js';

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
          name: 'softmax', line: { color: COL.option1, width: 3 },
          hovertemplate: 'u<sub>1</sub> − u<sub>2</sub> = %{x}<br>P(c<sub>1</sub>) = %{y:.2f}<extra></extra>' },
      ], layout({
        width: 1080, height: 300,
        xaxis: Object.assign(axis('utility 1 − utility 2  (u<sub>1</sub> − u<sub>2</sub>)'),
                             { zeroline: true, zerolinecolor: '#e2e6ea' }),
        yaxis: axis('probability of choosing option 1',
                    { range: [-0.04, 1.2], tickvals: [0, 0.25, 0.5, 0.75, 1] }),
        annotations: [
          { x: -98, y: 1.13, xanchor: 'left', showarrow: false, text: '··· deterministic rule',
            font: { size: 16, color: COL.muted } },
          { x: -30, y: 1.13, xanchor: 'left', showarrow: false, text: '— softmax',
            font: { size: 16, color: COL.option1 } },
        ],
      }), CONFIG);
    },
    note: (v) => `Shown with an inverse temperature of ${v.beta.toFixed(2)}. The dotted line is `
                 + `the deterministic rule, which always takes the better option; the green curve `
                 + `is the softmax, which takes it most of the time. A larger β makes the curve `
                 + `steeper, so behaviour approaches the deterministic rule.`,
  },
};

/* --- the deterministic rule, on the same axes as the softmax -------------- */

const deterministic = {
  'fig-deterministic': {
    params: {},
    draw(el) {
      const tint = (colour) => ({ type: 'rect', xref: 'x', yref: 'paper', y0: 0, y1: 1,
        line: { width: 0 }, fillcolor: colour, layer: 'below' });
      plotly().react(el, [
        // drawn as two separate segments, because the rule really does jump:
        // joining them would draw a vertical line that is not part of it
        { x: [-100, 0], y: [0, 0], mode: 'lines', line: { color: COL.option1, width: 3 },
          hovertemplate: 'always choose option 2<extra></extra>' },
        { x: [0, 100], y: [1, 1], mode: 'lines', line: { color: COL.option1, width: 3 },
          hovertemplate: 'always choose option 1<extra></extra>' },
        { x: [0], y: [0.5], mode: 'markers',
          marker: { size: 11, color: '#fff', line: { color: COL.ink, width: 2 } },
          hovertemplate: 'indifference: the two are worth the same<extra></extra>' },
      ], layout({
        width: 620, height: 340,
        shapes: [
          Object.assign(tint('rgba(237,108,2,0.08)'), { x0: -100, x1: 0 }),
          Object.assign(tint('rgba(46,125,50,0.08)'), { x0: 0, x1: 100 }),
          { type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 0.78,
            line: { color: COL.faint, width: 1.5, dash: 'dot' } },
        ],
        xaxis: Object.assign(axis('utility 1 \u2212 utility 2  (u<sub>1</sub> \u2212 u<sub>2</sub>)'),
                             { range: [-100, 100], tickvals: [-100, -50, 0, 50, 100] }),
        yaxis: axis('probability of choosing option 1', { range: [-0.12, 1.42], tickvals: [0, 0.5, 1] }),
        annotations: [
          { x: -52, y: 1.22, xanchor: 'center', showarrow: false,
            text: 'u<sub>2</sub> &gt; u<sub>1</sub><br>always choose option 2',
            font: { size: 16, color: COL.option2Text } },
          { x: 52, y: 1.22, xanchor: 'center', showarrow: false,
            text: 'u<sub>1</sub> &gt; u<sub>2</sub><br>always choose option 1',
            font: { size: 16, color: COL.option1 } },
          { x: 0, y: 0.5, ax: 46, ay: -26, text: 'indifference point',
            font: { size: 15, color: COL.ink }, arrowcolor: COL.ink, arrowhead: 2,
            arrowsize: 0.8, arrowwidth: 1, xanchor: 'left' },
        ],
      }), CONFIG);
    },
    note: () => 'The deterministic rule: whichever option is worth more is taken every time, so '
                + 'the probability of choosing option 1 jumps straight from 0 to 1. The jump sits at the '
                + 'indifference point, where the two options are worth the same.',
  },
};

/* --- the figures the later sessions show again ---------------------------- */

const choiceFigure = {
  'fig-choices': {
    params: {
      alpha: { min: 0.01, max: 1, step: 0.01, value: 0.3, label: '&alpha;' },
      beta:  { min: 0.01, max: 0.5, step: 0.01, value: 0.05, label: '&beta;' },
    },
    draw: drawChoices,
    note: choicesNote,
  },
};

const conditions = Object.fromEntries([1, 2].map((condition) => [`fig-condition-${condition}`, {
  params: {},
  draw: (el) => drawCondition(el, condition, { width: 530, height: 160 }),
  // the tag beside each title already names the condition
  note: () => (condition === 1 ? 'P(option 1 rewarded) holds at 0.75.'
                               : 'P(option 1 rewarded) reverses every 25 trials.'),
}]));

const magnitudes = {
  'fig-magnitudes': {
    params: {},
    draw: (el) => drawMagnitudes(el, { width: 1120, height: 125 }),
    note: () => 'Magnitudes, redrawn at random every trial.',
  },
};

const landscape = {
  'fig-likelihood': {
    params: {},
    draw: (el) => drawLikelihood(el, { width: 470, height: 360 }),
    note: () => likelihoodNote,
  },
  'fig-block-landscapes': {
    params: {},
    draw: (el) => drawBlockLandscapes(el, { width: 700, height: 360 }),
    note: () => blockLandscapesNote,
  },
  'fig-grid': {
    params: { points: { min: 4, max: 24, step: 1, value: 10, label: 'grid',
                        format: (v) => `${v}\u00d7${v}` } },
    draw: (el, v) => drawGridSearch(el, v, { width: 560, height: 340 }),
    note: gridSearchNote,
  },
  'fig-error-term': {
    params: {},
    draw: (el) => drawErrorTerm(el, { width: 560, height: 250 }),
    note: () => errorTermNote,
  },
};

const recovery = Object.fromEntries([
  ['fig-recovery-alpha', 'alpha', 'fitAlpha', '\u03b1', 0.72, 'Learning rate'],
  ['fig-recovery-beta', 'beta', 'fitBeta', '\u03b2', 0.32, 'Inverse temperature'],
  ['fig-recovery-alpha-2', 'alpha', 'fitAlpha', '\u03b1', 0.72, 'Learning rate'],
  ['fig-recovery-beta-2', 'beta', 'fitBeta', '\u03b2', 0.32, 'Inverse temperature'],
].map(([id, key, fitKey, label, hi, name]) => [id, {
  params: {},
  draw: (el) => drawRecovery(el, key, fitKey, label, hi, { width: 540, height: 300 }),
  note: () => recoveryNote(name),
}]));

const weights = Object.fromEntries(['fig-weights'].map((id) => [id, {
  params: { alpha: { min: 0.02, max: 1, step: 0.01, value: 0.15, label: '&alpha;' } },
  draw: (el, v) => drawWeights(el, v, { width: 1000, height: id === 'fig-weights' ? 340 : 200 }),
  note: weightsNote,
}]));

const regression = {
  'fig-regression': {
    params: {},
    draw: (el) => drawRegression(el, REGRESSION, { width: 1100, height: 380 }),
    note: () => regressionNote(REGRESSION),
  },
  'fig-softmax-pair': {
    params: {},
    draw: (el) => drawSoftmaxPair(el, { width: 640, height: 360 }),
    note: () => softmaxPairNote(),
  },
};

const falsification = {
  'fig-falsify-same-2': {
    params: {}, draw: (el) => drawFalsification(el, 'same'),
    note: () => 'Simulated with one learning rate for both blocks.',
  },
  'fig-falsify-diff-2': {
    params: {}, draw: (el) => drawFalsification(el, 'different'),
    note: () => 'Simulated with a higher learning rate in the volatile block.',
  },
  'fig-falsify-same': {
    params: {}, draw: (el) => drawFalsification(el, 'same', { width: 520, height: 240 }),
    note: () => 'Simulated with one learning rate for both blocks: the two recovered distributions '
                + 'sit on top of each other, as they should.',
  },
  'fig-falsify-diff': {
    params: {}, draw: (el) => drawFalsification(el, 'different', { width: 520, height: 240 }),
    note: () => 'Simulated with a higher learning rate in the volatile block: the difference comes '
                + 'back out, so the task can find it when it is really there.',
  },
};

const ascent = {
  'fig-ascent': {
    params: { step: { min: 1, max: ASCENT_STEPS, step: 1, value: ASCENT_STEPS,
                      label: 'step', format: (v) => String(Math.round(v)) } },
    draw: (el, v) => drawAscent(el, v),
    note: ascentNote,
    opts: { play: { param: 'step', fps: 14 } },
  },
};

registerAll(softmaxFigure, deterministic, choiceFigure, conditions, magnitudes, landscape,
            recovery, weights, regression, falsification, ascent);
