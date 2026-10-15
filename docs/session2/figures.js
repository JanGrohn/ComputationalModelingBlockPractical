/* The live figures for session 2. Each one mirrors a cell of session2.ipynb.
   The plots sessions 3 and 4 show again are drawn by ../assets/figures-shared.js. */

import { softmax } from '../assets/models.js';
import { layout, CONFIG, COL } from '../assets/deck.js';
import {
  plotly, axis, registerAll, drawChoices, choicesNote, drawCondition, conditionNote,
  drawMagnitudes, magnitudesNote, drawLikelihood, likelihoodNote, drawRecovery,
  drawFalsification, drawAscent, ascentNote,
} from '../assets/figures-shared.js';

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
        xaxis: Object.assign(axis('utility of option 1 − utility of option 2'),
                             { zeroline: true, zerolinecolor: '#e2e6ea' }),
        yaxis: axis('P(choose option 1)',
                    { range: [-0.04, 1.2], tickvals: [0, 0.25, 0.5, 0.75, 1] }),
        annotations: [
          { x: -98, y: 1.13, xanchor: 'left', showarrow: false, text: '··· deterministic rule',
            font: { size: 16, color: COL.muted } },
          { x: -30, y: 1.13, xanchor: 'left', showarrow: false, text: '— softmax',
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
  draw: (el) => drawCondition(el, condition),
  note: () => conditionNote(condition),
}]));

const magnitudes = {
  'fig-magnitudes': { params: {}, draw: drawMagnitudes, note: () => magnitudesNote },
};

const landscape = {
  'fig-likelihood': { params: {}, draw: drawLikelihood, note: () => likelihoodNote },
};

const recovery = {
  'fig-recovery-alpha-2': {
    params: {},
    draw: (el) => drawRecovery(el, 'alpha', 'fitAlpha', 'α', 0.72),
    note: () => 'Simulated against recovered learning rate, for 36 simulated participants.',
  },
  'fig-recovery-alpha': {
    params: {},
    draw: (el) => drawRecovery(el, 'alpha', 'fitAlpha', 'α', 0.72),
    note: () => 'Learning rate: 36 simulated participants, each fitted back with the same model. '
                + 'The dashed line is simulated = recovered.',
  },
  'fig-recovery-beta': {
    params: {},
    draw: (el) => drawRecovery(el, 'beta', 'fitBeta', 'β', 0.32),
    note: () => 'Inverse temperature, the same 36 participants. It recovers less tightly than the '
                + 'learning rate, which is what the ridge in the likelihood landscape predicts.',
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
    params: {}, draw: (el) => drawFalsification(el, 'same'),
    note: () => 'Simulated with one learning rate for both blocks: the two recovered distributions '
                + 'sit on top of each other, as they should.',
  },
  'fig-falsify-diff': {
    params: {}, draw: (el) => drawFalsification(el, 'different'),
    note: () => 'Simulated with a higher learning rate in the volatile block: the difference comes '
                + 'back out, so the task can find it when it is really there.',
  },
};

const ascent = { 'fig-ascent': { params: {}, draw: drawAscent, note: () => ascentNote } };

registerAll(softmaxFigure, choiceFigure, conditions, magnitudes, landscape,
            recovery, falsification, ascent);
