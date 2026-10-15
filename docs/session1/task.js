/* ---------------------------------------------------------------------------
   A few playable trials of the task, on the slide.

   The real experiment is served from this repository at /task/. This is a
   short stand-in so a slide can be played rather than described, and it
   follows the real task's interaction exactly: both options show the points on
   offer, you press Select on one, the chosen card is tinted, the rewarded
   option keeps its number while the other turns to a dash, and Next choice
   becomes available a second later.
--------------------------------------------------------------------------- */

import { makeRng } from '../assets/models.js';

const P_OPTION1 = 0.75;    // option 1 pays out on three trials in four
const REVERSE_EVERY = 4;   // ...until the contingency starts reversing
const STABLE_TRIALS = 6;
const REVEAL_DELAY = 1000; // how long the outcome shows before you may move on

export function playableTask(rootId, { seed = 20251015, showBlock = false } = {}) {
  const root = document.getElementById(rootId);
  if (!root) return;

  const rng = makeRng(seed);
  let trial = 0;
  let score = 0;
  let state = null;
  let timer = null;

  const el = (tag, cls, parent) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (parent) parent.appendChild(node);
    return node;
  };

  const board = el('div', 'task-board', root);
  const cards = el('div', 'task-cards', board);
  const next = el('button', 'task-next', board);
  next.type = 'button';
  next.textContent = 'Next choice';
  const status = el('p', 'task-status', board);

  /** The probability that option 1 is rewarded on the current trial. */
  function pOption1() {
    if (trial < STABLE_TRIALS) return P_OPTION1;
    const flips = Math.floor((trial - STABLE_TRIALS) / REVERSE_EVERY);
    return flips % 2 === 0 ? 1 - P_OPTION1 : P_OPTION1;
  }

  function nextTrial() {
    state = {
      mag: [1 + Math.floor(rng() * 99), 1 + Math.floor(rng() * 99)],
      oneRewarded: rng() < pOption1(),
      chosen: null,
    };
    render();
  }

  function choose(option) {
    if (state.chosen !== null) return;
    state.chosen = option;
    if ((option === 1) === state.oneRewarded) score += state.mag[option - 1];
    render();
    // the outcome stays up for a moment before you may move on, as in the
    // real task
    next.disabled = true;
    timer = setTimeout(() => { next.disabled = false; }, REVEAL_DELAY);
  }

  next.addEventListener('click', () => {
    if (next.disabled) return;
    clearTimeout(timer);
    trial += 1;
    nextTrial();
  });

  function render() {
    cards.innerHTML = '';
    [1, 2].forEach((option) => {
      const card = el('button', 'task-card', cards);
      card.type = 'button';
      card.disabled = state.chosen !== null;
      if (state.chosen === option) card.dataset.chosen = 'true';

      const img = el('img', null, card);
      img.src = option === 1
        ? '../assets/img/s1/stim-opt1-green.png'
        : '../assets/img/s1/stim-opt2-orange.png';
      img.alt = option === 1 ? 'the green option' : 'the orange option';

      // before a choice both numbers show; afterwards the rewarded option
      // keeps its number and the other turns to a dash
      const rewarded = (option === 1) === state.oneRewarded;
      const pts = el('span', 'task-points', card);
      pts.textContent = state.chosen === null ? state.mag[option - 1]
                      : rewarded ? state.mag[option - 1] : '–';

      const select = el('span', 'task-select', card);
      select.textContent = '✓ Select';
      card.addEventListener('click', () => choose(option));
    });

    next.disabled = state.chosen === null;
    const block = showBlock
      ? ` · ${trial < STABLE_TRIALS ? 'stable block' : 'volatile block'}`
      : '';
    status.textContent = `Trial ${trial + 1}${block} · ${score} points won`;
  }

  nextTrial();
}
