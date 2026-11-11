/* ---------------------------------------------------------------------------
   A few playable trials of the task, on a slide. Shared by every deck.

   The real experiment is served from this repository at /task/. This is a
   short stand-in so a slide can be played rather than described, and it
   follows the real task's interaction: both options show the points on offer,
   you press Select on one, the chosen card is tinted, and the rewarded option
   keeps its number while the other turns to a dash.

   Two things the real task has are left out here, because a lecture is not an
   experiment: there is no Next choice button, since on a slide the next trial
   should simply arrive, and no running score, which would invite the audience
   to play for points rather than watch what the model has to learn.
--------------------------------------------------------------------------- */

import { makeRng } from './models.js';
import { sync } from './deck.js';

const P_OPTION1 = 0.75;    // option 1 pays out on three trials in four
const REVERSE_EVERY = 4;   // ...until the contingency starts reversing
const STABLE_TRIALS = 6;
const REVEAL_DELAY = 1600; // how long the outcome shows before the next trial

export function playableTask(rootId, { seed = 20251015, showBlock = false } = {}) {
  const root = document.getElementById(rootId);
  if (!root) return;

  const rng = makeRng(seed);
  let trial = 0;
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

  // a choice made in the speaker view is made on the main screen too, and the
  // other way round; both draw the same trials from the same seed
  const send = sync(rootId, ({ option }) => choose(option));

  function choose(option) {
    if (state.chosen !== null) return;
    state.chosen = option;
    render();
    // the outcome stays up for a moment, then the next trial arrives on its
    // own: on a slide there is nothing useful for a Next button to do
    clearTimeout(timer);
    timer = setTimeout(() => { trial += 1; nextTrial(); }, REVEAL_DELAY);
  }

  function render() {
    cards.innerHTML = '';
    [1, 2].forEach((option) => {
      const card = el('button', 'task-card', cards);
      card.type = 'button';
      card.disabled = state.chosen !== null;
      if (state.chosen === option) card.dataset.chosen = 'true';

      // the very files the real task imports, not copies of them: the PNG
      // exports used before carried a white background, which showed as a
      // white box as soon as the chosen card was tinted
      const img = el('img', null, card);
      img.src = option === 1
        ? new URL('./img/stimuli/deer-silhouette.svg', import.meta.url).href
        : new URL('./img/stimuli/bird-crane-shape.svg', import.meta.url).href;
      img.alt = option === 1 ? 'the green option' : 'the orange option';

      // before a choice both numbers show; afterwards the rewarded option
      // keeps its number and the other turns to a dash
      const rewarded = (option === 1) === state.oneRewarded;
      const pts = el('span', 'task-points', card);
      pts.textContent = state.chosen === null ? state.mag[option - 1]
                      : rewarded ? state.mag[option - 1] : '–';

      const select = el('span', 'task-select', card);
      select.textContent = '✓ Select';
      card.addEventListener('click', () => { choose(option); send({ option }); });
    });

    const block = showBlock
      ? ` · ${trial < STABLE_TRIALS ? 'stable block' : 'volatile block'}`
      : '';
    status.textContent = `Trial ${trial + 1}${block}`;
  }

  nextTrial();
}
