/* ---------------------------------------------------------------------------
   A few playable trials of the task, on the slide.

   The real experiment is served from this repository at /task/. This is a
   short stand-in so the recap slide can be played rather than described: same
   two stimuli, same independent draw of magnitude and probability, same
   anticorrelated outcome. It runs a stable block then a volatile one so the
   manipulation is visible within a handful of trials.
--------------------------------------------------------------------------- */

import { makeRng } from '../assets/models.js';

const STABLE_P = 0.8;      // option 1 pays out on 80% of stable trials
const REVERSE_EVERY = 4;   // the volatile block flips this often
const STABLE_TRIALS = 6;

export function playableTask(rootId, seed = 20251015) {
  const root = document.getElementById(rootId);
  if (!root) return;

  const rng = makeRng(seed);
  let trial = 0;
  let score = 0;
  let state = null;

  const el = (tag, cls, parent) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (parent) parent.appendChild(n);
    return n;
  };

  const board = el('div', 'task-board', root);
  const cards = el('div', 'task-cards', board);
  const feedback = el('p', 'task-feedback', board);
  const status = el('p', 'task-status', board);

  function trueP() {
    // stable block first, then reversals
    if (trial < STABLE_TRIALS) return STABLE_P;
    const flips = Math.floor((trial - STABLE_TRIALS) / REVERSE_EVERY);
    return flips % 2 === 0 ? 1 - STABLE_P : STABLE_P;
  }

  function nextTrial() {
    state = {
      mag: [1 + Math.floor(rng() * 100), 1 + Math.floor(rng() * 100)],
      opt1Rewarded: rng() < trueP() ? 1 : 0,
      answered: false,
    };
    render();
  }

  function choose(option) {
    if (state.answered) return;
    state.answered = true;
    const won = (option === 1) === (state.opt1Rewarded === 1);
    if (won) score += state.mag[option - 1];
    feedback.textContent = won
      ? `Rewarded. You win ${state.mag[option - 1]} points.`
      : 'Not rewarded this time. No points.';
    feedback.dataset.outcome = won ? 'win' : 'loss';
    trial += 1;
    render();
    setTimeout(nextTrial, 1100);
  }

  function render() {
    cards.innerHTML = '';
    [1, 2].forEach((option) => {
      const card = el('button', 'task-card', cards);
      card.type = 'button';
      card.dataset.option = option;
      card.disabled = state.answered;
      if (state.answered) {
        const rewarded = (option === 1) === (state.opt1Rewarded === 1);
        card.dataset.result = rewarded ? 'rewarded' : 'not-rewarded';
      }
      const img = el('img', null, card);
      img.src = option === 1
        ? '../assets/img/s1/stim-opt1-green.png'
        : '../assets/img/s1/stim-opt2-orange.png';
      img.alt = option === 1 ? 'the green option' : 'the orange option';
      const pts = el('span', 'task-points', card);
      pts.textContent = state.mag[option - 1];
      const select = el('span', 'task-select', card);
      select.textContent = '\u2713 Select';
      card.addEventListener('click', () => choose(option));
    });
    const block = trial < STABLE_TRIALS ? 'stable block' : 'volatile block';
    status.textContent = `Trial ${trial + 1} · ${block} · ${score} points`;
    if (!state.answered) {
      feedback.textContent = 'Pick one.';
      delete feedback.dataset.outcome;
    }
  }

  nextTrial();
}
