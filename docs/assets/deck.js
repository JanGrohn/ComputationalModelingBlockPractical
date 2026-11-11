/* ---------------------------------------------------------------------------
   Shared deck machinery: reveal setup, LaTeX rendering, and the helper that
   turns a <figure data-figure="..."> into a live Plotly plot with sliders.

   Two rules hold the whole thing together:
     1. Every plot is drawn on load, not when its slide is reached, so that
        printing to PDF catches all of them.
     2. Every control writes its current value into a .fig-print-note, which
        is hidden on screen and shown in print. A printed figure therefore
        always says what it is set to.
--------------------------------------------------------------------------- */

const PLOT_FONT = { family: 'Inter, Helvetica, Arial, sans-serif', size: 16, color: '#14171a' };
// The course palette. Each colour stands for exactly one thing, here, in the
// notebooks (sessionN/palette.py) and in the task, so a colour can be read the
// same way wherever it is seen. task-source/palette-test.mjs checks they agree.
export const COL = {
  option1:  '#2e7d32',  // option 1, the green deer, and the model's probabilities about it
  option2:  '#ed6c02',  // option 2, the orange crane
  stable:   '#1f4b73',  // the stable block
  volatile: '#c2185b',  // the volatile block
  estimate: '#6a3d9a',  // a fitted or estimated value
  truth:    '#14171a',  // the true values a simulation used; always drawn dashed
  neutral:  '#7d868d',  // data with no category of its own

  // Not data: the darkened orange for words in option 2's colour (the marks'
  // orange fails WCAG at text sizes), and the type and annotation greys.
  option2Text: '#b35309',
  ink: '#14171a', muted: '#5b6670', faint: '#9aa3aa',
};

// Continuous quantities: perceptually uniform, and no single flat colour in it
// competes with the categorical colours above.
export const CONTINUOUS = 'Viridis';

/** The margins for a figure with a key or titles above its plotting area,
    which automargin does not count when it sizes the margins. */
export const HEADROOM = { l: 10, r: 16, t: 40, b: 10 };

/** Plot styling shared by every figure, so the decks read as one system. */
export function layout(extra = {}) {
  return Object.assign({
    font: PLOT_FONT,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    // automargin on both axes below sizes the margins itself, so these are
    // only a floor: an axis title can never end up on its own tick labels
    margin: { l: 10, r: 16, t: 10, b: 10 },
    showlegend: false,
    hovermode: 'closest',
    // no zooming or panning anywhere: a slide is not a place to explore, and
    // a stray drag during a lecture leaves the figure in a state nobody wants
    dragmode: false,
    xaxis: { zeroline: false, showgrid: false, linecolor: '#c9d0d5', ticks: 'outside',
             tickcolor: '#c9d0d5', automargin: true, title: { standoff: 14 } },
    yaxis: { zeroline: false, gridcolor: '#eef1f3', linecolor: '#c9d0d5', ticks: 'outside',
             tickcolor: '#c9d0d5', automargin: true, title: { standoff: 14 } },
  }, extra);
}

export const CONFIG = {
  displayModeBar: false,
  responsive: true,
  staticPlot: false,   // hover tooltips stay
  scrollZoom: false,
  doubleClick: false,
};

/**
 * Keeps each interactive element in step across every window showing this
 * deck. The speaker view draws the deck in frames of its own, so without this
 * a slider moved there would stay put on the main screen, and the other way
 * round. A window that opens late asks for each element's state, and only
 * windows where that element has been changed answer.
 */
const channel = new BroadcastChannel(`deck:${window.location.pathname}`);
const synced = new Map();

channel.onmessage = ({ data }) => {
  const element = synced.get(data.id);
  if (!element) return;   // not set up in this window yet: it will ask once it is
  if (data.ask) {
    if (element.changed) channel.postMessage({ id: data.id, change: element.state() });
  } else {
    element.changed = true;
    element.apply(data.change);
  }
};

/**
 * Register an interactive element for syncing.
 *
 * @param {string} id          the same in every window
 * @param {function} apply     (change) -> void, applies a change another window made
 * @param {function} [state]   () -> change, the element's whole state, for late windows
 * @returns {function}         (change) -> void, tells the other windows about a change
 */
export function sync(id, apply, state) {
  const element = { apply, state, changed: false };
  synced.set(id, element);
  if (state) channel.postMessage({ id, ask: true });
  return (change) => {
    element.changed = true;
    channel.postMessage({ id, change });
  };
}

/**
 * Wire up one interactive figure.
 *
 * @param {string} id        id of the <figure> element
 * @param {object} params    {name: {min, max, step, value, label, format}}
 * @param {function} draw    (el, values) -> void, draws or updates the plot
 * @param {function} note    (values) -> string, the line shown in the PDF
 * @param {object} [opts]    {play: {param, fps}} adds a play button that runs
 *                           that parameter from its minimum to its maximum
 */
export function interactiveFigure(id, params, draw, note, { play } = {}) {
  const fig = document.getElementById(id);
  if (!fig) return;
  const plot = fig.querySelector('.fig-plot');
  const controls = fig.querySelector('.fig-controls');
  const printNote = fig.querySelector('.fig-print-note');

  const values = {};
  for (const [key, p] of Object.entries(params)) values[key] = p.value;

  const fmt = (p, v) => (p.format ? p.format(v) : v.toFixed(p.step < 0.01 ? 3 : 2));

  const render = () => {
    draw(plot, values);
    if (printNote && note) printNote.textContent = note(values);
  };

  /** Set values, keeping each slider and its readout in step, and redraw. */
  const sliders = {};
  const set = (change) => {
    for (const [key, v] of Object.entries(change)) {
      values[key] = v;
      if (sliders[key]) {
        sliders[key].input.value = v;
        sliders[key].out.textContent = fmt(params[key], v);
      }
    }
    render();
  };
  const send = sync(id, set, () => ({ ...values }));
  /** A change made in this window: apply it here, then everywhere else. */
  const change = (c) => { set(c); send(c); };

  for (const [key, p] of Object.entries(params)) {
    const label = document.createElement('label');
    const name = document.createElement('span');
    name.className = 'ctrl-name';
    name.innerHTML = p.label ?? key;
    const input = document.createElement('input');
    input.type = 'range';
    input.min = p.min; input.max = p.max; input.step = p.step; input.value = p.value;
    input.setAttribute('aria-label', (p.label ?? key).replace(/<[^>]+>/g, ''));
    const out = document.createElement('output');
    out.textContent = fmt(p, p.value);
    sliders[key] = { input, out };
    input.addEventListener('input', () => change({ [key]: parseFloat(input.value) }));
    label.append(name, input, out);
    controls.appendChild(label);
  }

  if (play) {
    const p = params[play.param];
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'fig-play';
    let timer = null;
    const stop = () => { clearInterval(timer); timer = null; btn.textContent = '\u25b6 Play'; };
    stop();
    btn.addEventListener('click', () => {
      if (timer) { stop(); return; }
      // a second press after it has finished starts again from the beginning
      if (values[play.param] >= p.max) change({ [play.param]: p.min });
      btn.textContent = '\u275a\u275a Pause';
      timer = setInterval(() => {
        const next = values[play.param] + (p.step ?? 1);
        change({ [play.param]: Math.min(next, p.max) });
        if (next >= p.max) stop();
      }, 1000 / (play.fps ?? 12));
    });
    controls.appendChild(btn);
  }

  if (controls.dataset.reseed) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'new schedule';
    btn.addEventListener('click', () => change({ __seed: Math.floor(Math.random() * 1e6) }));
    controls.appendChild(btn);
  }

  render();
}

/** Render every $...$ and $$...$$ on the page with KaTeX. */
function renderMath() {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
    trust: true,
  });
}

/** Put the deck title and a slide counter at the foot of every slide. */
function addFooters(deckTitle) {
  document.querySelectorAll('.reveal .slides > section').forEach((s) => {
    if (s.classList.contains('title-slide') || s.querySelector('.deck-footer')) return;
    const f = document.createElement('div');
    f.className = 'deck-footer';
    f.innerHTML = `<span>${deckTitle}</span>`;
    s.appendChild(f);
  });
}

/**
 * Set the text of each slide that carries words and nothing else as large as
 * it will go, up to a third larger than usual. The original decks set such
 * slides large, with no figure to make room for; how far each can grow depends
 * on how much it says, so it is measured rather than fixed.
 */
const VISUAL = '.fig, .mermaid-figure, .overview, .pipeline, img, .task-demo, .weeks, .eq, .measure, .katex-display';
const TEXT_SCALES = [1.32, 1.24, 1.16, 1.08, 1];

function fitTextSlides() {
  const canvas = document.querySelector('.reveal .slides');
  document.querySelectorAll('.reveal .slides > section').forEach((s) => {
    if (s.matches('.title-slide, .section-slide') || s.querySelector(VISUAL)) return;
    s.classList.add('text-slide');
    // measured while forced into view, since reveal hides every slide but one,
    // and with transitions off: reveal animates every property of a fragment,
    // so a fragment would otherwise still be at its old size when measured
    const shown = s.style.display;
    s.style.display = 'block';
    s.classList.add('measuring');
    const scale = canvas.getBoundingClientRect().height / 720;
    const top = s.getBoundingClientRect().top;
    const at = (e) => (e.getBoundingClientRect().top - top) / scale;
    const limit = s.querySelector('.citation') ? at(s.querySelector('.citation'))
      : at(s.querySelector('.deck-footer > *') || s.querySelector('.deck-footer'));
    const content = [...s.children].filter((e) => !e.matches('.deck-footer, .citation, aside'));
    const bottom = () => Math.max(...content.map((e) => (e.getBoundingClientRect().bottom - top) / scale));
    for (const k of TEXT_SCALES) {
      s.style.setProperty('--text-scale', k);
      if (bottom() <= limit - 8) break;
    }
    s.classList.remove('measuring');
    s.style.display = shown;
  });
}

export function initDeck({ title }) {
  addFooters(title);
  renderMath();

  // ?export swaps the live controls for the line of text that says what each
  // figure is set to. DeckTape renders screen media, so this cannot be done
  // with an @media print rule.
  if (/[?&]export\b/.test(window.location.search)) {
    document.documentElement.classList.add('exporting');
    // Flatten the click-to-reveal builds: without this the exporter would
    // emit one page per build step, so a slide with four reveals would take
    // four near-identical pages in the handout.
    document.querySelectorAll('.fragment').forEach((el) => el.classList.remove('fragment'));
  }

  Reveal.initialize({
    width: 1280,
    height: 720,
    margin: 0.07,
    minScale: 0.2,
    maxScale: 1.6,
    hash: true,
    // top-aligned: a content slide's title must sit in the same place on
    // every slide, which reveal's vertical centring would break
    center: false,
    slideNumber: 'c/t',
    transition: 'none',
    controls: false,   // a clicker or the arrow keys; keeps the screen clean
    progress: true,
    // one page per slide in the PDF, with incremental builds collapsed
    pdfSeparateFragments: false,
    pdfMaxPagesPerSlide: 1,
    plugins: [RevealNotes],
  });
  // measured once the webfont is in: the fallback font sets narrower, so a
  // slide sized against it would overflow once Inter arrives
  Reveal.on('ready', () => document.fonts.ready.then(fitTextSlides));

  // Figures are sized in slide coordinates and drawn on load, so there is
  // nothing to resize and nothing to wait for when the deck is exported.
}
