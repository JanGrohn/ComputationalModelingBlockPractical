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

const PLOT_FONT = { family: 'Inter, Helvetica, Arial, sans-serif', size: 13, color: '#14171a' };
export const COL = { opt1: '#2e7d32', opt2: '#e8700a', ink: '#14171a', muted: '#5b6670', faint: '#b4bcc2' };

/** Plot styling shared by every figure, so the decks read as one system. */
export function layout(extra = {}) {
  return Object.assign({
    font: PLOT_FONT,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 52, r: 16, t: 10, b: 42 },
    showlegend: false,
    hovermode: 'closest',
    xaxis: { zeroline: false, showgrid: false, linecolor: '#c9d0d5', ticks: 'outside', tickcolor: '#c9d0d5' },
    yaxis: { zeroline: false, gridcolor: '#eef1f3', linecolor: '#c9d0d5', ticks: 'outside', tickcolor: '#c9d0d5' },
  }, extra);
}

export const CONFIG = { displayModeBar: false, responsive: true, staticPlot: false };

/**
 * Wire up one interactive figure.
 *
 * @param {string} id        id of the <figure> element
 * @param {object} params    {name: {min, max, step, value, label, format}}
 * @param {function} draw    (el, values) -> void, draws or updates the plot
 * @param {function} note    (values) -> string, the line shown in the PDF
 */
export function interactiveFigure(id, params, draw, note) {
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
    input.addEventListener('input', () => {
      values[key] = parseFloat(input.value);
      out.textContent = fmt(p, values[key]);
      render();
    });
    label.append(name, input, out);
    controls.appendChild(label);
  }

  if (controls.dataset.reseed) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'new schedule';
    btn.addEventListener('click', () => { values.__seed = Math.floor(Math.random() * 1e6); render(); });
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

  // Figures are sized in slide coordinates and drawn on load, so there is
  // nothing to resize and nothing to wait for when the deck is exported.
}
