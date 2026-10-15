# Slides

The lecture slides, served as a GitHub Pages site from this `docs/` folder.
They behave like PowerPoint in fullscreen, the plots are live, and they export
to PDF one page per slide.

Live at <https://jangrohn.github.io/ComputationalModelingBlockPractical/>.

## Presenting

| Key | |
|---|---|
| `F` | fullscreen |
| `←` `→` or a clicker | move between slides |
| `Esc` | overview of every slide |
| `S` | speaker view: notes, timer, next slide |

The figures are live. Dragging a slider re-runs the model in the browser —
there is no Python behind them, so nothing has to be installed and nothing can
time out mid-lecture.

## Exporting to PDF

```bash
./docs/export-pdf.sh            # every deck
./docs/export-pdf.sh session1   # just one
```

PDFs land in `docs/pdf/`, which is gitignored. Needs `node` and Google Chrome;
the script installs nothing permanently.

Every interactive figure carries a sentence, hidden on screen and shown in the
export, stating the parameter values it is drawn with. That is what keeps the
PDF readable once the sliders are dead — see `.fig-print-note` in
`assets/theme.css`.

## How it is put together

```
docs/
  index.html             landing page listing the four sessions
  assets/
    theme.css            the whole design system; edit colours here
    models.js            the RL model, utilities and softmax, in JavaScript
    fitting.js           grid search and gradient ascent, enough to run in a browser
    deck.js              reveal setup, KaTeX, and the interactive-figure helper
    figures-shared.js    the plots more than one session shows
    figures-fits.js      the plots drawn from the fits to real participants
    simulated-fits.js    generated: sessions 2-4's simulated results
    participant-fits.js  generated: the Blain and Rutledge fits, and the tests on them
    task-demo.js         the playable task, shared by the decks
    img/                 figures taken from the original decks
  sessionN/
    index.html           the slides
    figures.js           which figure goes on which slide, at what size
    diagrams/*.mmd       mermaid sources, pre-rendered into the HTML
  export-pdf.sh
```

`assets/models.js` is a deliberate line-for-line port of the Python in
`sessionN/fitting.py`. If you change a model in the notebooks, change it there
too, or a slide and the practical will quietly disagree.

### The generated data files

Two files under `assets/` are written by scripts and should not be edited:

```bash
cd task-source
node precompute-session2.mjs   # -> assets/simulated-fits.js
python3 precompute-fits.py     # -> assets/participant-fits.js; needs the notebooks' requirements
```

The first simulates and refits participants for the recovery, falsification and
gradient-ascent figures. The second runs the practical's own
`session3.fitting` and `session4.fitting` over the Blain and Rutledge data, so
every number sessions 3 and 4 put on a slide — the likelihood ratio tests, the
BIC totals, the model recovery — is the number students will get from the
notebook. Re-run them if a model or a saved fit changes.

### Writing a slide

Slides are plain HTML `<section>` elements. The pieces the theme provides:

```html
<section>
  <h2>Title, always top left</h2>

  <div class="cols">            <!-- or .cols.narrow-left / .narrow-right -->
    <div>…</div>
    <div>…</div>
  </div>

  <div class="eq">              <!-- LaTeX, rendered by KaTeX -->
    $$p_{t+1} = p_t + \alpha \delta_t$$
    <span class="eq-tag">(2)</span>
  </div>

  <p class="takeaway">The one thing to remember.</p>

  <div class="discuss">
    <span class="discuss-label">Discuss with your neighbour</span>
    <ul><li>…</li></ul>
  </div>

  <p class="citation">Behrens et al., <em>Nature Neuroscience</em>, 2007</p>
  <aside class="notes">Only visible in speaker view.</aside>
</section>
```

Each colour means exactly one thing, here, in the notebooks and in the task:

| colour | means |
|---|---|
| green `#2e7d32` | option 1, the deer: its outcomes, magnitude, utility, choices of it, and the model's probabilities about it |
| orange `#ed6c02` | option 2, the crane |
| blue `#1f4b73` | the stable block |
| raspberry `#c2185b` | the volatile block |
| purple `#6a3d9a` | a fitted or estimated value |
| ink, dashed | the true values a simulation used |
| grey `#7d868d` | data with no category of its own |
| viridis | continuous quantities: likelihoods, utilities, proportions |

Interface chrome uses ink and greys only, so a colour on a slide always says
something about the data. The palette is written down in `assets/deck.js`,
`assets/theme.css` and each `sessionN/palette.py`, since Colab copies one
session folder at a time; `task-source/palette-test.mjs` fails if those or the
task's stimuli ever disagree, or if a live figure draws a colour outside them,
and `task-source/notebook-colours-test.py` checks every notebook figure the same
way. Nothing is distinguished by colour alone, so the greyscale print still
works.

Equations are the notebooks' own, character for character, with the same
numbers; an equation the notebooks do not number is not numbered here either.

Type is one sans family throughout, separated by weight rather than by switching to a serif; maths is
left to KaTeX's own faces, where a serif is the convention and matches how the
notebooks render.

Diagrams are authored in mermaid under `sessionN/diagrams/*.mmd` and
pre-rendered to inline SVG, so nothing has to run at display time:

```bash
cd docs && python3 -m http.server 8731 &
cd task-source && node prerender-mermaid.mjs ../docs/session1
```

### Adding a live figure

Put a container on the slide:

```html
<figure class="fig" id="fig-example">
  <div class="fig-plot"></div>
  <div class="fig-controls"></div>
  <p class="fig-print-note"></p>
</figure>
```

and wire it up in that session's `figures.js`:

```js
interactiveFigure(
  'fig-example',
  { alpha: { min: 0.01, max: 1, step: 0.01, value: 0.2, label: '&alpha;' } },
  (el, v) => plotly().react(el, traces(v), layout({ width: 1080, height: 330 })),
  (v) => `Shown with a learning rate of ${v.alpha.toFixed(2)}.`,
);
```

Give every plot an explicit `width` and `height` in slide coordinates (the
canvas is 1280×720), and let `layout()` handle the margins: it sets
`automargin` on both axes, so an axis title can never end up sitting on its own
tick labels. Do not pass a `margin` of your own unless you have a reason, and
give axis titles as `{ text, standoff }` rather than a bare string, which
discards the standoff.

`task-source/layout-test.mjs` opens every deck, live and as exported, and fails
if anything reaches the footer or the slide edge, or an equation number is
drawn over its equation.

A slide carrying little besides a figure should give the figure the room —
around 1000-1120 wide is right for a full-width plot. Plots are drawn on load rather than when their slide comes
up, so a fixed size is what lets a figure on slide 20 render correctly while
slide 1 is showing — which is how the PDF export gets complete figures.

Figures use 2-D contour plots rather than 3-D surfaces: 3-D needs WebGL, which
is not guaranteed on a lecture-room machine and does not export cleanly.
