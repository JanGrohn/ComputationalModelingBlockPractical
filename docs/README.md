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
  index.html          landing page listing the four sessions
  assets/
    theme.css         the whole design system; edit colours here
    models.js         the RL model, utilities and softmax, in JavaScript
    deck.js           reveal setup, KaTeX, and the interactive-figure helper
    img/              figures taken from the original decks
  sessionN/
    index.html        the slides
    figures.js        that session's live figures
  export-pdf.sh
```

`assets/models.js` is a deliberate line-for-line port of the Python in
`sessionN/fitting.py`. If you change a model in the notebooks, change it there
too, or a slide and the practical will quietly disagree.

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

Colour means one thing only: `--opt1` green is option 1, `--opt2` orange is
option 2, matching the task and the notebook plots. Nothing is distinguished by
colour alone, so the greyscale print still works. Type is one sans family
throughout, separated by weight rather than by switching to a serif; maths is
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

A slide carrying little besides a figure should give the figure the room —
around 1000-1120 wide is right for a full-width plot. Plots are drawn on load rather than when their slide comes
up, so a fixed size is what lets a figure on slide 20 render correctly while
slide 1 is showing — which is how the PDF export gets complete figures.

Figures use 2-D contour plots rather than 3-D surfaces: 3-D needs WebGL, which
is not guaranteed on a lecture-room machine and does not export cleanly.
