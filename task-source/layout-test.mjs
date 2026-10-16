/**
 * Checks that nothing on any slide runs off the slide or into its footer, that
 * no label inside a figure is clipped by the figure's own edge, that
 * no equation number is drawn over its equation, and that no diagram is shrunk
 * until its labels are too small to read.
 *
 *   node layout-test.mjs
 *
 * Each deck is checked twice: as presented, with every fragment shown and the
 * figure controls visible, and as exported to PDF, where the controls are
 * replaced by a caption. A slide's own box cannot be trusted for this, since
 * reveal's sections grow with their content, so every block is measured
 * against the 1280x720 canvas instead.
 */

import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import puppeteer from 'puppeteer-core';

const ROOT = new URL('../docs/', import.meta.url).pathname;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
               '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg' };
const server = createServer((req, res) => {
  const path = normalize(join(ROOT, decodeURIComponent(req.url.split('?')[0])));
  const file = path.endsWith('/') ? join(path, 'index.html') : path;
  let body;
  try { body = readFileSync(file); } catch { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': MIME[extname(file)] ?? 'application/octet-stream' });
  res.end(body);
}).listen(0);
const port = server.address().port;

// the blocks a slide is built from; anything inside them moves with them
const BLOCKS = '.fig-plot, .fig-controls, .fig-print-note, .mermaid-figure > svg, figure img, '
  + '.discuss, .takeaway, .eq, .panel, .note, .lead, ul, ol, .weeks, .task-board, .citation, p';

const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new',
});
const failures = [];
for (const n of [1, 2, 3, 4]) {
  for (const mode of ['', '?export']) {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 720 });
    await page.goto(`http://localhost:${port}/session${n}/index.html${mode}`, { waitUntil: 'networkidle0' });
    await new Promise((r) => setTimeout(r, 3500));
    const bad = await page.evaluate(async (BLOCKS) => {
      const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const out = [];
      const sections = [...document.querySelectorAll('.reveal .slides > section')];
      // Each slide is measured while it is the one being shown: a hidden slide
      // forced into view is not laid out the way reveal lays out the present
      // one, and measures differently.
      for (let i = 0; i < sections.length; i++) {
        window.Reveal.slide(i);
        await frame();
        const s = sections[i];
        s.querySelectorAll('.fragment').forEach((f) => f.classList.add('visible'));
        await frame();
        const canvas = document.querySelector('.reveal .slides').getBoundingClientRect();
        const scale = canvas.height / 720;
        const title = ((s.querySelector('h1,h2') || {}).textContent || '(title)').trim().slice(0, 50);
        const top = (e) => (e.getBoundingClientRect().top - canvas.top) / scale;
        // the content may run down to the footer, or to the citation pinned
        // just above it where a slide has one
        const footer = s.querySelector('.deck-footer');
        const citation = s.querySelector('.citation');
        const footerTop = footer ? top(footer.firstElementChild || footer) : 720;
        s.querySelectorAll(BLOCKS).forEach((e) => {
          if (e.closest('.deck-footer') || e.closest('.title-slide')) return;
          const r = e.getBoundingClientRect();
          if (!r.width || !r.height || getComputedStyle(e).display === 'none') return;
          const limit = citation && !e.closest('.citation') ? top(citation) : footerTop;
          const bottom = (r.bottom - canvas.top) / scale;
          const right = (r.right - canvas.left) / scale;
          if (bottom > limit + 1 || right > 1281) {
            out.push(`slide ${i + 1} "${title}": ${e.tagName.toLowerCase()}.${(e.className.baseVal ?? e.className).split(' ')[0]}`
              + (bottom > limit + 1 ? ` reaches y=${Math.round(bottom)}, limit ${Math.round(limit)}` : '')
              + (right > 1281 ? ` reaches x=${Math.round(right)}` : ''));
          }
        });
        // Plotly draws a figure inside an SVG of fixed size and clips anything
        // outside it, so an axis title longer than its plot is tall, or a label
        // placed too far out, is cut off without any error
        s.querySelectorAll('.js-plotly-plot').forEach((plot) => {
          const box = plot.querySelector('.main-svg').getBoundingClientRect();
          const id = plot.closest('figure')?.id || '?';
          plot.querySelectorAll('.g-xtitle text, .g-ytitle text, .g-x2title text, .g-y2title text, '
              + '.g-y3title text, .xtick text, .ytick text, .annotation-text, .cbtitle text, '
              + '.colorbar text').forEach((t) => {
            const r = t.getBoundingClientRect();
            if (!r.width || !(t.textContent || '').trim()) return;
            const over = Math.max(box.left - r.left, r.right - box.right, box.top - r.top,
                                  r.bottom - box.bottom) / scale;
            if (over > 1.5) {
              out.push(`slide ${i + 1} "${title}": ${id} clips "${t.textContent.trim().slice(0, 30)}" `
                + `by ${Math.round(over)}px`);
            }
          });
        });
        // an axis title must sit clear of its axis's whole row of tick labels:
        // a title in the gap between two ticks is still on top of the axis
        s.querySelectorAll('.js-plotly-plot').forEach((plot) => {
          const id = plot.closest('figure')?.id || '?';
          const layout = plot._fullLayout || {};
          Object.keys(layout).filter((k) => /^[xy]axis\d*$/.test(k)).forEach((name) => {
            const ax = layout[name];
            const letter = name[0];
            const titleText = ax.title?.text;
            if (!titleText) return;
            const sub = ax._id;   // x, x2, y3, ...
            const titleEl = plot.querySelector(`.g-${sub}title text`);
            const tickEls = [...plot.querySelectorAll(`.${sub}tick text`)];
            if (!titleEl || !tickEls.length) return;
            const t = titleEl.getBoundingClientRect();
            const k = tickEls.map((e) => e.getBoundingClientRect()).filter((r) => r.width);
            if (!t.width || !k.length) return;
            const clash = letter === 'x'
              ? t.top < Math.max(...k.map((r) => r.bottom)) - 1
              : t.right > Math.min(...k.map((r) => r.left)) + 1;
            if (clash) out.push(`slide ${i + 1} "${title}": ${id} draws "${titleEl.textContent.trim()}" `
              + 'on its row of tick labels');
          });
        });
        // tick labels on one axis that run into each other, which happens when
        // an axis is given more ticks than its length has room for
        s.querySelectorAll('.js-plotly-plot').forEach((plot) => {
          const id = plot.closest('figure')?.id || '?';
          plot.querySelectorAll('.xaxislayer-above, .yaxislayer-above').forEach((layer) => {
            const vertical = layer.classList.contains('yaxislayer-above');
            const boxes = [...layer.querySelectorAll('.xtick text, .ytick text')]
              .map((t) => t.getBoundingClientRect()).filter((r) => r.width)
              .sort((a, b) => (vertical ? a.top - b.top : a.left - b.left));
            for (let k = 1; k < boxes.length; k++) {
              const gap = vertical ? boxes[k].top - boxes[k - 1].bottom : boxes[k].left - boxes[k - 1].right;
              if (gap / scale < -1) {
                out.push(`slide ${i + 1} "${title}": ${id} has ${vertical ? 'y' : 'x'} tick labels `
                  + 'running into each other');
                break;
              }
            }
          });
        });
        // a diagram is drawn at 16px and scaled to fit its box; shrunk much
        // further than that, its labels stop being readable from the back
        s.querySelectorAll('.mermaid-figure > svg').forEach((svg) => {
          const vb = svg.viewBox.baseVal;
          if (!vb || !vb.width) return;
          const px = 16 * (svg.getBoundingClientRect().width / scale) / vb.width;
          if (px < 14.5) {
            out.push(`slide ${i + 1} "${title}": diagram "${svg.parentElement.dataset.diagram}" `
              + `labels drawn at ${px.toFixed(1)}px`);
          }
        });
        // KaTeX pins an equation's \\tag to the right edge of its box, so in a
        // box too narrow for both, the number is drawn over the equation
        s.querySelectorAll('.katex-display').forEach((d) => {
          const tag = d.querySelector('.tag');
          const bases = [...d.querySelectorAll('.katex-html > .base')].map((b) => b.getBoundingClientRect());
          if (!tag || !bases.length) return;
          if (Math.max(...bases.map((b) => b.right)) > tag.getBoundingClientRect().left - 4) {
            out.push(`slide ${i + 1} "${title}": an equation number overlaps its equation `
              + `(${(tag.textContent || '').trim()})`);
          }
        });
      }
      return [...new Set(out)];
    }, BLOCKS);
    const label = `session ${n} ${mode ? 'export' : 'live'}`;
    bad.forEach((b) => failures.push(`${label}: ${b}`));
    console.log(`  ${label}: ${bad.length ? bad.length + ' overflowing' : 'fits'}`);
    await page.close();
  }
}
await browser.close();
server.close();

console.log();
if (failures.length) {
  console.log('FAIL');
  failures.forEach((f) => console.log('  ' + f));
  process.exit(1);
}
console.log('PASS  every block on every slide stays clear of the footer and the edges');
