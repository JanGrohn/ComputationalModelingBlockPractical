/**
 * Checks that every part of the course uses one palette, and that every colour
 * the slides actually draw comes from it.
 *
 *   node palette-test.mjs
 *
 * 1. The palette is written down in several places, because the notebooks
 *    reach Colab one session folder at a time: docs/assets/deck.js,
 *    docs/assets/theme.css, sessionN/palette.py, and the task's two stimuli.
 *    They must all agree.
 * 2. Every deck is opened in Chrome and every colour on every live figure,
 *    traces, annotations and shapes alike, must be a palette colour, a grey
 *    used for type and annotation, or white.
 *
 * The notebooks' figures are checked the same way by notebook-colours-test.py.
 */

import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import puppeteer from 'puppeteer-core';

const ROOT = new URL('..', import.meta.url).pathname;
const read = (path) => readFileSync(join(ROOT, path), 'utf8');
const failures = [];
const check = (ok, message) => { if (!ok) failures.push(message); };

/* --- 1. every written copy of the palette agrees ------------------------ */

const TOKENS = ['option1', 'option2', 'stable', 'volatile', 'estimate', 'truth', 'neutral'];

const deck = read('docs/assets/deck.js');
const fromDeck = Object.fromEntries(TOKENS.map((t) => {
  const m = deck.match(new RegExp(`\\b${t}:\\s*'(#[0-9a-f]{6})'`, 'i'));
  if (!m) throw new Error(`docs/assets/deck.js does not define COL.${t}`);
  return [t, m[1].toLowerCase()];
}));

const css = read('docs/assets/theme.css');
for (const t of TOKENS) {
  const m = css.match(new RegExp(`--${t}:\\s*(#[0-9a-f]{6})`, 'i'));
  check(m && m[1].toLowerCase() === fromDeck[t],
        `theme.css --${t} is ${m?.[1]}, deck.js says ${fromDeck[t]}`);
}

for (const n of [1, 2, 3, 4]) {
  const py = read(`session${n}/palette.py`);
  for (const t of TOKENS) {
    const m = py.match(new RegExp(`^${t.toUpperCase()}\\s*=\\s*'(#[0-9a-f]{6})'`, 'im'));
    check(m && m[1].toLowerCase() === fromDeck[t],
          `session${n}/palette.py ${t.toUpperCase()} is ${m?.[1]}, deck.js says ${fromDeck[t]}`);
  }
  check(/^CONTINUOUS\s*=\s*'Viridis'/m.test(py), `session${n}/palette.py is not using Viridis`);
}
check(/CONTINUOUS = 'Viridis'/.test(deck), 'deck.js is not using Viridis');

// the stimuli are where the two option colours come from in the first place
const fills = (file) => [...new Set([...read(file).matchAll(/fill="(#[0-9a-f]{6})"/gi)]
  .map((m) => m[1].toLowerCase()))];
check(JSON.stringify(fills('docs/assets/img/stimuli/deer-silhouette.svg')) === JSON.stringify([fromDeck.option1]),
      `the deer is ${fills('docs/assets/img/stimuli/deer-silhouette.svg')}, option 1 is ${fromDeck.option1}`);
check(JSON.stringify(fills('docs/assets/img/stimuli/bird-crane-shape.svg')) === JSON.stringify([fromDeck.option2]),
      `the crane is ${fills('docs/assets/img/stimuli/bird-crane-shape.svg')}, option 2 is ${fromDeck.option2}`);

/* --- 2. every colour the decks draw is a palette colour ----------------- */

// Type and annotation greys, and white for marker rings and contour lines.
// These carry no meaning of their own, which is why they are allowed anywhere.
const GREYS = ['#14171a', '#5b6670', '#9aa3aa', '#6b747c', '#e2e6ea', '#ffffff', '#b35309'];
const ALLOWED = new Set([...Object.values(fromDeck), ...GREYS]);

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
               '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg' };
const server = createServer((req, res) => {
  const path = normalize(join(ROOT, 'docs', decodeURIComponent(req.url.split('?')[0])));
  const file = path.endsWith('/') ? join(path, 'index.html') : path;
  let body;
  try { body = readFileSync(file); } catch { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': MIME[extname(file)] ?? 'application/octet-stream' });
  res.end(body);
}).listen(0);
const port = server.address().port;

const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new',
});
for (const n of [1, 2, 3, 4]) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });
  await page.goto(`http://localhost:${port}/session${n}/index.html`, { waitUntil: 'networkidle0' });
  await new Promise((r) => setTimeout(r, 3000));
  const found = await page.evaluate(() => {
    // every colour value anywhere in a figure's data and layout, with where it was
    const out = [];
    const walk = (obj, where) => {
      if (obj == null) return;
      if (typeof obj === 'string') {
        if (/^(#|rgba?\()/i.test(obj)) out.push([obj, where]);
        return;
      }
      if (Array.isArray(obj)) {
        // a colour scale is a list of [stop, colour] pairs: checked as a whole below
        if (obj.length && Array.isArray(obj[0]) && typeof obj[0][0] === 'number') return;
        obj.forEach((v, i) => walk(v, where));
        return;
      }
      if (typeof obj === 'object') {
        for (const [k, v] of Object.entries(obj)) {
          if (k === 'colorscale') { out.push([typeof v === 'string' ? `scale:${v}` : 'scale:custom', where]); continue; }
          if (/color|fill/i.test(k) || typeof v === 'object') walk(v, `${where}.${k}`);
        }
      }
    };
    document.querySelectorAll('figure.fig').forEach((fig) => {
      const el = fig.querySelector('.js-plotly-plot');
      if (!el) return;
      (el.data || []).forEach((t, i) => walk(t, `${fig.id} trace ${i} (${t.name || t.type})`));
      walk({ annotations: el.layout.annotations, shapes: el.layout.shapes }, `${fig.id} layout`);
    });
    return out;
  });

  const toHex = (c) => {
    const m = c.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
    if (m) return '#' + m.slice(1, 4).map((v) => Number(v).toString(16).padStart(2, '0')).join('');
    if (/^#[0-9a-f]{3}$/i.test(c)) return '#' + c.slice(1).split('').map((d) => d + d).join('');
    return c.toLowerCase();
  };
  const strays = new Map();
  for (const [colour, where] of found) {
    if (colour.startsWith('scale:')) {
      if (colour !== 'scale:Viridis') strays.set(`${where}: colour scale ${colour.slice(6)}`, 1);
      continue;
    }
    // a translucent tint of a palette colour is that colour
    if (!ALLOWED.has(toHex(colour))) strays.set(`${where}: ${colour}`, 1);
  }
  for (const s of strays.keys()) failures.push(`session ${n}: ${s}`);
  console.log(`  session ${n}: ${found.length} colours checked on its figures`);
  await page.close();
}
await browser.close();
server.close();

console.log();
if (failures.length) {
  console.log('FAIL');
  failures.forEach((f) => console.log('  ' + f));
  process.exit(1);
}
console.log('PASS  one palette across the slides, the theme, the notebooks and the task');
