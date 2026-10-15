/**
 * Renders each .mmd file in a deck's diagrams/ folder to a static SVG and
 * drops it into the matching <div class="mermaid-figure" data-diagram="name">.
 *
 * The source lives in its own file rather than in an HTML comment beside the
 * diagram: mermaid's arrow syntax contains "-->", which closes an HTML comment
 * early and spills the rest of the source onto the slide.
 *
 *   node prerender-mermaid.mjs ../docs/session1
 */
import puppeteer from 'puppeteer-core';
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { join, basename } from 'node:path';

const deck = process.argv[2];
const htmlPath = join(deck, 'index.html');
let html = await readFile(htmlPath, 'utf8');

const files = (await readdir(join(deck, 'diagrams'))).filter((f) => f.endsWith('.mmd'));
if (!files.length) { console.log('no .mmd files'); process.exit(0); }

const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new' });
const page = await browser.newPage();
await page.goto('http://localhost:8731/session1/', { waitUntil: 'networkidle0' });
await page.evaluate(() => document.fonts.ready);

for (const file of files) {
  const name = basename(file, '.mmd');
  const src = (await readFile(join(deck, 'diagrams', file), 'utf8')).trim();
  const svg = await page.evaluate(async (text) => {
    const mermaid = (await import('https://cdn.jsdelivr.net/npm/mermaid@11.3.0/dist/mermaid.esm.min.mjs')).default;
    mermaid.initialize({
      startOnLoad: false, theme: 'base', securityLevel: 'loose',
      // htmlLabels off: with them on, mermaid measures the labels in one font
      // and draws them in another, clipping the last character off every node
      flowchart: { curve: 'linear', nodeSpacing: 42, rankSpacing: 56, useMaxWidth: true, htmlLabels: false },
      themeVariables: { fontFamily: 'Inter, Helvetica, Arial, sans-serif', fontSize: '16px',
        primaryColor: '#f4f6f7', primaryTextColor: '#14171a', primaryBorderColor: '#14171a',
        lineColor: '#14171a', tertiaryColor: '#ffffff' },
    });
    const { svg } = await mermaid.render('d' + Math.random().toString(36).slice(2), text);
    return svg;
  }, src);

  // mermaid's stylesheet always defines .error-icon, so look for the text its
  // error card actually renders
  if (/Syntax error in text|mermaid version/.test(svg)) {
    console.error(`mermaid failed on ${file}:\n${src}`); process.exit(1);
  }

  const re = new RegExp(`(<div class="mermaid-figure" data-diagram="${name}">)[\\s\\S]*?(</div>)`);
  if (!re.test(html)) { console.error(`no placeholder for "${name}" in ${htmlPath}`); process.exit(1); }
  html = html.replace(re, `$1${svg}$2`);
  console.log(`rendered ${file}`);
}

await writeFile(htmlPath, html);
await browser.close();
