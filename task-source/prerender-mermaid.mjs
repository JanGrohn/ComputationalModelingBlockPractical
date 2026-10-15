/** Renders the mermaid sources in a deck to static SVG and writes them back. */
import puppeteer from 'puppeteer-core';
import { readFile, writeFile } from 'node:fs/promises';

const file = process.argv[2];
let html = await readFile(file, 'utf8');
const sources = [...html.matchAll(/<pre class="mermaid">([\s\S]*?)<\/pre>/g)];
if (!sources.length) { console.log('no mermaid blocks'); process.exit(0); }

const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new' });
const page = await browser.newPage();
await page.goto('http://localhost:8731/session1/', { waitUntil: 'networkidle0' });
await page.evaluate(() => document.fonts.ready);

for (const [whole, body] of sources) {
  const lines = body.replace(/\s+$/, '').split('\n').filter((l) => l.trim());
  const indent = Math.min(...lines.map((l) => l.match(/^ */)[0].length));
  const src = lines.map((l) => l.slice(indent)).join('\n');
  const svg = await page.evaluate(async (text) => {
    const mermaid = (await import('https://cdn.jsdelivr.net/npm/mermaid@11.3.0/dist/mermaid.esm.min.mjs')).default;
    mermaid.initialize({
      startOnLoad: false, theme: 'base', securityLevel: 'loose',
      flowchart: { curve: 'linear', nodeSpacing: 42, rankSpacing: 54, useMaxWidth: true, htmlLabels: false },
      themeVariables: { fontFamily: 'Inter, Helvetica, Arial, sans-serif', fontSize: '15px',
        primaryColor: '#f4f6f7', primaryTextColor: '#14171a', primaryBorderColor: '#14171a',
        lineColor: '#14171a', tertiaryColor: '#ffffff' },
    });
    const { svg } = await mermaid.render('diagram-' + Math.random().toString(36).slice(2), text);
    return svg;
  }, src);
  // mermaid's stylesheet always defines .error-icon, so look for the text the
  // error card actually renders
  if (/Syntax error in text|mermaid version/.test(svg)) {
    console.error('mermaid failed on:\n' + src); process.exit(1);
  }
  html = html.replace(whole,
    `<!-- Generated from the mermaid source below by prerender-mermaid.mjs.\n`
    + `     Edit the source, then re-run: node prerender-mermaid.mjs <file>\n`
    + src.split('\n').map((l) => '     ' + l).join('\n') + `\n-->\n`
    + `<div class="mermaid-figure">${svg}</div>`);
  console.log('rendered', src.split('\n')[0]);
}
await writeFile(file, html);
await browser.close();
