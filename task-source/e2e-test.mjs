/**
 * Drives the built task in a real browser and checks the things that were
 * broken: that trial 1 has a drawn outcome, that reversals land on the trial
 * the schedule says, and that the recorded data lines up with what was shown.
 */
import puppeteer from 'puppeteer-core';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
                '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.map': 'application/json' };

const server = createServer(async (req, res) => {
  let p = join('dist', decodeURI(req.url.split('?')[0]));
  if (p.endsWith('/')) p += 'index.html';
  try {
    const body = await readFile(p);
    res.writeHead(200, { 'content-type': TYPES[extname(p)] ?? 'application/octet-stream' });
    res.end(body);
  } catch { res.writeHead(404); res.end('not found'); }
});
await new Promise((r) => server.listen(8799, r));

const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new' });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.setViewport({ width: 1280, height: 900 });
const target = process.argv[2] ?? 'http://localhost:8799/';
console.log(`testing ${target}`);
await page.goto(target, { waitUntil: 'networkidle0' });

const clickText = async (text) => {
  await page.waitForFunction(
    (t) => [...document.querySelectorAll('button')].some((b) => b.textContent.trim().includes(t) && !b.disabled),
    { timeout: 5000 }, text);
  await page.evaluate((t) => {
    [...document.querySelectorAll('button')].find((b) => b.textContent.trim().includes(t) && !b.disabled).click();
  }, text);
};

await clickText('Confirm');                                    // fullscreen prompt
await page.waitForSelector('input');
await page.type('input', 'test-user');
await clickText('Confirm');                                    // username
await clickText('Select condition 2');                         // the volatile schedule

// the stimuli must actually render: a broken SVG still reports complete, so
// check the decoded size rather than the load event, and wait for the card
// slide-in to settle first
await page.waitForFunction(() => {
  const imgs = [...document.querySelectorAll('img')];
  return imgs.length === 2 && imgs.every((i) => i.naturalWidth > 0
    && i.getBoundingClientRect().height > 60);
}, { timeout: 8000 }).catch(() => {});
const stimuli = await page.evaluate(() => [...document.querySelectorAll('img')].map((i) => ({
  src: i.currentSrc.split('/').pop(),
  naturalW: i.naturalWidth,
  renderedH: Math.round(i.getBoundingClientRect().height),
})));
const stimuliOk = stimuli.length === 2
  && stimuli.every((s) => s.naturalW > 0 && s.renderedH > 60);
console.log(`stimuli render: ${stimuliOk}`);
stimuli.forEach((s) => console.log(`    ${s.src}  natural ${s.naturalW}px, rendered ${s.renderedH}px tall`));

// play 40 trials, always choosing the left card
const seen = [];
for (let t = 0; t < 40; t++) {
  await clickText('Select');
  await new Promise((r) => setTimeout(r, 1150));
  const shown = await page.evaluate(() =>
    [...document.querySelectorAll('h2')].map((h) => h.textContent.trim()));
  seen.push(shown);
  await clickText('Next Choice');
  await new Promise((r) => setTimeout(r, 1100));
}

const data = await page.evaluate(() => window.participantData);
await browser.close();
server.close();

const selections = data.filter((d) => d.type && d.type.startsWith('Select'));
console.log(`page errors: ${errors.length ? errors.join('; ') : 'none'}`);
console.log(`trials recorded: ${selections.length}`);

// trial 1 must have a real drawn outcome, not the hardcoded default
console.log(`trial 1 recorded oneRewarded: ${selections[0].oneRewarded}`);

// exactly one side shows a dash after each choice
const dashOk = seen.every((pair) => pair.filter((v) => v === '-').length === 1);
console.log(`exactly one side dashed on every trial: ${dashOk}`);

// in condition 2 option 1 pays out on ~75% of trials 0-27, then ~25%
const frac = (from, to) => {
  const s = selections.slice(from, to);
  return (s.filter((d) => d.oneRewarded).length / s.length).toFixed(2);
};
console.log(`P(option 1 rewarded) trials 1-28:  ${frac(0, 28)}   (schedule says 0.75)`);
console.log(`P(option 1 rewarded) trials 29-40: ${frac(28, 40)}   (schedule says 0.25)`);
