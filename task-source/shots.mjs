/**
 * Screenshots individual slides, for checking a deck without opening it.
 *
 *   cd docs && python3 -m http.server 8731 &
 *   node shots.mjs http://localhost:8731/session2/index.html /tmp/out s2 15 27
 *
 * Writes <out>/<tag>-NN.png per slide and reports any console errors the page
 * threw. Append ?export to the URL to see what the PDF will contain.
 */

import puppeteer from 'puppeteer-core';
const [base, out, tag, ...slides] = process.argv.slice(2);
const browser = await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless:'new'});
const page = await browser.newPage(); await page.setViewport({width:1280,height:720});
const errs=[]; page.on('pageerror',e=>errs.push(String(e).slice(0,160)));
for (const n of slides) {
  await page.goto(`${base}#/${n}`, {waitUntil:'networkidle0'});
  await new Promise(r=>setTimeout(r,2400));
  await page.screenshot({path:`${out}/${tag}-${String(n).padStart(2,'0')}.png`});
}
if (errs.length) console.log('PAGE ERRORS:', [...new Set(errs)].join(' | '));
await browser.close();
