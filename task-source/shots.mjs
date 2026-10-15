import puppeteer from 'puppeteer-core';
const [base, out, ...slides] = process.argv.slice(2);
const browser = await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless:'new'});
const page = await browser.newPage(); await page.setViewport({width:1280,height:720});
const errs=[]; page.on('pageerror',e=>errs.push(String(e).slice(0,120)));
for (const n of slides) {
  await page.goto(`${base}#/${n}`, {waitUntil:'networkidle0'});
  await new Promise(r=>setTimeout(r,2200));
  await page.screenshot({path:`${out}/s-${String(n).padStart(2,'0')}.png`});
}
if (errs.length) console.log('PAGE ERRORS:', [...new Set(errs)].join(' | '));
await browser.close();
