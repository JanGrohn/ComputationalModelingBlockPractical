# The volatility task

The learning task students play in session 1. It used to live in the private
`jangrohn.github.io` repository and be served from
`jangrohn.github.io/volatility_study`. It is now hosted from this repository at

<https://jangrohn.github.io/ComputationalModelingBlockPractical/task/>

## Where the source came from

The old repository only ever held the compiled output — no source tree. The
build did ship source maps with `sourcesContent`, so the original React source
was recovered from `static/js/main.e52e1237.js.map`. What is in `src/` is that
source, with the fixes below applied and the build moved from create-react-app
to Vite.

## Layout

```
task-source/            this directory: the source, not served
  index.html            Vite entry point
  vite.config.js        base: "./" so the build works at any URL
  schedule-test.mjs     statistical checks on the trial schedule
  e2e-test.mjs          drives the built app in a real browser
  src/
    main.jsx
    App.jsx             fullscreen prompt -> username -> condition -> task
    schedule.js         the trial schedule, free of React so it can be tested
    components/
      Choice.jsx        the task itself: trials, outcomes, instructions
      EnterData.jsx     username entry
      TwoButtons.jsx    condition 1 (stable) or condition 2 (volatile)
      EndStudy.jsx      comments box and the results download
      img/              the two stimuli, as SVG
../docs/task/           the compiled app that is served
```

## Working on it

```bash
cd task-source
npm install
npm run dev            # live reload at localhost:5173

npm run build          # writes dist/
node schedule-test.mjs # schedule properties over 4000 runs
node e2e-test.mjs      # 40 trials in a real browser

rm -rf ../docs/task && cp -R dist ../docs/task
```

`e2e-test.mjs` drives the Google Chrome already installed on the machine
through puppeteer-core; nothing is downloaded.

## The two conditions

Condition 1 is the stable schedule and condition 2 the volatile one. The
probability array holds P(option 2 rewarded), so the value 0.25 means option 1
pays out on three trials in four. Condition 1 holds that for all 102 trials;
condition 2 runs 28 trials at 0.25 and then reverses every 25.

The first two trials are fixed at magnitudes 66/45 and 31/60 with the deer on
the left, so the instruction screens always describe the same thing.

## What was fixed

Found by reading the recovered source, and each one covered by a test.

- **Each trial's outcome was drawn from the previous trial's probability.**
  `handleConfirmClick` advanced the index and drew the next outcome in the same
  handler, reading `prob[currentIndex]` before the increment, so every reversal
  landed one trial late. The whole schedule is now drawn once up front, in
  `schedule.js`, which removes the class of bug rather than the instance.
- **The first trial's outcome was never drawn.** `oneRewarded` started as
  `true`, so trial 1 always rewarded option 1.
- **An unknown condition silently produced `undefined`.** `prob` was declared
  with `var` inside two branches; it now raises a clear error.
- **Left and right were not balanced over the trials actually run.** The side
  order was shuffled over 200 entries of which only 102 were used. It is now
  shuffled over exactly the trials that run, so every run is 50/50.
- **The first two trials carried their magnitudes as strings** while the rest
  were numbers, so any arithmetic on them would have concatenated.
- **Prolific is gone**, along with the dead completion link whose `url` prop
  was shadowed by `URL.createObjectURL` and never used.
- **The expectation slider is gone.** It was unreachable anyway, since nothing
  ever set `showSlider`.
- A stray `displaySide` attribute on an `<img>`, missing `alt` text on both
  stimuli, an empty 60-second `setInterval`, a commented-out dialog and a
  number of unused imports have all gone.

Magnitudes stay at 1–99 and the two conditions stay as they are, both as
intended.

## Why Vite rather than create-react-app

`react-scripts` 5.0.1 is unmaintained and does not build reliably on current
Node. Vite installs 132 packages in about 20 seconds and builds in just over a
second, against several minutes for CRA, and the bundle came out 80 KB smaller
once `react-helmet` and `web-vitals` went. `react-helmet` existed only to
inject a preloadjs script that nothing used.
