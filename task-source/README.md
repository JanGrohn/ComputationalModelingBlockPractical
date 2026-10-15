# The volatility task

The learning task students play in session 1. It used to live in the private
`jangrohn.github.io` repository and be served from
`jangrohn.github.io/volatility_study`. It is now hosted from this repository at

<https://jangrohn.github.io/ComputationalModelingBlockPractical/task/>

## Where the source came from

That repository only ever held the compiled output — no source tree. The build
did ship source maps with `sourcesContent`, so the original React source was
recovered from `static/js/main.e52e1237.js.map` and is what you see in `src/`.
It is the real source, not a decompilation, but it has never been built from
this directory, so treat the first `npm install && npm run build` as unproven.

`package.json` is reconstructed from the package list in the same source map.
The versions are the latest of each major that the bundle is consistent with,
not necessarily the exact ones originally used.

## Layout

```
task-source/           this directory: the source, not served
  src/
    App.js             fullscreen prompt -> username -> condition -> task
    components/
      Choice.js        the task itself: trials, outcomes, instructions
      EnterData.js     username entry
      TwoButtons.js    condition 1 or condition 2
      EndStudy.js      comments box and the results download
      img/             the two stimuli, as SVG
../docs/task/          the compiled app that is actually served
```

## Rebuilding

```bash
cd task-source
npm install
npm run build
rm -rf ../docs/task && cp -R build ../docs/task
```

`homepage` is set to `.` so the build uses relative paths and works at whatever
URL it is served from. The copy currently in `docs/task/` predates this setting
and had its absolute `/volatility_study/` paths rewritten to relative ones by
hand; a fresh build will produce them correctly.

`react-scripts` 5.0.1 is unmaintained and may need an older Node than the one
you have. If it refuses to build, migrating to Vite is a half-hour job for an
app this small and removes the problem permanently.

## Known problems in the current build

Found by reading the recovered source. None of them are fixed in what is
currently served, because fixing them needs a rebuild, and the first two are
questions about what the task is meant to do rather than clear defects.

**1. Condition 1 has no volatile block.** `Choice.js` builds `prob1` as 103
trials at a constant 0.25, and `prob2` as 28 trials at 0.25 then alternating
blocks of 25. So condition 1 is stable throughout and condition 2 is volatile
after trial 28. The course teaches a within-participant stable *and* volatile
design, counterbalanced, and the Blain data has 80 trials of each. Whether the
two conditions are meant to be the two block orders, and are simply missing
their second half, is a question for whoever designed it.

**2. The outcome of each trial is drawn from the previous trial's
probability.** In `handleConfirmClick` the index is advanced and the next
outcome drawn in the same handler, using `prob[currentIndex]` — the index
before the increment. Every reversal therefore lands one trial late. It makes
no difference in condition 1, where the probability never changes.

**3. The first trial's outcome is never drawn at all.** `oneRewarded` is
initialised to `true` and only changes on the first "next choice", so trial 1
always rewards option 1.

**4. `prob` is undefined for any condition other than 1 or 2**, because it is
declared with `var` inside the two branches. Nothing currently passes anything
else, so it cannot fire today.

**5. The Prolific completion link never works.** `EndStudy` takes a `url`
prop, but `handleDownload` declares `const url = URL.createObjectURL(blob)`,
which shadows it, and the prop is never used. Harmless here — the course does
not use Prolific — but the link in `Choice.js` is dead.

**6. The expectation slider is unreachable.** `showSlider` is initialised to
`false` and nothing ever sets it to `true`, so the "how many points do you
expect?" rating never appears. Blain & Rutledge, whose data the course
analyses, do collect ratings, so this may have been switched off deliberately.

**7. Magnitudes run 1–99, not 1–100.** `Math.floor(Math.random() * 99) + 1`.
The notebooks and slides say 1–100.

**8. The first two trials carry string magnitudes** (`"66"`, `"45"`, `"31"`,
`"60"` are unshifted as strings while the rest are numbers). Only display is
affected now, but any arithmetic on them would concatenate.

Smaller things: `displaySide` is shuffled over 200 entries but only the first
102 are used, so left/right is not balanced across the trials actually run; a
stray `displaySide` attribute is passed to an `<img>`; neither stimulus has an
`alt`; a 60-second `setInterval` runs an empty function; and several imports
are unused.
