/** Checks the statistical properties of the trial schedule over many runs. */
import { buildSchedule, rewardProbabilities, N_TRIALS } from './src/schedule.js';

let failures = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '   ' + detail : ''}`);
  if (!ok) failures++;
};

const RUNS = 4000;
const runs = Array.from({ length: RUNS }, () => buildSchedule(2));
const stable = Array.from({ length: RUNS }, () => buildSchedule(1));

// every array covers exactly the trials that are run
check('schedule arrays are all N_TRIALS long',
  runs.every((r) => r.reward1.length === N_TRIALS && r.reward2.length === N_TRIALS
                 && r.displaySide.length === N_TRIALS && r.oneRewarded.length === N_TRIALS),
  `N_TRIALS=${N_TRIALS}`);

// trial 1's outcome is drawn, not hardcoded
const firstTrue = runs.filter((r) => r.oneRewarded[0]).length / RUNS;
check('trial 1 outcome is actually drawn', firstTrue > 0.70 && firstTrue < 0.80,
  `P(option 1 rewarded on trial 1) = ${firstTrue.toFixed(3)}, expected 0.75`);

// the reversal is on the trial the schedule says, not one late
const at = (i) => runs.filter((r) => r.oneRewarded[i]).length / RUNS;
const probs = rewardProbabilities(2);
for (const i of [26, 27, 28, 29, 52, 53]) {
  const want = 1 - probs[i];
  check(`trial ${i + 1} matches its own probability`, Math.abs(at(i) - want) < 0.035,
    `observed ${at(i).toFixed(3)}, schedule ${want.toFixed(2)}`);
}

// the stable condition never moves
const stableRates = [0, 30, 60, 101].map((i) => stable.filter((r) => r.oneRewarded[i]).length / RUNS);
check('condition 1 is stable throughout', stableRates.every((p) => Math.abs(p - 0.75) < 0.035),
  stableRates.map((p) => p.toFixed(3)).join(', '));

// left/right is balanced over the trials actually run
const sideMeans = runs.map((r) => r.displaySide.slice(2).reduce((a, b) => a + b, 0) / (N_TRIALS - 2));
check('left/right is balanced across the run', sideMeans.every((m) => m === 0.5),
  `every run is exactly 50/50`);

// magnitudes stay in range, and the fixed opening trials are numbers
const mags = runs.flatMap((r) => [...r.reward1.slice(2), ...r.reward2.slice(2)]);
const lo = mags.reduce((a, b) => Math.min(a, b), Infinity);
const hi = mags.reduce((a, b) => Math.max(a, b), -Infinity);
check('magnitudes are integers in 1..99',
  mags.every((m) => Number.isInteger(m) && m >= 1 && m <= 99),
  `min ${lo}, max ${hi}`);
check('the two fixed opening trials are numbers, not strings',
  runs.every((r) => [r.reward1[0], r.reward1[1], r.reward2[0], r.reward2[1]].every(Number.isInteger)),
  'reward1 31,60 / reward2 66,45');

// option 1 is the green stimulus, as in the notebooks: trial 1 shows green on
// the left worth 31, orange on the right worth 66
check('trial 1 matches the task screenshot and the slides',
  runs.every((r) => r.reward1[0] === 31 && r.reward2[0] === 66 && r.displaySide[0] === 1),
  'green option 1 on the left worth 31, orange option 2 on the right worth 66');

// an unknown condition fails loudly rather than reading undefined
let threw = false;
try { buildSchedule(3); } catch { threw = true; }
check('an unknown condition raises a clear error', threw);

console.log(`\n${failures ? failures + ' FAILURES' : 'all checks passed'}`);
process.exit(failures ? 1 : 0);
