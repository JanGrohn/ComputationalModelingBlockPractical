/**
 * The trial schedule, with no React or DOM in sight so that its statistical
 * properties can be tested directly.
 *
 * Everything is drawn once, before the first trial is shown. Drawing a trial
 * at a time is what made each outcome depend on the previous trial's
 * probability, putting every reversal one trial late, and left the very first
 * trial with no draw at all.
 */

export const INSTRUCTION_TRIALS = 2;
export const MAIN_TRIALS = 100;
export const N_TRIALS = INSTRUCTION_TRIALS + MAIN_TRIALS;

export function shuffle(array, random = Math.random) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

/**
 * Option 1 is the green stimulus and option 2 the orange one, matching the
 * notebooks and the slides: reward1 and oneRewarded below both refer to the
 * green option.
 *
 * This returns the probability that option 2 is rewarded on each trial. Option
 * 1 is rewarded whenever option 2 is not, so 0.25 here means the green option
 * pays out on three trials in four.
 *
 * Condition 1 is the stable schedule: the contingency never moves.
 * Condition 2 is the volatile one: it reverses every 25 trials.
 */
export function rewardProbabilities(condition) {
  if (condition === 1) {
    return new Array(N_TRIALS).fill(0.25);
  }
  if (condition === 2) {
    return [
      ...new Array(28).fill(0.25),
      ...new Array(25).fill(0.75),
      ...new Array(25).fill(0.25),
      ...new Array(25).fill(0.75),
    ].slice(0, N_TRIALS);
  }
  throw new Error(`Unknown condition ${condition}: expected 1 (stable) or 2 (volatile).`);
}

export const randomMagnitude = (random = Math.random) => Math.floor(random() * 99) + 1;

export function buildSchedule(condition, random = Math.random) {
  const probabilities = rewardProbabilities(condition);

  // the first two trials are fixed, so the instructions always describe the
  // same thing: the green option worth 31 on the left, the orange one worth 66
  // on the right
  const reward1 = [31, 60];
  const reward2 = [66, 45];
  for (let i = 0; i < MAIN_TRIALS; i++) {
    reward1.push(randomMagnitude(random));
    reward2.push(randomMagnitude(random));
  }

  // 1 when option 1 (green) is drawn on the left, balanced across the trials
  // actually run rather than across a longer list that then gets truncated
  const sides = new Array(MAIN_TRIALS / 2).fill(0).concat(new Array(MAIN_TRIALS / 2).fill(1));
  const displaySide = [1, 0, ...shuffle(sides, random)];

  // true when option 1 (green) is the rewarded one on that trial
  const oneRewarded = probabilities.map((p) => random() > p);

  return { probabilities, reward1, reward2, displaySide, oneRewarded };
}
