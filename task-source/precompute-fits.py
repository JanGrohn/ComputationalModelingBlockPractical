"""
Computes everything sessions 3 and 4 plot about the Blain and Rutledge data,
once, and writes it to docs/assets/participant-fits.js.

    cd task-source && python3 precompute-fits.py

Those two decks show fits to real participants, two simulations run against the
participants' own schedules, and a model recovery over 100 simulated datasets.
None of that can be redone in a browser, so it is done here with the practical's
own code -- session3.fitting and session4.fitting, the same functions the
notebooks call -- and the results are written out as a JavaScript module.

Re-run this if the model, the fitting routine or the saved fits change.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_1samp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# session3.loading looks for the participants' data file in the working
# directory, the way it does when the notebook runs from the repository root.
os.chdir(REPO)

from session3 import fitting as fitting3          # noqa: E402
from session4 import fitting as fitting4          # noqa: E402

N_PARTICIPANTS = 75
N_TRIALS = 160
SEED = 2025

# named as the notebooks name them
MODELS = ['1 α, multiplicative utility', '2 α, multiplicative utility',
          '1 α, additive utility', '2 α, additive utility']


def rounded(values, dp=4):
    return [round(float(v), dp) for v in values]


def likelihood_ratio(ll_alt, ll_null, df=N_PARTICIPANTS):
    """The notebooks' test: twice the summed log likelihood gain, against chi2."""
    statistic = 2 * float(np.sum(np.asarray(ll_alt) - np.asarray(ll_null)))
    return {'lambda': round(statistic, 2), 'df': df,
            'p': float(chi2.sf(statistic, df))}


def paired_t(stable, volatile):
    """One sample t test on the within participant difference, as in the notebook."""
    result = ttest_1samp(np.asarray(volatile) - np.asarray(stable), 0,
                         alternative='greater')
    return {'t': round(float(result.statistic), 3), 'df': N_PARTICIPANTS - 1,
            'p': float(result.pvalue)}


# --- the fits to real participants, saved alongside the session 4 notebook ---

fits = {name: pd.read_csv(REPO / 'session4' / f'{name}.csv') for name in
        ('data1AlphaMul', 'data2AlphaMul', 'data1AlphaAdd', 'data2AlphaAdd')}

FITS = {
    'mul1': {'alpha': rounded(fits['data1AlphaMul'].alpha),
             'beta': rounded(fits['data1AlphaMul'].beta)},
    'mul2': {'alphaStable': rounded(fits['data2AlphaMul'].alphaStable),
             'alphaVolatile': rounded(fits['data2AlphaMul'].alphaVolatile),
             'beta': rounded(fits['data2AlphaMul'].beta)},
    'add1': {'alpha': rounded(fits['data1AlphaAdd'].alpha),
             'beta': rounded(fits['data1AlphaAdd'].beta),
             'omega': rounded(fits['data1AlphaAdd'].omega)},
    'add2': {'alphaStable': rounded(fits['data2AlphaAdd'].alphaStable),
             'alphaVolatile': rounded(fits['data2AlphaAdd'].alphaVolatile),
             'beta': rounded(fits['data2AlphaAdd'].beta),
             'omega': rounded(fits['data2AlphaAdd'].omega)},
}

# --- the two simulations the session 3 notebook runs -------------------------

print('simulating with one learning rate for both blocks, and refitting...')
rng = np.random.default_rng(SEED)
same1, same2 = fitting3.fit_participant_data(
    fitting3.multiplicative_utility, simulate=True,
    alpha_S=fits['data2AlphaMul'].alphaStable,
    alpha_V=fits['data2AlphaMul'].alphaStable,
    beta=fits['data2AlphaMul'].beta, rng=rng)

print('simulating again with the volatile learning rates shuffled between '
      'participants, and refitting...')
shuffled1, shuffled2 = fitting3.fit_participant_data(
    fitting3.multiplicative_utility, simulate=True,
    alpha_S=fits['data2AlphaMul'].alphaStable,
    alpha_V=rng.permutation(fits['data2AlphaMul'].alphaStable),
    beta=fits['data2AlphaMul'].beta, rng=rng)

SIMULATIONS = {
    key: {
        'alphaStable': rounded(two.alphaStable),
        'alphaVolatile': rounded(two.alphaVolatile),
        'test': likelihood_ratio(two.LL, one.LL),
    }
    for key, (one, two) in (('same', (same1, same2)),
                            ('shuffled', (shuffled1, shuffled2)))
}

# --- the tests the slides quote ---------------------------------------------

STATS = {
    'likelihoodRatio': {
        'mul': likelihood_ratio(fits['data2AlphaMul'].LL, fits['data1AlphaMul'].LL),
        'add': likelihood_ratio(fits['data2AlphaAdd'].LL, fits['data1AlphaAdd'].LL),
    },
    'tTest': {
        'mul': paired_t(fits['data2AlphaMul'].alphaStable,
                        fits['data2AlphaMul'].alphaVolatile),
        'add': paired_t(fits['data2AlphaAdd'].alphaStable,
                        fits['data2AlphaAdd'].alphaVolatile),
    },
    'bic': {key: round(float(frame.BIC.sum()), 1) for key, frame in
            (('mul1', fits['data1AlphaMul']), ('mul2', fits['data2AlphaMul']),
             ('add1', fits['data1AlphaAdd']), ('add2', fits['data2AlphaAdd']))},
    'nParticipants': N_PARTICIPANTS,
    'nTrials': N_TRIALS,
}

# --- random effects model comparison, from section 3 of the notebook --------

# Stacked the way the notebook stacks them, one column per model, no
# participants excluded: the slide shows the whole sample.
BICs = np.column_stack([fits[name].BIC for name in
                        ('data1AlphaMul', 'data2AlphaMul', 'data1AlphaAdd', 'data2AlphaAdd')])

# The simple count first: which model has the lowest BIC for each participant.
best_counts = np.zeros(len(MODELS))
for row in BICs:
    best_counts[np.argmin(row)] += 1

print('estimating model frequencies and exceedance probabilities...')
dirichlet_alphas, dirichlet_sd = fitting3.estimate_model_frequencies(BICs)
exceedance = fitting3.estimate_exceedance_probabilities(dirichlet_alphas)

MODEL_FREQUENCIES = {
    'labels': MODELS,
    # the count of participants each model wins outright, as a proportion
    'counted': rounded(best_counts / best_counts.sum(), 3),
    # and the Dirichlet estimate, which also weighs how clear each win was
    'estimated': rounded(dirichlet_alphas / dirichlet_alphas.sum(), 3),
    'sd': rounded(dirichlet_sd, 3),
    'exceedance': rounded(exceedance, 3),
}

# --- the chi2 null distribution the likelihood ratio tests are read against --

chi2_x = np.linspace(30, 430, 401)
CHI2 = {
    'df': N_PARTICIPANTS,
    'x': rounded(chi2_x, 1),
    'density': [round(float(v), 8) for v in chi2.pdf(chi2_x, N_PARTICIPANTS)],
    # the value beyond which the test is significant at p < 0.05
    'critical': round(float(chi2.isf(0.05, N_PARTICIPANTS)), 1),
}

# --- model and parameter recovery, from the saved 100 simulated datasets -----

recoveries = {name: pd.read_csv(REPO / 'session4' / f'{name}.csv') for name in
              ('recov1AlphaMul', 'recov2AlphaMul', 'recov1AlphaAdd', 'recov2AlphaAdd')}
n_reps = int((recoveries['recov2AlphaAdd'].ID.max() + 1) / N_PARTICIPANTS)

MODEL_RECOVERY = {
    'labels': MODELS,
    'nDatasets': n_reps,
    # Row: the model a dataset was simulated with. Column: the model the BIC
    # then picked. The diagonal is how often the BIC found the right one.
    'matrix': [rounded(fitting4.recov_BICs(recoveries[name], N_PARTICIPANTS, n_reps) / n_reps, 3)
               for name in ('recov1AlphaMul', 'recov2AlphaMul',
                            'recov1AlphaAdd', 'recov2AlphaAdd')],
}

# One dataset's worth of simulated against recovered parameters, rather than all
# 100: the scatter makes its point at 75 points and stays small enough to ship.
one_dataset = recoveries['recov2AlphaMul'].iloc[:N_PARTICIPANTS * 4]
PARAMETER_RECOVERY = {
    'alphaStable': {'simulated': rounded(one_dataset.alphaStable),
                    'recovered': rounded(one_dataset.recovered2MulAlphaS)},
    'alphaVolatile': {'simulated': rounded(one_dataset.alphaVolatile),
                      'recovered': rounded(one_dataset.recovered2MulAlphaV)},
    'beta': {'simulated': rounded(one_dataset.beta),
             'recovered': rounded(one_dataset.recovered2MulBeta)},
}

# --- write it out ------------------------------------------------------------

blocks = [
    ('FITS', FITS, 'Parameters fitted to each of the 75 participants, under all '
                   'four models.'),
    ('SIMULATIONS', SIMULATIONS, "The session 3 notebook's two control simulations, "
                                 'run against the participants own schedules.'),
    ('STATS', STATS, 'The tests the slides quote, computed exactly as the notebook '
                     'computes them.'),
    ('MODEL_FREQUENCIES', MODEL_FREQUENCIES,
     'How often each model wins across participants, allowing different '
     'participants to use different models, and how likely each is to be the '
     'most common one.'),
    ('CHI2', CHI2, 'The chi2 null distribution the likelihood ratio statistics are read '
                   'against, and its 5% critical value.'),
    ('MODEL_RECOVERY', MODEL_RECOVERY, 'How often the BIC picks each model out, over '
                                       f'{n_reps} simulated datasets.'),
    ('PARAMETER_RECOVERY', PARAMETER_RECOVERY, 'Simulated against recovered parameters '
                                               'for four simulated datasets.'),
]

out = ['/* Generated by task-source/precompute-fits.py. Do not edit by hand:',
       '   re-run that script instead. It holds everything sessions 3 and 4 plot',
       '   about the Blain and Rutledge data, fitted with the practical\'s own code. */',
       '']
for name, value, comment in blocks:
    out.append(f'/** {comment} */')
    out.append(f'export const {name} = {json.dumps(value)};')
    out.append('')

(REPO / 'docs' / 'assets' / 'participant-fits.js').write_text('\n'.join(out))

print('wrote docs/assets/participant-fits.js')
for key, test in STATS['likelihoodRatio'].items():
    print(f"  likelihood ratio, {key}: chi2({test['df']}) = {test['lambda']}, p = {test['p']:.3g}")
for key, test in SIMULATIONS.items():
    print(f"  simulated {key}: chi2 = {test['test']['lambda']}, p = {test['test']['p']:.3g}")
for key, test in STATS['tTest'].items():
    print(f"  t test, {key}: t({test['df']}) = {test['t']}, p = {test['p']:.3g}")
print('  BIC totals: ' + ', '.join(f'{k} {v}' for k, v in STATS['bic'].items()))
print('  model frequencies: '
      + ', '.join(f'{l.split(",")[0]} {f:.2f} (xp {x:.3f})' for l, f, x
                  in zip(MODELS, MODEL_FREQUENCIES['estimated'], MODEL_FREQUENCIES['exceedance'])))
print('  model recovery diagonal: '
      + ', '.join(f'{row[i]:.2f}' for i, row in enumerate(MODEL_RECOVERY['matrix'])))
