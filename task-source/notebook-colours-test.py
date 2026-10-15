"""
Draws every figure the four notebooks can draw, and checks each one only uses
colours from the shared palette (sessionN/palette.py).

    cd task-source && python3 notebook-colours-test.py

Needs the notebooks' own requirements (sessionN/requirements.txt). Each trace
must name its colour explicitly: the notebooks use plotly's "none" template,
under which an uncoloured trace silently comes out in plotly's default blue,
and that blue means nothing in this course.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.basedatatypes as basedatatypes

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# session3 and session4 load the participants' data from the working directory
os.chdir(REPO)

# --- capture every figure instead of displaying it --------------------------

captured = []
basedatatypes.BaseFigure.show = lambda fig, *a, **k: captured.append(fig)

import IPython.display  # noqa: E402

def _display(*objects, **kwargs):
    """Collect figures, looking inside the widget boxes most of them come in."""
    for o in objects:
        if isinstance(o, basedatatypes.BaseFigure):
            captured.append(o)
        elif hasattr(o, 'children'):
            _display(*o.children)
        else:
            widgets_shown.append(o)

widgets_shown = []
IPython.display.display = _display

from session1 import plotting as p1, palette  # noqa: E402
from session2 import plotting as p2  # noqa: E402
from session3 import plotting as p3  # noqa: E402
from session4 import plotting as p4, loading as l4  # noqa: E402

for module in (p1, p2, p3, p4):
    module.display = _display   # they import display by name

PALETTE = {c.lower() for c in (palette.OPTION1, palette.OPTION2, palette.STABLE,
                               palette.VOLATILE, palette.ESTIMATE, palette.TRUTH,
                               palette.NEUTRAL)}
VIRIDIS = go.Surface(colorscale=palette.CONTINUOUS).colorscale

# --- the figures ---------------------------------------------------------------

rng = np.random.default_rng(0)
n = 120
true_p = np.r_[np.full(60, 0.8), np.full(60, 0.2)]
rewarded = (rng.random(n) < true_p).astype(float)
mag1, mag2 = rng.integers(1, 100, n).astype(float), rng.integers(1, 100, n).astype(float)

# The notebooks have students write these themselves; these stand in for them.

def rl_model(opt1Rewarded, alpha, startingProb=0.5):
    p = np.empty(len(opt1Rewarded)); p[0] = startingProb
    for t in range(len(opt1Rewarded) - 1):
        p[t + 1] = p[t] + alpha * (opt1Rewarded[t] - p[t])
    return p

def multiplicative_utility(mag, prob):
    return mag * prob

def simulate_RL_model(opt1Rewarded, magOpt1, magOpt2, alpha, beta, *omega,
                      utility_function=multiplicative_utility, startingProb=0.5):
    probOpt1 = rl_model(opt1Rewarded, alpha, startingProb)
    u1 = utility_function(magOpt1, probOpt1, *omega)
    u2 = utility_function(magOpt2, 1 - probOpt1, *omega)
    return probOpt1, 1 / (1 + np.exp(-beta * (u1 - u2)))

def figures():
    """(label, call) for every plotting function, with each optional series on."""
    probs = rl_model(rewarded, 0.3)
    choice_p = 1 / (1 + np.exp(-(probs - 0.5) * 6))
    choices = (rng.random(n) < choice_p).astype(float)

    yield 's1 plot_schedule', lambda: captured.append(p1.plot_schedule(rewarded, true_p, probs))
    yield 's1 plot_interactive_RL_model', lambda: p1.plot_interactive_RL_model(
        rewarded, true_p, rl_model, lambda tp: (rng.random(len(tp)) < tp).astype(float))
    yield 's1 visualise_utility_function', lambda: p1.visualise_utility_function(lambda m, p: m * p)
    yield 's1 plot_RL_weights', lambda: p1.plot_RL_weights()

    yield 's2 plot_schedule', lambda: captured.append(p2.plot_schedule(
        rewarded, true_p, mag1, mag2, probs, choice_p, choices, mag1 * probs, mag2 * (1 - probs)))
    yield 's2 plot_interactive_RL_model', lambda: p2.plot_interactive_RL_model(
        simulate_RL_model, multiplicative_utility, rewarded, mag1, mag2, true_p)
    yield 's2 visualise_utility_function', lambda: p2.visualise_utility_function(lambda m, p: m * p)
    yield 's2 visualise_softmax', lambda: p2.visualise_softmax(
        lambda u1, u2, beta: 1 / (1 + np.exp(-beta * (u1 - u2))))
    yield 's2 plot_likelihood_landscapes', lambda: p2.plot_likelihood_landscapes(
        rewarded[:60], mag1[:60], mag2[:60], choices[:60])
    yield 's2 plot_recovered_parameters', lambda: p2.plot_recovered_parameters(pd.DataFrame({
        'simulatedAlpha': rng.random(20), 'recoveredAlpha': rng.random(20),
        'simulatedBeta': rng.random(20) * 0.5, 'recoveredBeta': rng.random(20) * 0.5}))
    yield 's2 visualise_alpha_difference', lambda: p2.visualise_alpha_difference(
        rng.random(30) * 0.4, rng.random(30) * 0.6, '')
    yield 's2 plot_regression_weights', regression_weights

    fits = {name: pd.read_csv(REPO / 'session4' / f'{name}.csv') for name in
            ('data1AlphaMul', 'data2AlphaMul', 'data1AlphaAdd', 'data2AlphaAdd')}
    for label, module in (('s3', p3), ('s4', p4)):
        yield f'{label} plot_schedule', lambda m=module: captured.append(m.plot_schedule(0, fits['data2AlphaAdd']))
        yield f'{label} visualise_alpha_distributions', lambda m=module: m.visualise_alpha_distributions(
            fits['data2AlphaMul'].alphaStable, fits['data2AlphaMul'].alphaVolatile, '')
        yield f'{label} visualise_alpha_difference', lambda m=module: m.visualise_alpha_difference(
            fits['data2AlphaMul'].alphaStable, fits['data2AlphaMul'].alphaVolatile, '')
        for name in fits:
            yield f'{label} plot_parameter_corrs {name}', lambda m=module, d=fits[name]: m.plot_parameter_corrs(d)

    yield 's3 plot_model_frequencies', lambda: p3.plot_model_frequencies(
        np.array([9., 3., 35., 28.]), np.array([.03, .02, .05, .05]), np.array([0., 0., .77, .23]))

    recoveries = l4.load_parameter_recovery()
    for name, recovery in zip(('1 α mul', '2 α mul', '1 α add', '2 α add'), recoveries):
        yield f's4 plot_recovered_parameters {name}', lambda r=recovery: p4.plot_recovered_parameters(r)
    for fn in ('visualise_LR_recovery', 'visualise_BIC_recovery', 'visualise_t_test_recovery'):
        yield f's4 {fn}', lambda fn=fn: getattr(p4, fn)(*recoveries)

def regression_weights():
    """Fit a small logistic regression and toggle the checkbox, so the derived
    beta and omega bars are drawn too."""
    import statsmodels.formula.api as smf
    data = pd.DataFrame({f'lag{k}': rng.normal(size=200) for k in range(1, 6)})
    data['magDiff'] = rng.normal(size=200)
    data['choice'] = (data.sum(axis=1) + rng.normal(size=200) > 0).astype(int)
    model = smf.logit('choice ~ 0 + lag1 + lag2 + lag3 + lag4 + lag5 + magDiff', data).fit(disp=0)
    p2.plot_regression_weights(model)
    checkbox = next(w for w in reversed(widgets_shown) if w.__class__.__name__ == 'Checkbox')
    checkbox.value = True

# --- what each trace is coloured with ----------------------------------------------

def colours(trace):
    """Every explicit colour a trace carries, and whether it carries a colour scale."""
    found, scale = [], None
    for path in ('marker.color', 'line.color', 'marker.line.color', 'textfont.color',
                 'error_y.color', 'fillcolor'):
        obj = trace
        for part in path.split('.'):
            obj = getattr(obj, part, None) if obj is not None else None
        if obj is None:
            continue
        found.extend(obj if isinstance(obj, (list, tuple, np.ndarray)) else [obj])
    for holder in (trace, getattr(trace, 'marker', None)):
        cs = getattr(holder, 'colorscale', None) if holder is not None else None
        if cs:
            scale = cs
    return [c for c in found if isinstance(c, str)], scale

failures = []
for label, draw in figures():
    before = len(captured)
    draw()
    figs = captured[before:]
    if not figs:
        failures.append(f'{label}: drew no figure')
    for fig in figs:
        for i, trace in enumerate(fig.data):
            found, scale = colours(trace)
            name = f'{label} trace {i} ({trace.type} "{trace.name or ""}")'
            if scale is not None and tuple(map(tuple, scale)) != tuple(map(tuple, VIRIDIS)):
                failures.append(f'{name}: colour scale is not {palette.CONTINUOUS}')
            if scale is None and not found:
                failures.append(f'{name}: no explicit colour, so it would come out in plotly blue')
            stray = sorted({c for c in found if c.lower() not in PALETTE})
            if stray:
                failures.append(f'{name}: colours outside the palette: {stray}')
    print(f'  drew {label}: {len(figs)} figure(s)')

print()
if failures:
    print('FAIL')
    for f in failures:
        print('  ' + f)
    sys.exit(1)
print(f'PASS  {len(captured)} figures, every trace coloured from the palette')
