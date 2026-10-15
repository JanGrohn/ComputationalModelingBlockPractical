'''
The colours every plot in this Advanced Lab uses, and what each one means.

Each colour stands for exactly one thing, in the notebooks, on the slides and
in the task alike, so a colour seen anywhere can be read the same way
everywhere. This file is copied into each session folder because Colab copies
one session folder at a time; task-source/palette-test.mjs fails if the copies,
the slides (docs/assets/deck.js) or the task's stimuli ever disagree.
'''

OPTION1  = '#2e7d32'  # option 1, the green deer: its outcomes, magnitude, utility,
                      # choices of it, and the model's probabilities about it
OPTION2  = '#ed6c02'  # option 2, the orange crane, likewise
STABLE   = '#1f4b73'  # the stable block
VOLATILE = '#c2185b'  # the volatile block
ESTIMATE = '#6a3d9a'  # a fitted or estimated value: best fits, fitted lines
TRUTH    = '#14171a'  # the true values a simulation used; drawn dashed
NEUTRAL  = '#7d868d'  # data with no category of its own

# Continuous quantities (likelihoods, utilities, proportions) use viridis: it is
# perceptually uniform, and no single flat colour in it competes with the above.
CONTINUOUS = 'Viridis'


def by_option(option1_flags):
  '''One colour per trial: option 1's where the flag is set, option 2's where not.'''
  return [OPTION1 if flag else OPTION2 for flag in option1_flags]
