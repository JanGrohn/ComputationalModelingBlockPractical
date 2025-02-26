# Import all functions from fitting module
from .fitting import (
    utility_fun,
    loss_RL_model,
    run_opt,
    fit_model_same_alpha,
    fit_model_alpha_difference,
    fit_with_multiple_initial_values,
    fit_multiple_participants,
    run_paramterer_recovery,
    run_paramterer_recovery_with_difference
)

# Import all functions from plotting module
from .plotting import (
    plot_schedule,
    visualise_utility_function,
    visualise_softmax,
    plot_interactive_RL_model,
    plot_likelihood_landscapes,
    plot_recovered_parameters,
    visualise_alpha_difference
)

