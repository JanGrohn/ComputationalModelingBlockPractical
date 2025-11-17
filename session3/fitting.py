# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

from scipy.stats import logistic

# import jax (for fast vectorised operations) and optax (for optimizers)
import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from functools import partial

# import typing to define the types of the variables
from typing import Tuple, Dict, Any, Callable, Optional, Union
import numpy.typing as npt
from jax._src.prng import PRNGKeyArray

# import the digamma and betainc functions from scipy
from scipy.special import digamma, betainc

from session3 import loading


@jax.jit # this is a decorator that tells jax to compile the function
def multiplicative_utility(mag: jnp.ndarray, prob: jnp.ndarray) -> jnp.ndarray:
  return mag*prob

@jax.jit 
def additive_utility(mag: jnp.ndarray, prob: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
  return (omega*(mag/100) + (1-omega)*prob)*100

@partial(jax.jit, static_argnames=['utility_function']) # this is a decorator that tells jax to compile the function and to use the utility_function as a static argument
def loss_RL_model(
  data:             Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray],
  params:           Dict[str, jnp.ndarray],
  startingProb:     float = 0.5,
  utility_function: Callable = multiplicative_utility) -> float:
    
    '''
    Returns the loss of the data given the choices and the model 

    Parameters:
        data(tuple): tuple containing the following arrays:
          opt1rewarded(bool array): True if option 1 is rewarded on a trial, False
            if option 2 is rewarded on a trial.
          magOpt1(int array): reward points between 1 and 100 for option 1 on each
            trial
          magOpt2(int array): reward points between 1 and 100 for option 2 on each
            trial
          choice1(bool array): True if option 1 was chosen on a trial, False if
            option 2 was chosen on a trial.
        params(dict): dictionary of parameters for the model
        startingProb(float): starting probability of choosing option 1
        utility_function(function): function to calculate the utility of the options

    Returns:
        loss(float): total loss of the data given the input parameters
    '''
    
    opt1Rewarded, magOpt1, magOpt2, choice1 = data

    alpha = otu.tree_get(params, 'alpha')[0]
    beta  = otu.tree_get(params, 'beta')[0]

    nTrials = len(opt1Rewarded)

    # This function will be called for each trial and will update the probability of choosing option 1
    def compute_probOpt1(probOpt1, t):
        delta = opt1Rewarded[t] - probOpt1
        new_probOpt1 = probOpt1 + alpha * delta
        return new_probOpt1, new_probOpt1

    # Convert everything to JAX arrays
    opt1Rewarded = jnp.array(opt1Rewarded)
    magOpt1      = jnp.array(magOpt1)
    magOpt2      = jnp.array(magOpt2)
    choice1      = jnp.array(choice1)

    # Use jax.lax.scan to iterate over the trials and collect results
    _, probOpt1 = jax.lax.scan(compute_probOpt1, startingProb, jnp.arange(nTrials))

    # Create the new vector with startingProb as the first element and probOpt1 excluding the last element
    probOpt1 = jnp.concatenate([jnp.array([startingProb]), probOpt1[:-1]])
    
    # Compute the utility of the two options
    if utility_function == multiplicative_utility:
      utility1 = utility_function(magOpt1, probOpt1)
      utility2 = utility_function(magOpt2, 1-probOpt1)
    elif utility_function == additive_utility:
      omega = otu.tree_get(params, 'omega')[0]
      utility1 = utility_function(magOpt1, probOpt1, omega)
      utility2 = utility_function(magOpt2, 1-probOpt1, omega)
    else:
      raise ValueError(f"Utility function {utility_function} not supported")
    
    # Compute the loss of the model
    choice1Logits = jnp.clip(beta * (utility1 - utility2), -50.0, 50.0)
    loss = optax.sigmoid_binary_cross_entropy(choice1Logits, choice1).sum()
    
    # Return NaN if loss is invalid, this helps debugging
    loss = jnp.where(jnp.isfinite(loss), loss, jnp.inf)
    
    return loss

def run_opt(
    init_params: Dict[str, jnp.ndarray],
    fun:         Callable,
    opt:         optax.GradientTransformation = optax.lbfgs(),
    max_iter:    int = 100,
    tol:         float = 1e-3,
    paramsClip:  Dict[str, float] = {'alphaMin': 0, 'alphaMax': 1, 'betaMin': 0, 'betaMax': 1, 'omegaMin': 0, 'omegaMax': 1}
    ) -> Tuple[Dict[str, jnp.ndarray], Any]:
  '''
  Runs the optimization process. Based on https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html

  Parameters:
    init_params(dict): the initial parameters for the model
    fun(function): the loss function to minimize
    opt(optax optimizer): the optimizer to use
    max_iter(int): the maximum number of iterations to run
    tol(float): the tolerance for the optimization
    paramsClip(dict): the parameters to clip

  Returns:
    final_params(dict): the final parameters after optimization
  '''
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def project_params(params, paramsClip):
    def clip_param(param, name):
      if 'alpha' in name:
        return jnp.clip(param, otu.tree_get(paramsClip, 'alphaMin'), otu.tree_get(paramsClip, 'alphaMax'))
      elif 'beta' in name:
        return jnp.clip(param, otu.tree_get(paramsClip, 'betaMin'), otu.tree_get(paramsClip, 'betaMax'))
      elif 'omega' in name:
        return jnp.clip(param, otu.tree_get(paramsClip, 'omegaMin'), otu.tree_get(paramsClip, 'omegaMax'))
      return param

    def apply_clip(params, path=''):
      if isinstance(params, dict):
        return {k: apply_clip(v, path + '.' + k if path else k) for k, v in params.items()}
      return clip_param(params, path)

    return apply_clip(params)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = optax.apply_updates(params, updates)
    params = project_params(params, paramsClip)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = otu.tree_get(state, 'count')
    grad = otu.tree_get(state, 'grad')
    err = otu.tree_l2_norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(
      continuing_criterion, step, init_carry
  )
  return final_params, final_state

@partial(jax.jit,static_argnames=['utility_function', 'opt', 'max_iter', 'tol', 'startingProb'])
def fit_model_same_alpha(
  data:             Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray],
  rng:              Optional[PRNGKeyArray] = None,
  startingProb:     float = 0.5,
  paramsClip:       Dict[str, float] = {'alphaMin': 0, 'alphaMax': 1, 'betaMin': 0, 'betaMax': 1},
  utility_function: Callable = multiplicative_utility,
  opt:              optax.GradientTransformation = optax.lbfgs(),
  max_iter:         int = 100,
  tol:              float = 1e-3,
  startingParams:   Optional[Dict[str, jnp.ndarray]] = None
  ) -> Tuple[Dict[str, jnp.ndarray], Any, float]:
    '''
    Fits the model with the same learning rate for both blocks

    Parameters:
        data(tuple): tuple containing the following arrays:
          opt1rewardedStable(bool array): True if option 1 is rewarded on a trial, False
            if option 2 is rewarded on a trial.
          magOpt1Stable(int array): reward points between 1 and 100 for option 1 on each
            trial
          magOpt2Stable(int array): reward points between 1 and 100 for option 2 on each
            trial
          choice1Stable(bool array): True if option 1 was chosen on a trial, False if
            option 2 was chosen on a trial.
          opt1rewardedVolatile(bool array): True if option 1 is rewarded on a trial, False
            if option 2 is rewarded on a trial.
          magOpt1Volatile(int array): reward points between 1 and 100 for option 1 on each
            trial
          magOpt2Volatile(int array): reward points between 1 and 100 for option 2 on each
            trial
          choice1Volatile(bool array): True if option 1 was chosen on a trial, False if
            option 2 was chosen on a trial.
        rng(PRNGKeyArray): the random number generator to use
        startingProb(float): the starting probability of choosing option 1
        paramsClip(dict): the parameters to constrain
        utility_function(function): the utility function to use
        opt(optax optimizer): the optimizer to use
        max_iter(int): the maximum number of iterations to run
        tol(float): the tolerance for the optimization
        startingParams(dict): the starting parameters to use

    Returns:
        finalParams(dict): the final parameters after optimization
        finalState(Any): the final state after optimization
        finalLoss(float): the final loss after optimization
    '''

    # split the data into stable and volatile blocks
    opt1RewardedStable, magOpt1Stable, magOpt2Stable, choice1Stable, opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile = data
    dataStable   = (opt1RewardedStable,   magOpt1Stable,   magOpt2Stable,   choice1Stable)
    dataVolatile = (opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile)

    if utility_function == multiplicative_utility:
      if rng is None:
        rng = jax.random.PRNGKey(0)
      rng, alphaRng, betaRng = jax.random.split(rng, 3)

      if startingParams is not None:
        alpha = jax.random.uniform(alphaRng, shape=(1,), minval=startingParams['alpha'], maxval=startingParams['alpha'],)
        beta  = jax.random.uniform(betaRng,  shape=(1,), minval=startingParams['beta'],  maxval=startingParams['beta'])
      else:
        alpha = jax.random.uniform(alphaRng, shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
        beta  = jax.random.uniform(betaRng,  shape=(1,), minval=paramsClip['betaMin'],  maxval=paramsClip['betaMax'] )

        # initialise the parameters
        initParams = {'alpha': alpha, 'beta': beta}
    
    elif utility_function == additive_utility:
      if rng is None:
        rng = jax.random.PRNGKey(0)
      rng, alphaRng, betaRng, omegaRng = jax.random.split(rng, 4)

      if startingParams is not None:
        alpha = jax.random.uniform(alphaRng, shape=(1,), minval=startingParams['alpha'], maxval=startingParams['alpha'],)
        beta  = jax.random.uniform(betaRng,  shape=(1,), minval=startingParams['beta'],  maxval=startingParams['beta'])
        omega = jax.random.uniform(omegaRng, shape=(1,), minval=startingParams['omega'], maxval=startingParams['omega'])
      else:
        alpha = jax.random.uniform(alphaRng, shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
        beta  = jax.random.uniform(betaRng,  shape=(1,), minval=paramsClip['betaMin'],  maxval=paramsClip['betaMax'] )
        omega = jax.random.uniform(omegaRng, shape=(1,), minval=paramsClip['omegaMin'], maxval=paramsClip['omegaMax'])

        # initialise the parameters
        initParams = {'alpha': alpha, 'beta': beta, 'omega': omega}
         
    # define the loss function
    def loss_fun(params):
      lossStable   = loss_RL_model(dataStable,   params, startingProb = startingProb, utility_function = utility_function)
      lossVolatile = loss_RL_model(dataVolatile, params, startingProb = startingProb, utility_function = utility_function)
      return lossStable + lossVolatile
    
    finalParams, finalState = run_opt(initParams, loss_fun, opt, max_iter=max_iter, tol=tol, paramsClip=paramsClip)
    finalLoss = loss_fun(finalParams)
    return finalParams, finalState, finalLoss

@partial(jax.jit,static_argnames=['utility_function', 'opt', 'max_iter', 'tol', 'startingProb'])
def fit_model_alpha_difference(
  data:             Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray],
  rng:              Optional[PRNGKeyArray] = None,
  startingProb:     float = 0.5,
  paramsClip:       Dict[str, float] = {'alphaMin': 0, 'alphaMax': 1, 'betaMin': 0, 'betaMax': 1},
  utility_function: Callable = multiplicative_utility,
  opt:              optax.GradientTransformation = optax.lbfgs(),
  max_iter:         int = 100,
  tol:              float = 1e-3,
  startingParams:   Optional[Dict[str, jnp.ndarray]] = None
  ) -> Tuple[Dict[str, jnp.ndarray], Any, float]:
    '''
    Fits the model with different learning rates for the stable and volatile blocks

    Parameters:
        opt1rewardedStable(bool array): True if option 1 is rewarded on a trial, False
            if option 2 is rewarded on a trial.
          magOpt1Stable(int array): reward points between 1 and 100 for option 1 on each
            trial
          magOpt2Stable(int array): reward points between 1 and 100 for option 2 on each
            trial
          choice1Stable(bool array): True if option 1 was chosen on a trial, False if
            option 2 was chosen on a trial.
          opt1rewardedVolatile(bool array): True if option 1 is rewarded on a trial, False
            if option 2 is rewarded on a trial.
          magOpt1Volatile(int array): reward points between 1 and 100 for option 1 on each
            trial
          magOpt2Volatile(int array): reward points between 1 and 100 for option 2 on each
            trial
          choice1Volatile(bool array): True if option 1 was chosen on a trial, False if
            option 2 was chosen on a trial.
        rng(PRNGKeyArray): the random number generator to use
        startingProb(float): the starting probability of choosing option 1
        paramsClip(dict): the parameters to constrain
        utility_function(function): the utility function to use
        opt(optax optimizer): the optimizer to use
        max_iter(int): the maximum number of iterations to run
        tol(float): the tolerance for the optimization
        startingParams(dict): the starting parameters to use
    Returns:
        finalParams(dict): the final parameters after optimization
        finalState(Any): the final state after optimization
        finalLoss(float): the final loss after optimization
    '''

    # split the data into stable and volatile blocks
    opt1RewardedStable, magOpt1Stable, magOpt2Stable, choice1Stable, opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile = data
    dataStable   = (opt1RewardedStable,   magOpt1Stable,   magOpt2Stable,   choice1Stable)
    dataVolatile = (opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile)

    if utility_function == multiplicative_utility:
      if rng is None:
        rng = jax.random.PRNGKey(0)
      rng, alphaStableRng, alphaVolatileRng, betaRng = jax.random.split(rng, 4)

      # initialise the parameters
      if startingParams is not None:
        alphaStable   = jax.random.uniform(alphaStableRng,   shape=(1,), minval=startingParams['alphaStable'],   maxval=startingParams['alphaStable'],)
        alphaVolatile = jax.random.uniform(alphaVolatileRng, shape=(1,), minval=startingParams['alphaVolatile'], maxval=startingParams['alphaVolatile'])
        beta          = jax.random.uniform(betaRng,          shape=(1,), minval=startingParams['beta'],          maxval=startingParams['beta'])
      else:
        alphaStable   = jax.random.uniform(alphaStableRng,   shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
        alphaVolatile = jax.random.uniform(alphaVolatileRng, shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
        beta          = jax.random.uniform(betaRng,          shape=(1,), minval=paramsClip['betaMin'],  maxval=paramsClip['betaMax'])

      initParams = {'alphaStable': alphaStable, 'alphaVolatile': alphaVolatile, 'beta': beta}
    
    elif utility_function == additive_utility:
      if rng is None:
        rng = jax.random.PRNGKey(0)
      rng, alphaStableRng, alphaVolatileRng, betaRng, omegaRng = jax.random.split(rng, 5)

      # initialise the parameters
      if startingParams is not None:
        alphaStable   = jax.random.uniform(alphaStableRng,   shape=(1,), minval=startingParams['alphaStable'],   maxval=startingParams['alphaStable'],)
        alphaVolatile = jax.random.uniform(alphaVolatileRng, shape=(1,), minval=startingParams['alphaVolatile'], maxval=startingParams['alphaVolatile'])
        beta          = jax.random.uniform(betaRng,          shape=(1,), minval=startingParams['beta'],          maxval=startingParams['beta'])
        omega         = jax.random.uniform(omegaRng,         shape=(1,), minval=startingParams['omega'],         maxval=startingParams['omega'])
      else:
        alphaStable   = jax.random.uniform(alphaStableRng,   shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
        alphaVolatile = jax.random.uniform(alphaVolatileRng, shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
        beta          = jax.random.uniform(betaRng,          shape=(1,), minval=paramsClip['betaMin'],  maxval=paramsClip['betaMax'])
        omega         = jax.random.uniform(omegaRng,         shape=(1,), minval=paramsClip['omegaMin'], maxval=paramsClip['omegaMax'])

      initParams = {'alphaStable': alphaStable, 'alphaVolatile': alphaVolatile, 'beta': beta, 'omega': omega}
    
    # define the loss function
    def loss_fun(params):
      if utility_function == multiplicative_utility:
        stableParams   = {'alpha': params['alphaStable'],   'beta': params['beta']}
        volatileParams = {'alpha': params['alphaVolatile'], 'beta': params['beta']}
      elif utility_function == additive_utility:
        stableParams   = {'alpha': params['alphaStable'],   'beta': params['beta'], 'omega': params['omega']}
        volatileParams = {'alpha': params['alphaVolatile'], 'beta': params['beta'], 'omega': params['omega']}
      lossStable   = loss_RL_model(dataStable,   stableParams,   startingProb = startingProb, utility_function = utility_function)
      lossVolatile = loss_RL_model(dataVolatile, volatileParams, startingProb = startingProb, utility_function = utility_function)
      return lossStable + lossVolatile
    
    # run the optimization
    finalParams, finalState = run_opt(initParams, loss_fun, opt, max_iter = max_iter, tol = tol, paramsClip = paramsClip)
    finalLoss = loss_fun(finalParams)

    return finalParams, finalState, finalLoss

@partial(jax.jit, static_argnames=['nInits', 'utility_function', 'fit_model'])
def fit_with_multiple_initial_values(
  data:             Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray],
  nInits:           int = 10,
  utility_function: Callable = multiplicative_utility,
  fit_model:        Callable = fit_model_same_alpha,
  paramsClip:       Dict[str, float] = {'alphaMin': 0, 'alphaMax': 1, 'betaMin': 0, 'betaMax': 1},
  startingParams:   Optional[Dict[str, jnp.ndarray]] = None
  ) -> Dict[str, jnp.ndarray]:
    '''
    Fits the model with multiple initial values

    Parameters:
        data(tuple): tuple containing the following arrays:
          opt1rewardedStable(bool array): True if option 1 is rewarded on a trial, False
            if option 2 is rewarded on a trial.
          magOpt1Stable(int array): reward points between 1 and 100 for option 1 on each
            trial
          magOpt2Stable(int array): reward points between 1 and 100 for option 2 on each
            trial
          choice1Stable(bool array): True if option 1 was chosen on a trial, False if
            option 2 was chosen on a trial.
          opt1rewardedVolatile(bool array): True if option 1 is rewarded on a trial, False
            if option 2 is rewarded on a trial.
          magOpt1Volatile(int array): reward points between 1 and 100 for option 1 on each
            trial
          magOpt2Volatile(int array): reward points between 1 and 100 for option 2 on each
            trial
          choice1Volatile(bool array): True if option 1 was chosen on a trial, False if
            option 2 was chosen on a trial.
        nInits(int): the number of initial values to use
        utility_function(function): the utility function to use
        fit_model(function): the model to fit
        startingParams(dict): the starting parameters to use

    Returns:
        finalParams(dict): the final parameters after optimization
    '''
    opt1RewardedStable, magOpt1Stable, magOpt2Stable, choice1Stable, opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile = data
    opt1RewardedStableStack   = jnp.tile(opt1RewardedStable,   (nInits, 1))
    magOpt1StableStack        = jnp.tile(magOpt1Stable,        (nInits, 1))
    magOpt2StableStack        = jnp.tile(magOpt2Stable,        (nInits, 1))
    choice1StableStack        = jnp.tile(choice1Stable,        (nInits, 1))
    opt1RewardedVolatileStack = jnp.tile(opt1RewardedVolatile, (nInits, 1))
    magOpt1VolatileStack      = jnp.tile(magOpt1Volatile,      (nInits, 1))
    magOpt2VolatileStack      = jnp.tile(magOpt2Volatile,      (nInits, 1))
    choice1VolatileStack      = jnp.tile(choice1Volatile,      (nInits, 1))
    tiledData = (opt1RewardedStableStack, magOpt1StableStack, magOpt2StableStack, choice1StableStack, opt1RewardedVolatileStack, magOpt1VolatileStack, magOpt2VolatileStack, choice1VolatileStack)
      
    fit_model_vectorised = jax.vmap(partial(fit_model, utility_function = utility_function, paramsClip = paramsClip))
    
    initial_rngs = jnp.stack([jax.random.PRNGKey(i) for i in range(nInits)], axis=0)
    
    params, _, loss = fit_model_vectorised(tiledData, initial_rngs)

    # if startingParams is not None, we start one optimisation run from the supplied startingParams
    if startingParams is not None:
      paramsKnown, _, lossKnown = fit_model(data, utility_function = utility_function, paramsClip = paramsClip, startingParams = startingParams)
      
      loss = jnp.concatenate([loss, jnp.array([lossKnown])], axis=0)
      params = {
        k: jnp.concatenate([params[k], jnp.expand_dims(paramsKnown[k], axis=0)], axis=0)
        for k in params.keys()
      }
      

    def get_params(params, loss):
        '''
        Gets the best fitting parameters

        Parameters:
            params(dict): the final parameters
            loss(float): the loss of the parameters

        Returns:
            bestParams(dict): the best fitting parameters
        '''
        bestFit = jnp.nanargmin(loss)
        if fit_model == fit_model_same_alpha:
          bestAlpha = params['alpha'][bestFit]
          bestBeta  = params['beta'][bestFit]
          bestLoss  = loss[bestFit]
          if utility_function == multiplicative_utility:
            bestParams = {'alpha': bestAlpha, 'beta': bestBeta}
          elif utility_function == additive_utility:
            bestOmega = params['omega'][bestFit]
            bestParams = {'alpha': bestAlpha, 'beta': bestBeta, 'omega': bestOmega}
          
        elif fit_model == fit_model_alpha_difference:
          bestAlphaStable   = params['alphaStable'][bestFit]
          bestAlphaVolatile = params['alphaVolatile'][bestFit]
          bestBeta          = params['beta'][bestFit]
          bestLoss          = loss[bestFit]
          if utility_function == multiplicative_utility:
            bestParams = {'alphaStable': bestAlphaStable, 'alphaVolatile': bestAlphaVolatile, 'beta': bestBeta}
          elif utility_function == additive_utility:
            bestOmega = params['omega'][bestFit]
            bestParams = {'alphaStable': bestAlphaStable, 'alphaVolatile': bestAlphaVolatile, 'beta': bestBeta, 'omega': bestOmega}
          
        return bestParams, bestLoss

    return get_params(params, loss)

def fit_participant_data(
  utility_function: Callable,
  simulate:         bool = False,
  alpha_S:          Optional[npt.NDArray] = None,
  alpha_V:          Optional[npt.NDArray] = None,
  beta:             Optional[npt.NDArray] = None,
  omega:            Optional[npt.NDArray] = None,
  rng:              Optional[np.random.Generator] = None,
  nInits:           int = 10,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Fits the model to the data

    Parameters:
        utility_function(function): the utility function to use
        simulate(bool): whether to simulate the data
        alpha_S(array): the alpha values for the stable condition (if simulate is True)
        alpha_V(array): the alpha values for the volatile condition (if simulate is True)
        beta(array): the beta values (if simulate is True)
        omega(array): the omega values (if simulate is True)
        rng(generator): the random number generator
        nInits(int): the number of initial values to use

    Returns:
        fitData1Alpha(dataframe): the fit data for the same alpha model
        fitData2Alpha(dataframe): the fit data for the alpha difference model
    '''
    
    numSubjects = 75
    numTrials = 160

    if utility_function == multiplicative_utility:
  
      fitData1Alpha = pd.DataFrame(np.zeros((numSubjects, 5)),
                                  columns = ["ID",
                                            "alpha",
                                            "beta",
                                            "LL",
                                            "BIC"])

      fitData2Alpha = pd.DataFrame(np.zeros((numSubjects, 6)),
                                  columns = ["ID",
                                            "alphaStable",
                                            "alphaVolatile",
                                            "beta",
                                            "LL",
                                            "BIC"])
      
      paramsClip = {'alphaMin': 0, 'alphaMax': 1, 'betaMin': 0, 'betaMax': 1}
    
    elif utility_function == additive_utility:
      
      fitData1Alpha = pd.DataFrame(np.zeros((numSubjects, 6)),
                                  columns = ["ID",
                                            "alpha",
                                            "beta",
                                            "omega",
                                            "LL",
                                            "BIC"])

      fitData2Alpha = pd.DataFrame(np.zeros((numSubjects, 7)),
                                  columns = ["ID",
                                            "alphaStable",
                                            "alphaVolatile",
                                            "beta",
                                            "omega",
                                            "LL",
                                            "BIC"])
      
      paramsClip = {'alphaMin': 0, 'alphaMax': 1, 'betaMin': 0, 'betaMax': 1, 'omegaMin': 0, 'omegaMax': 1}

    # create matrices to store the data
    opt1RewardedStableMat = np.zeros((numSubjects, numTrials//2))
    magOpt1StableMat      = np.zeros((numSubjects, numTrials//2))
    magOpt2StableMat      = np.zeros((numSubjects, numTrials//2))
    choice1StableMat      = np.zeros((numSubjects, numTrials//2))

    opt1RewardedVolatileMat = np.zeros((numSubjects, numTrials//2))
    magOpt1VolatileMat      = np.zeros((numSubjects, numTrials//2))
    magOpt2VolatileMat      = np.zeros((numSubjects, numTrials//2))
    choice1VolatileMat      = np.zeros((numSubjects, numTrials//2))


    for s in range(numSubjects):
      # load in data
      trueProbability, choice1, magOpt1, magOpt2, opt1Rewarded = loading.load_blain(s)
      
      if simulate:
        # simulate an artificial participant
        if utility_function == multiplicative_utility:
          if s < 37:
            choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_S[s], beta[s], utility_function = utility_function, rng = rng)
            choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_V[s], beta[s], utility_function = utility_function, rng = rng)
          else:
            choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_V[s], beta[s], utility_function = utility_function, rng = rng)
            choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_S[s], beta[s], utility_function = utility_function, rng = rng)
        elif utility_function == additive_utility:
          if s < 37:
            choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_S[s], beta[s], omega[s], utility_function = utility_function, rng = rng)
            choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_V[s], beta[s], omega[s], utility_function = utility_function, rng = rng)
          else:
            choice1[0:80], _, _, _, _   = simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   alpha_V[s], beta[s], omega[s], utility_function = utility_function, rng = rng)
            choice1[80:160], _, _, _, _ = simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], alpha_S[s], beta[s], omega[s], utility_function = utility_function, rng = rng)


      if s < 37:
        opt1RewardedStableMat[s, :] = opt1Rewarded[0:80]
        magOpt1StableMat[s, :]      = magOpt1[0:80]
        magOpt2StableMat[s, :]      = magOpt2[0:80]
        choice1StableMat[s, :]      = choice1[0:80]

        opt1RewardedVolatileMat[s, :] = opt1Rewarded[80:160]
        magOpt1VolatileMat[s, :]      = magOpt1[80:160]
        magOpt2VolatileMat[s, :]      = magOpt2[80:160]
        choice1VolatileMat[s, :]      = choice1[80:160]
      else:
        opt1RewardedStableMat[s, :] = opt1Rewarded[80:160]
        magOpt1StableMat[s, :]      = magOpt1[80:160]
        magOpt2StableMat[s, :]      = magOpt2[80:160]
        choice1StableMat[s, :]      = choice1[80:160]

        opt1RewardedVolatileMat[s, :] = opt1Rewarded[0:80]
        magOpt1VolatileMat[s, :]      = magOpt1[0:80]
        magOpt2VolatileMat[s, :]      = magOpt2[0:80]
        choice1VolatileMat[s, :]      = choice1[0:80]
    
    data = (opt1RewardedStableMat, magOpt1StableMat, magOpt2StableMat, choice1StableMat, opt1RewardedVolatileMat, magOpt1VolatileMat, magOpt2VolatileMat, choice1VolatileMat)

    # fit the model assuming the same alpha for both conditions
    fit_model_vectorised_same_alpha = jax.vmap(partial(fit_with_multiple_initial_values, nInits = nInits, utility_function = utility_function, fit_model = fit_model_same_alpha, paramsClip = paramsClip))
    params_same_alpha, loss_same_alpha = fit_model_vectorised_same_alpha(data)

    # fit the model assuming the alpha is different for the two conditions. Here we also start at one optimisation run from inital values based on the same alpha model
    if utility_function == multiplicative_utility:
      starting_params_alpha_difference = {'alphaStable': params_same_alpha['alpha'], 'alphaVolatile': params_same_alpha['alpha'], 'beta': params_same_alpha['beta']}
    elif utility_function == additive_utility:
      starting_params_alpha_difference = {'alphaStable': params_same_alpha['alpha'], 'alphaVolatile': params_same_alpha['alpha'], 'beta': params_same_alpha['beta'], 'omega': params_same_alpha['omega']}
    fit_model_vectorised_alpha_difference = jax.vmap(partial(fit_with_multiple_initial_values, nInits = nInits, utility_function = utility_function, fit_model = fit_model_alpha_difference, paramsClip = paramsClip))
    params_alpha_difference, loss_alpha_difference = fit_model_vectorised_alpha_difference(data, startingParams = starting_params_alpha_difference)
    
    # convert jax arrays to numpy arrays for dataframe assignment
    alpha_values = np.array(params_same_alpha["alpha"]).flatten()
    beta_values = np.array(params_same_alpha["beta"]).flatten()
    alpha_stable_values = np.array(params_alpha_difference["alphaStable"]).flatten()
    alpha_volatile_values = np.array(params_alpha_difference["alphaVolatile"]).flatten()
    beta_diff_values = np.array(params_alpha_difference["beta"]).flatten()
    
    for s in range(numSubjects):
       fitData1Alpha.loc[s, "alpha"] = alpha_values[s]
       fitData1Alpha.loc[s, "beta"]  = beta_values[s]
       fitData1Alpha.loc[s, "LL"]    = -float(loss_same_alpha[s])
       fitData1Alpha.loc[s, "ID"]    = s
       
       fitData2Alpha.loc[s, "alphaStable"]   = alpha_stable_values[s]
       fitData2Alpha.loc[s, "alphaVolatile"] = alpha_volatile_values[s]
       fitData2Alpha.loc[s, "beta"]          = beta_diff_values[s]
       fitData2Alpha.loc[s, "LL"]            = -float(loss_alpha_difference[s])
       fitData2Alpha.loc[s, "ID"]            = s

       if utility_function == multiplicative_utility:
         fitData1Alpha.loc[s, "BIC"] = 2*np.log(numTrials) + 2*float(loss_same_alpha[s])
         fitData2Alpha.loc[s, "BIC"] = 3*np.log(numTrials) + 2*float(loss_alpha_difference[s])
       elif utility_function == additive_utility:
         omega_values = np.array(params_same_alpha["omega"]).flatten()
         omega_diff_values = np.array(params_alpha_difference["omega"]).flatten()
         fitData1Alpha.loc[s, "omega"] = omega_values[s]
         fitData2Alpha.loc[s, "omega"] = omega_diff_values[s]
         fitData1Alpha.loc[s, "BIC"] = 3*np.log(numTrials) + 2*float(loss_same_alpha[s])
         fitData2Alpha.loc[s, "BIC"] = 4*np.log(numTrials) + 2*float(loss_alpha_difference[s])
    
    return fitData1Alpha, fitData2Alpha

def simulate_RL_model(
    opt1Rewarded: npt.NDArray,
    magOpt1:      npt.NDArray,
    magOpt2:      npt.NDArray,
    alpha:        float,
    beta:         float,
    *omega:       float,
    startingProb = 0.5,
    utility_function = multiplicative_utility,
    rng = np.random.default_rng(12345)
    ):
  '''
  Returns how likely option 1 is rewarded on each trial, the probability of
  choosing option 1 on a trial, and a simulated choice for each trial

    Parameters:
        opt1rewarded(bool array): True if option 1 is rewarded on a trial, False
          if option 2 is rewarded on a trial.
        magOpt1(int array): reward points between 1 and 100 for option 1 on each
          trial
        magOpt2(int array): reward points between 1 and 100 for option 2 on each
           trial
        alpha(float): fixed learning rate, greater than 0, less than/equal to 1
        beta(float): fixed inverse temperature, greater than 0
        *omega(float, optional): omega used in additive utility
        startingProb(float): starting probability (defaults to 0.5).
        utility_function(function): what utility function to use to combine
          reward magnitude and probability. Defaults to multiplicative_utility
        choice_function(function): what choice function to use to decide
          between utility1 and utility2. Has free parameter beta. Defaults
          to softmax.

    Returns:
        probOpt1(float array): how likely option 1 is rewarded on each trial
          according to the RL model.
        choiceProb1(float array): the probability of choosing option 1 on each
          trial when combining infrmation about magnitude and probability
  '''

  # check that alpha has been set appropriately
  assert alpha > 0, 'Learning rate (alpha) must be greater than 0'
  assert alpha <= 1,'Learning rate (alpha) must be less than or equal to 1'

  # check that inverse temperateure has been set appropriately
  assert beta >= 0, 'beta must be greater or equal than 0'

  # check that startingProb has been set appropriately
  assert startingProb >= 0, 'startingProb must be greater or equal than 0'
  assert startingProb <= 1, 'startingProb must be less than or equal to 1'

  # calculate the number of trials
  nTrials = len(opt1Rewarded)

  # pre-create some vectors we're going to assign into
  probOpt1    = np.zeros(nTrials, dtype = float)
  delta       = np.zeros(nTrials, dtype = float)
  choiceProb1 = np.zeros(nTrials, dtype = float)
  utility1    = np.zeros(nTrials, dtype = float)
  utility2    = np.zeros(nTrials, dtype = float)

  # set the first trial's prediction to be equal to the starting probability
  probOpt1[0] = startingProb

  for t in range(nTrials-1):
        # calculate the utility of the two options. *omega would only be needed
        # if the utility function has >2 inputs, which is not the case for multiplicative
        # utility.
        utility1[t] = utility_function(magOpt1[t], probOpt1[t], *omega)
        utility2[t] = utility_function(magOpt2[t], (1 - probOpt1[t]), *omega)

        # get the probability of making choice 1
        choiceProb1[t] = logistic.cdf((utility1[t]-utility2[t]) * beta)

        # calculate the prediction error
        delta[t] = opt1Rewarded[t] - probOpt1[t]

        # update the probability of option 1 being rewarded
        probOpt1[t+1] = probOpt1[t] + alpha * delta[t]
  
  t = nTrials-1
  utility1[t] = utility_function(magOpt1[t], probOpt1[t], *omega)
  utility2[t] = utility_function(magOpt2[t], (1 - probOpt1[t]), *omega)
  choiceProb1[t] = logistic.cdf((utility1[t]-utility2[t]) * beta)
        
  choice1 = (choiceProb1 > rng.random(len(opt1Rewarded))).astype(int)

  return choice1, probOpt1, choiceProb1, utility1, utility2


def estimate_model_frequencies(BICs, alpha0=None):
    # Based on https://github.com/spm/spm12/blob/main/spm_BMS.m and
    # Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
    # Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017

    Ni, Nk     = BICs.shape
    c          = 1.
    cc         = 1e-3
    log_u      = np.zeros([Ni, Nk])
    u          = np.zeros([Ni, Nk])
    g          = np.zeros([Ni, Nk])
    beta       = np.zeros(Nk)
    prev_alpha = np.zeros(Nk)

    if alpha0 is None:
        alpha0 = np.ones(Nk)
    alpha = np.array(alpha0)

    # iterative VB estimation
    while c > cc:

        # compute posterior belief g(i,k)=q(m_i=k|y_i) that model k generated
        # the data for the i-th subject
        for i in range(Ni):
            for k in range(Nk):
                # integrate out prior probabilities of models (in log space)
                log_u[i, k] = -BICs[i,k] + digamma(alpha[k]) - digamma(np.sum(alpha))

            # exponentiate (to get back to non-log representation)
            u[i,:] = np.exp(log_u[i,:])

            # normalisation: sum across all models for i-th subject
            u_i    = np.sum(u[i,:])
            g[i,:] = u[i,:]/u_i

        # expected number of subjects whose data we believe to have been 
        # generated by model k
        for k in range(Nk):
            beta[k] = np.sum(g[:,k])

        # update alpha
        prev_alpha[:] = alpha
        for k in range(Nk):
            alpha[k] = alpha0[k] + beta[k]

        # convergence?
        c = np.sum(np.abs(alpha - prev_alpha))
    
    # compute the standard deviation of the model frequencies
    sd = np.sqrt(alpha * (alpha.sum() - alpha) / (alpha.sum()**2 * (alpha.sum() + 1)))

    return alpha, sd

def estimate_exceedance_probabilities(alpha, Nsamp=int(1e6)):
    # Based on https://github.com/spm/spm12/blob/main/spm_dirichlet_exceedance.m and
    # Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
    # Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017

    Nk = len(alpha)
    xp = np.zeros(Nk)

    for i in range(Nsamp):
        # Sample from univariate gamma densities then normalise
        # (see Dirichlet entry in Wikipedia or Ferguson (1973) Ann. Stat. 1,
        # 209-230)
        r = np.zeros(Nk)
        for k in range(Nk):
            r[k] = np.random.gamma(alpha[k], 1)

        sr = np.sum(r)
        for k in range(Nk):
            r[k] = r[k]/sr

        # Exceedance probabilities:
        # For any given model k1, compute the probability that it is more
        # likely than any other model k2~=k1
        xp[np.argmax(r)] += 1

    return xp/sum(xp)