# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# we use the minimize function from scipy to fit models
from scipy.optimize import minimize

# import the logistic function
from scipy.stats import logistic

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from functools import partial

@jax.jit
def utility_fun(mag, prob):
  return mag*prob

@partial(jax.jit,static_argnames=['utility_function'])
def loss_RL_model(data,
                  params,
                  startingProb = 0.5,
                  utility_function = utility_fun):
    
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
    def compute_probOpt1(carry, t):
        probOpt1 = carry
        delta = opt1Rewarded[t] - probOpt1
        probOpt1 = probOpt1 + alpha * delta
        return probOpt1, probOpt1

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
    utility1 = utility_function(magOpt1, probOpt1)
    utility2 = utility_function(magOpt2, 1-probOpt1)
    
    # Compute the loss of the model
    choice1Logits = beta * (utility1 - utility2)
    loss = optax.sigmoid_binary_cross_entropy(choice1Logits, choice1).mean()
    
    return loss

def run_opt(init_params, fun, opt, max_iter, tol, paramsClip):
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def project_params(params, paramsClip):
    def clip_param(param, name):
      if 'alpha' in name:
        return jnp.clip(param, otu.tree_get(paramsClip, 'alphaMin'), otu.tree_get(paramsClip, 'alphaMax'))
      elif 'beta' in name:
        return jnp.clip(param, otu.tree_get(paramsClip, 'betaMin'), otu.tree_get(paramsClip, 'betaMax'))
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

@partial(jax.jit,static_argnames=['utility_function', 'opt'])
def fit_model_same_alpha(data, rng = None, startingProb = 0.5, paramsClip = {'alphaMin': 0, 'alphaMax': 1, 'betaMin': 0, 'betaMax': 1}, utility_function = utility_fun, opt = optax.lbfgs(), max_iter = 100, tol = 1e-3):
    
    if rng is None:
      rng = jax.random.PRNGKey(0)
    rng, alphaRng, betaRng = jax.random.split(rng, 3)
    alpha = jax.random.uniform(alphaRng, shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
    beta  = jax.random.uniform(betaRng,  shape=(1,), minval=paramsClip['betaMin'],  maxval=paramsClip['betaMax'] )
    
    initParams = {'alpha': alpha, 'beta': beta}
    
    def loss_fun(params): 
      loss = loss_RL_model(data, params, startingProb = startingProb, utility_function = utility_function)
      return loss
    
    finalParams, finalState = run_opt(initParams, loss_fun, opt, max_iter=max_iter, tol=tol, paramsClip=paramsClip)
    finalLoss = loss_fun(finalParams)
    return finalParams, finalState, finalLoss

@partial(jax.jit,static_argnames=['utility_function', 'opt'])
def fit_model_alpha_difference(data, rng = None, startingProb = 0.5, paramsClip = {'alphaMin': 0, 'alphaMax': 1, 'betaMin': 0, 'betaMax': 1}, utility_function = utility_fun, opt = optax.lbfgs(), max_iter = 100, tol = 1e-3):
    opt1RewardedStable, magOpt1Stable, magOpt2Stable, choice1Stable, opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile = data
    dataStable = (opt1RewardedStable, magOpt1Stable, magOpt2Stable, choice1Stable)
    dataVolatile = (opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile)

    if rng is None:
      rng = jax.random.PRNGKey(0)
    rng, alphaStableRng, alphaVolatileRng, betaRng = jax.random.split(rng, 4)
    alphaStable   = jax.random.uniform(alphaStableRng,   shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
    alphaVolatile = jax.random.uniform(alphaVolatileRng, shape=(1,), minval=paramsClip['alphaMin'], maxval=paramsClip['alphaMax'])
    beta          = jax.random.uniform(betaRng,          shape=(1,), minval=paramsClip['betaMin'],  maxval=paramsClip['betaMax'] )
    
    initParams = {'alphaStable': alphaStable, 'alphaVolatile': alphaVolatile, 'beta': beta}
    
    def loss_fun(params):
      stableParams   = {'alpha': params['alphaStable'],   'beta': params['beta']}
      volatileParams = {'alpha': params['alphaVolatile'], 'beta': params['beta']}
      lossStable = loss_RL_model(dataStable, stableParams, startingProb = startingProb, utility_function = utility_function)
      lossVolatile = loss_RL_model(dataVolatile, volatileParams, startingProb = startingProb, utility_function = utility_function)
      return lossStable + lossVolatile
    
    finalParams, finalState = run_opt(initParams, loss_fun, opt, max_iter=max_iter, tol=tol, paramsClip=paramsClip)
    finalLoss = loss_fun(finalParams)
    return finalParams, finalState, finalLoss

@partial(jax.jit, static_argnames=['nInits', 'fit_model'])
def fit_with_multiple_initial_values(data, nInits = 100, fit_model = fit_model_same_alpha):
    if fit_model == fit_model_same_alpha:
      opt1Rewarded, magOpt1, magOpt2, choice1 = data
      opt1RewardedStack  = jnp.tile(opt1Rewarded, (nInits, 1))
      magOpt1Stack       = jnp.tile(magOpt1,      (nInits, 1))
      magOpt2Stack       = jnp.tile(magOpt2,      (nInits, 1))
      choice1Stack       = jnp.tile(choice1,      (nInits, 1))
      data = (opt1RewardedStack, magOpt1Stack, magOpt2Stack, choice1Stack)
    elif fit_model == fit_model_alpha_difference:
      opt1RewardedStable, magOpt1Stable, magOpt2Stable, choice1Stable, opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile = data
      opt1RewardedStableStack   = jnp.tile(opt1RewardedStable,   (nInits, 1))
      magOpt1StableStack        = jnp.tile(magOpt1Stable,        (nInits, 1))
      magOpt2StableStack        = jnp.tile(magOpt2Stable,        (nInits, 1))
      choice1StableStack        = jnp.tile(choice1Stable,        (nInits, 1))
      opt1RewardedVolatileStack = jnp.tile(opt1RewardedVolatile, (nInits, 1))
      magOpt1VolatileStack      = jnp.tile(magOpt1Volatile,      (nInits, 1))
      magOpt2VolatileStack      = jnp.tile(magOpt2Volatile,      (nInits, 1))
      choice1VolatileStack      = jnp.tile(choice1Volatile,      (nInits, 1))
      data = (opt1RewardedStableStack, magOpt1StableStack, magOpt2StableStack, choice1StableStack, opt1RewardedVolatileStack, magOpt1VolatileStack, magOpt2VolatileStack, choice1VolatileStack)
    else :
        raise ValueError("fit_model must be either fit_model_same_alpha or fit_model_alpha_difference")
    
    fit_model_vectorised = jax.vmap(fit_model)
    
    
    
    initial_rngs = jnp.stack([jax.random.PRNGKey(i) for i in range(nInits)], axis=0)
    
    params, _, loss = fit_model_vectorised(data, initial_rngs)
    
    def get_params(params, loss):
        bestFit = jnp.argmin(loss)
        if fit_model == fit_model_same_alpha:
          bestAlpha = params['alpha'][bestFit]
          bestBeta  = params['beta'][bestFit]
          bestParams = {'alpha': bestAlpha, 'beta': bestBeta}
        elif fit_model == fit_model_alpha_difference:
          bestAlphaStable   = params['alphaStable'][bestFit]
          bestAlphaVolatile = params['alphaVolatile'][bestFit]
          bestBeta          = params['beta'][bestFit]
          bestParams = {'alphaStable': bestAlphaStable, 'alphaVolatile': bestAlphaVolatile, 'beta': bestBeta}
        return bestParams

    return get_params(params, loss)

def fit_multiple_participants(data, nInits = 100, fit_model = fit_model_same_alpha):

  fit_model_vectorised = jax.vmap(partial(fit_with_multiple_initial_values, nInits=nInits, fit_model = fit_model))
  params = fit_model_vectorised(data)
  
  return params

def run_paramterer_recovery(
                        simulatedAlphaRange, 
                        simulatedBetaRange,
                        simulate_RL_model,
                        generate_schedule,
                        trueProbability,
                        rng,
                        ):
    '''
    This function simulates participants with different learning rates and then fits the data . This allows us to see if the model can recover the true learning rates.
    
    Parameters:
        simulatedAlphaRange(float array): alpha values to simulate
        simulatedBetaRange(float array): beta values to simulate
        simulate_RL_model(function): the function we use to simulate the RL model
        generate_schedule(function): the function we use to generate a schedule
        trueProbability(float array): the reward probabilities
        rng(random number generator): the random number generator we use

    Returns:
        recoveryData(dataframe): a dataframe containing the simulated and recovered parameters
    '''

    nSimulatedSubjects = len(simulatedAlphaRange) * len(simulatedBetaRange)

    # initialise a table to store the simualted and recoverd parameters in
    recoveryData = pd.DataFrame(np.zeros((nSimulatedSubjects, 4)),
                                columns = ["simulatedAlpha",
                                           "simulatedBeta",
                                           "recoveredAlpha",
                                           "recoveredBeta"])

    # initialize the iteration counter
    counter = 0

    opt1RewardedMat = np.zeros((nSimulatedSubjects, len(trueProbability)))
    magOpt1Mat = np.zeros((nSimulatedSubjects, len(trueProbability)))
    magOpt2Mat = np.zeros((nSimulatedSubjects, len(trueProbability)))
    choice1Mat = np.zeros((nSimulatedSubjects, len(trueProbability)))

    for alpha in range(len(simulatedAlphaRange)):
        for beta in range(len(simulatedBetaRange)):

            # generate a new schedule
            opt1RewardedMat[counter,:], magOpt1Mat[counter,:], magOpt2Mat[counter,:] = generate_schedule(trueProbability)

            # simulate an artificial participant
            probOpt1, choiceProb1 = simulate_RL_model(opt1RewardedMat[counter,:], magOpt1Mat[counter,:], magOpt2Mat[counter,:], simulatedAlphaRange[alpha], simulatedBetaRange[beta])
            choice1Mat[counter,:] = (choiceProb1 > rng.random(len(trueProbability))).astype(int)

            # recoverdAlpha, recoveredBeta = fit_with_multiple_initial_values(opt1Rewarded, magOpt1, magOpt2, choice1, nInits = 100)

            # save the data of the current iteration
            recoveryData.loc[counter,"simulatedAlpha"] = simulatedAlphaRange[alpha]
            recoveryData.loc[counter,"simulatedBeta"]  = simulatedBetaRange[beta]
            # recoveryData.loc[counter,"recoveredAlpha"] = recoverdAlpha
            # recoveryData.loc[counter,"recoveredBeta"]  = recoveredBeta

            # increase the iteration counter
            counter += 1
    params = fit_multiple_participants((opt1RewardedMat, magOpt1Mat, magOpt2Mat, choice1Mat))
    recoveryData["recoveredAlpha"] = params["alpha"]
    recoveryData["recoveredBeta"]  = params["beta"]
    return recoveryData


def run_paramterer_recovery_with_difference(
                stableAlphas,           
                volatileAlphas,         
                betas,                  
                simulate_RL_model,      
                generate_schedule,       
                trueProbabilityStable,   
                trueProbabilityVolatile, 
                rng,
                ):
    ''' 
    This function simulates participants with different learning rates in the stable and volatile conditions, and then fits the data with a model that assumes the same learning rate in both conditions. This allows us to see if the model can recover the true learning rates in the stable and volatile conditions.
    
    Parameters:
        stableAlphas(float array): alpha values in the stable condition
        volatileAlphas(float array): alpha values in the volatile condition
        betas(float array): beta values in both conditions
        simulate_RL_model(function): the function we use to simulate the RL model
        generate_schedule(function): the function we use to generate a schedule
        trueProbabilityStable(float array): the reward probabilities in the stable condition
        trueProbabilityVolatile(float array): the reward probabilities in the volatile condition
        rng(random number generator): the random number generator we use

    Returns:
        fittedParameters(dataframe): a dataframe containing the recovered parameters
    '''

    nSimulatedSubjects = len(betas)

    # initialise a table to store the simualted and recoverd parameters in
    fittedParameters = pd.DataFrame(np.zeros((nSimulatedSubjects, 3)),
                                        columns = ["alpha stable",
                                                   "alpha volatile",
                                                   "inverse temperature"])

    opt1RewardedStableMat    = np.zeros((nSimulatedSubjects, len(trueProbabilityStable)))
    magOpt1StableMat         = np.zeros((nSimulatedSubjects, len(trueProbabilityStable)))
    magOpt2StableMat         = np.zeros((nSimulatedSubjects, len(trueProbabilityStable)))
    choice1StableMat         = np.zeros((nSimulatedSubjects, len(trueProbabilityStable)))
    opt1RewardedVolatileMat  = np.zeros((nSimulatedSubjects, len(trueProbabilityStable)))
    magOpt1VolatileMat       = np.zeros((nSimulatedSubjects, len(trueProbabilityStable)))
    magOpt2VolatileMat       = np.zeros((nSimulatedSubjects, len(trueProbabilityStable)))
    choice1VolatileMat       = np.zeros((nSimulatedSubjects, len(trueProbabilityStable)))

    for p in range(nSimulatedSubjects):
        # generate a new schedule
        opt1RewardedStableMat[p,:],   magOpt1StableMat[p,:],   magOpt2StableMat[p,:]   = generate_schedule(trueProbabilityStable)
        opt1RewardedVolatileMat[p,:], magOpt1VolatileMat[p,:], magOpt2VolatileMat[p,:] = generate_schedule(trueProbabilityVolatile)

        # simulate an artificial participant
        probOpt1Stable,   choiceProb1Stable   = simulate_RL_model(opt1RewardedStableMat[p,:],   magOpt1StableMat[p,:],   magOpt2StableMat[p,:],   stableAlphas[p],   betas[p])
        probOpt1Volatile, choiceProb1Volatile = simulate_RL_model(opt1RewardedVolatileMat[p,:], magOpt1VolatileMat[p,:], magOpt2VolatileMat[p,:], volatileAlphas[p], betas[p])
        
        choice1StableMat[p,:]   = (choiceProb1Stable   > rng.random(len(opt1RewardedStableMat[p,:]))).astype(int)
        choice1VolatileMat[p,:] = (choiceProb1Volatile > rng.random(len(opt1RewardedVolatileMat[p,:]))).astype(int)

        # create function to be minimized
        # def min_fun(x):
        #     LL1 = loglikelihood_RL_model(opt1RewardedStable, magOpt1Stable, magOpt2Stable, choice1Stable, logistic.cdf(x[0]), np.exp(x[2]))
        #     LL2 = loglikelihood_RL_model(opt1RewardedVolatile, magOpt1Volatile, magOpt2Volatile, choice1Volatile, logistic.cdf(x[1]), np.exp(x[2]))
        #     return -(LL1 + LL2)

        # # fit the data of this simulated participant
        # pars = minimize(min_fun, [0, 0, -1.5], method = method)

        # fittedParameters.loc[p,"alpha stable"] = logistic.cdf(pars.x[0])
        # fittedParameters.loc[p,"alpha volatile"] = logistic.cdf(pars.x[1])
        # fittedParameters.loc[p,"inverse temperature"] = np.exp(pars.x[2])

    params = fit_multiple_participants((opt1RewardedStableMat, magOpt1StableMat, magOpt2StableMat, choice1StableMat, opt1RewardedVolatileMat, magOpt1VolatileMat, magOpt2VolatileMat, choice1VolatileMat), fit_model = fit_model_alpha_difference)
    fittedParameters["alpha stable"] = params["alphaStable"]
    fittedParameters["alpha volatile"] = params["alphaVolatile"]
    fittedParameters["inverse temperature"] = params["beta"]
    return fittedParameters