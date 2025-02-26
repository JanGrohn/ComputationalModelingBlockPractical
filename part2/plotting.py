# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np
import jax.numpy as jnp

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# we use plotly to make our figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
from tqdm.auto import tqdm

# set the style of the plotly figures
pio.templates.default = "none"

# this allows us to make interactive figures
import ipywidgets as widgets

# import some custom fitting functions we wrote
from part2 import fitting

# Add type hint imports
from typing import Optional, Callable, Union, Dict, Any
import numpy.typing as npt
from plotly.graph_objs._figure import Figure
from ipywidgets import VBox

def plot_schedule(
    opt1Rewarded:    npt.NDArray,
    trueProbability: npt.NDArray,
    magOpt1:         Optional[npt.NDArray] = None,
    magOpt2:         Optional[npt.NDArray] = None,
    probOpt1:        Optional[npt.NDArray] = None,
    choiceProb1:     Optional[npt.NDArray] = None,
    choice1:         Optional[npt.NDArray] = None,
    utility1:        Optional[npt.NDArray] = None,
    utility2:        Optional[npt.NDArray] = None,
    ) -> Figure:
  '''
  Plots the experimental schedule and the RL model estimate using plotly.

    Parameters:
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if
          option 2 is rewarded on a trial.
        trueProbability(float array): The probability with which option 1 is
          rewareded on each trial.
        magOpt1(int array): The reward magnitude of option 1. Defaults to None,
          which excludes it from the plot.
        magOpt2(int array): The reward magnitude of option 1. Defaults to None,
          which excludes it from the plot.
        probOpt1(float array): how likely option 1 is rewarded on each trial
          according to the RL model. Defaults to None, which excludes it
          from the plot.
        choiceProb1(float array): the probability of choosing option 1 on each
          trial according to the RL model. Defaults to None, which excludes it
          from the plot.
        choice1(int array): whether option 1 (1) or option 2 (2) was chosen on
          each. Defaults to None, which excludes it from the plot.
        utility1(float array): the utility of option 1 on each trial according to
          the RL model. Defaults to None, which excludes it from the plot.
        utility2(float array): the utility of option 2 on each trial according to
          the RL model. Defaults to None, which excludes it from the plot.

  '''
  # compute number of trials
  nTrials = len(opt1Rewarded)

  # create 2 subplots if needed
  if magOpt1 is not None:
    fig = make_subplots(rows=2, cols=1)
  else:
    fig = go.Figure()

  # plot opt1rewarded as a scatterplot
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = opt1Rewarded,
          mode = 'markers',
          marker = dict(size = 5),
          name = "trial outcomes"
      ))

  # plot trueProbability as a line
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = trueProbability,
          mode = 'lines',
          line = dict(color='black', dash='dash'),
          name = "true probability"
      ))

  # check if we should plot probOpt1, and if so plot it as a line
  if probOpt1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = probOpt1,
            mode = 'lines',
            line = dict(color='red'),
            name = "RL model probability"
        ))

  # check if we should plot choiceProb1, and if so plot it as a scatterplot
  if choiceProb1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = choiceProb1,
            mode = 'markers',
            marker = dict(size = 5, symbol='x'),
            name = "choice probability"
        ))

  # check if we should plot choice1, and if so plot it as a scatterplot
  if choice1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = choice1,
            mode = 'markers',
            marker = dict(size = 5, symbol='cross', opacity=0.8),
            name = "simulated choice"
        ))

  # label the axes
  fig.update_layout(xaxis_title="", yaxis_title="green rewarded?")

  # check if we should plot reward magnitudes, and if so plot them as lines
  if magOpt1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = magOpt1,
            mode = 'lines',
            line = dict(color='orange'),
            name = "magnitude option 1",
            xaxis = 'x2',
            yaxis = 'y2',
        ))
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = magOpt2,
            mode = 'lines',
            line = dict(color='green'),
            name = "magnitude option 2",
            xaxis = 'x2',
            yaxis = 'y2',
        ))

    if utility1 is not None:
            
      # plot the utility of option 1 and option 2
      fig.add_trace(
          go.Scatter(
              x = list(range(nTrials)),
              y = utility1,
              mode = 'lines',
              line = dict(color='magenta'),
              name = "utility option 1",
              xaxis = 'x2',
              yaxis = 'y2',
          ))
      fig.add_trace(
          go.Scatter(
              x = list(range(nTrials)),
              y = utility2,
              mode = 'lines',
              line = dict(color='purple'),
              name = "utility option 2",
              xaxis = 'x2',
              yaxis = 'y2',
          ))
    # label the axes
    fig.update_layout(xaxis2_title="trial number", yaxis2_title="reward magnitude")

  # set x-axis range for all subplots
  # this sets the range for all x-axes
  fig.update_xaxes(range=[0, nTrials])
  # this is needed to set the range for the second subplot if it exists
  if magOpt1 is not None:
    fig.update_xaxes(range=[0, nTrials], row=2, col=1)

  return fig

def visualise_utility_function(
    utility_function: Callable,
    omega:            bool = False,
    nSamples:         int = 100
    ) -> None:
  '''
  Visualises a utility function using plotly.

    Parameters:
        utility_function(function): The utility function to visualise.
        omega(bool): Whether the utility function has an omega parameter.
        nSamples(int): The number of samples to take from the utility function
          to visualise it.
  '''

  # get a slider for omega
  omegaSlider = widgets.FloatSlider(
                            value=0.5,
                            max=1,
                            min=0,
                            step=0.01,
                            description='omega:',
                            continuous_update=False)

  # the range of mag and prob values for which we want to visualise the function
  magRange  = np.linspace(1, 100, nSamples)
  probRange = np.linspace(0, 1, nSamples)

  # function to fill in the utility matrix
  def compute_utilityMatrix(
      utility_function: Callable = utility_function,
      show_omega:       bool = omega,
      nSamples:         int = nSamples,
      omega:            float = omegaSlider.value
      ) -> npt.NDArray:

    # empty matrix to fill in with utility values
    utilityMatrix = np.zeros((nSamples,nSamples))

    # loop through all possible values of mag and prob and get their utility
    for mag in range(nSamples):
      for prob in range(nSamples):
        if show_omega:
          utilityMatrix[mag,prob] = utility_function(magRange[mag], probRange[prob], omega)
        else:
          utilityMatrix[mag,prob] = utility_function(magRange[mag], probRange[prob])
    
    return utilityMatrix
  
  utilityMatrix = compute_utilityMatrix(utility_function)

  # make a 3d plot of the utility function
  fig = go.Figure(go.Surface(z = utilityMatrix,
                            y = magRange,
                            x = probRange))

  fig.update_scenes(yaxis_title='magnitude',
                    xaxis_title='probability',
                    zaxis_title='utility')

  fig.update_layout(scene_camera=dict(eye=dict(x=-2, y=-2, z=2)),
                    autosize=False,
                    width=500,
                    height=500,
                    margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=4
                    ))

  # if omega is true, add a slider to change omega
  if omega:
    # make the figure interactive
    fig = go.FigureWidget(fig)

    # function that triggers when the omega value changes
    def change_omega(change: Dict[str, Any]) -> None:
      # calculate the utility matrix for the new omega value
      utilityMatrix = compute_utilityMatrix(omega = omegaSlider.value)
      # update the figure
      with fig.batch_update():
        fig.data[0].z = utilityMatrix
    
    # listen to changes of the omega slider
    omegaSlider.observe(change_omega, names="value")

    # show the slider and figure
    display(widgets.VBox([omegaSlider, fig]))

  else:
     display(fig)

 
def visualise_softmax(softmax: Callable) -> None:
  '''
  Visualises a softmax function using plotly.
  
      Parameters:
          softmax(function): The softmax function to visualise.
    '''

  # define a slider for the inverse temperature
  betaSlider = widgets.FloatSlider(
                              value=3,
                              max=50,
                              min=0,
                              step=0.01,
                              description='beta:',
                              continuous_update=False)

  # the initial beta to display
  beta = 3
  utilityRange = np.linspace(-2, 2, 1000)

  # start a new figure
  fig = go.FigureWidget(go.Scatter(
            x = utilityRange,
            y = softmax(utilityRange, 0, beta),
            mode = 'lines',
            line = dict(color='black')
        ))


  # label the axes
  fig.update_layout(xaxis_title="utility 1 - utility 2", yaxis_title="probability of choosing option 1")

  # set the range for the y-axis
  fig.update_yaxes(range=[0, 1])

  # function that triggers when the beta value changes
  def change_beta(change: Dict[str, Any]) -> None:
    # get the current value of beta
    beta = betaSlider.value
    # calculate the corresponding utilities
    u = softmax(utilityRange, 0, beta)
    # update the figure
    with fig.batch_update():
      fig.data[0].y = u

  # listen to changes of the beta slider
  betaSlider.observe(change_beta, names="value")

  # show the slider and figure
  display(widgets.VBox([betaSlider, fig]))

def plot_interactive_RL_model(
    simulate_RL_model: Callable,
    utility_function:  Callable,
    opt1Rewarded:      npt.NDArray,
    magOpt1:           npt.NDArray,
    magOpt2:           npt.NDArray,
    trueProbability:   npt.NDArray,
    omega:             bool = False
    ) -> None:
  '''
  Plots the experimental schedule and the RL model estimate using plotly.

    Parameters:
        simulate_RL_model(function): The RL model to use to simulate the data.
        utiity_function(function): The utility function to use in the RL model.
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if
          option 2 is rewarded on a trial.
        magOpt1(int array): The reward magnitude of option 1.
        magOpt2(int array): The reward magnitude of option 1.
        trueProbability(float array): The probability with which option 1 is
          rewareded on each trial.
  '''

  # make sliders for alpha and beta
  alphaSlider = widgets.FloatSlider(
                                value=0.1,
                                max=1,
                                min=0.001,
                                step=0.01,
                                description='alpha:',
                                continuous_update=False
                                )

  betaSlider = widgets.FloatSlider(
                                value=0.02,
                                max=1,
                                min=0,
                                step=0.01,
                                description='beta:',
                                continuous_update=False
                                )
  
  omegaSlider = widgets.FloatSlider(
                            value=0.5,
                            max=1,
                            min=0,
                            step=0.01,
                            description='omega:',
                            continuous_update=False)

  if omega:
    sliders = widgets.VBox(children=[
                                  alphaSlider,
                                  betaSlider,
                                  omegaSlider])
    # run the RL model
    probOpt1, choiceProb1 = simulate_RL_model(opt1Rewarded, magOpt1, magOpt2, alphaSlider.value, betaSlider.value, omegaSlider.value, utility_function = utility_function)
    
    # calcualte utility
    utility1 = utility_function(magOpt1, probOpt1, omegaSlider.value)
    utility2 = utility_function(magOpt2, 1 - probOpt1, omegaSlider.value)

  else:
    sliders = widgets.VBox(children=[
                                    alphaSlider,
                                    betaSlider])
      
    # run the RL model
    probOpt1, choiceProb1 = simulate_RL_model(opt1Rewarded, magOpt1, magOpt2, alphaSlider.value, betaSlider.value, utility_function = utility_function)

    # calcualte utility
    utility1 = utility_function(magOpt1, probOpt1)
    utility2 = utility_function(magOpt2, 1 - probOpt1)
    
  # call the figure function we wrote and make it interactive
  fig = go.FigureWidget(plot_schedule(opt1Rewarded, trueProbability, magOpt1, magOpt2, probOpt1, choiceProb1, None, utility1, utility2))

  fig.update_layout(
                  autosize=True,
                  height=500)
  
  # function to run if alpha or beta have changed
  def change_model(change: Dict[str, Any]) -> None:
    # rerun the RL model
    if omega:
      probOpt1, choiceProb1 = simulate_RL_model(opt1Rewarded, magOpt1, magOpt2, alphaSlider.value, betaSlider.value, omegaSlider.value, utility_function = utility_function)
      utility1 = utility_function(magOpt1, probOpt1, omegaSlider.value)
      utility2 = utility_function(magOpt2, 1 - probOpt1, omegaSlider.value)
    else:
      probOpt1, choiceProb1 = simulate_RL_model(opt1Rewarded, magOpt1, magOpt2, alphaSlider.value, betaSlider.value, utility_function = utility_function)
      utility1 = utility_function(magOpt1, probOpt1)
      utility2 = utility_function(magOpt2, 1 - probOpt1)
    
    # update the figure
    with fig.batch_update():
      fig.data[2].y = probOpt1
      fig.data[3].y = choiceProb1
      fig.data[6].y = utility1
      fig.data[7].y = utility2

  # run the function if a slider value changes
  alphaSlider.observe(change_model, names="value")
  betaSlider.observe(change_model, names="value")
  if omega:
    omegaSlider.observe(change_model, names="value")

  # show the figure and the sliders
  display(widgets.VBox([sliders, fig]))

def plot_likelihood_landscapes(
    opt1Rewarded: npt.NDArray,
    magOpt1:      npt.NDArray,
    magOpt2:      npt.NDArray,
    choice1:      npt.NDArray,
    ) -> None:
  '''
  Plots the likelihood landscape for alpha and beta using plotly.

    Parameters:
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if
          option 2 is rewarded on a trial.
        magOpt1(int array): The reward magnitude of option 1.
        magOpt2(int array): The reward magnitude of option 1.
        choice1(int array): whether option 1 (1) or option 2 (2) was chosen on
          each.
        loglikelihood_RL_model(function): The loglikelihood function to use.
  '''

  # ensure the inptus are jnp arrays
  opt1Rewarded = jnp.array(opt1Rewarded)
  magOpt1 = jnp.array(magOpt1)
  magOpt2 = jnp.array(magOpt2)
  choice1 = jnp.array(choice1)
  
  # the values of alpha and beta to plug into the likelihood function
  alphaRange = np.arange(0.01, 1, 0.005)
  betaRange  = np.arange(0.01, 1, 0.005)

  # matrix to store the log likelihoods for each value of alpha and beta we try
  LLMatrix = np.zeros((len(alphaRange),len(betaRange)))

  # loop through alpha and beta and get the corresponding log likelihoods
  for a in tqdm(range(len(alphaRange))):
    for b in range(len(betaRange)):
      loss = fitting.loss_RL_model((opt1Rewarded, magOpt1, magOpt2, choice1), {'alpha': jnp.array([alphaRange[a]]), 'beta': jnp.array([betaRange[b]])})
      # average cross entropy loss needs to be negated to get the log likelihood and multiplied by the number of trials
      LLMatrix[a,b] = -loss*len(opt1Rewarded)

  # also calculate the normalised likelihood
  LMatrix = np.exp(LLMatrix)/sum(sum(np.exp(LLMatrix)))

  # visualise the likelihood for each parameter pair
  fig = make_subplots(rows=1, cols=2,
                      specs=[[{'is_3d': True}, {'is_3d': True}]],
                      subplot_titles=['normalised likelihood', 'log likelihood'],
                      )

  # this plots the normalised likelihood
  fig.add_trace(go.Surface(z = LMatrix,
                           y = alphaRange,
                           x = betaRange,
                           colorbar_x = -0.07), 1, 1)

  # this plots the log likelihood
  fig.add_trace(go.Surface(z = LLMatrix,
                           y = alphaRange,
                           x = betaRange), 1, 2)

  fig.update_scenes(yaxis_title='alpha',
                    xaxis_title='beta',
                    zaxis_title='')

  fig.show()



def plot_recovered_parameters(recoveryData: pd.DataFrame) -> None:
  '''
  Plots the simulated against the recoverd parameters

    Parameters:
        recoveryData(DataFrame): DataFrame with columns simulatedAlpha,
          simulatedBeta, recoverdAlpha, recoverdBeta
  '''

  fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("alpha","inverse temperature"))

  # Add line traces
  fig.add_trace(go.Scatter(x=[0, 1.1], y=[0, 1.1], mode='lines', line=dict(color='black', width=1)), row=1, col=1)
  fig.add_trace(go.Scatter(x=[0, 0.7], y=[0, 0.7], mode='lines', line=dict(color='black', width=1)), row=1, col=2)

  # Add scatter traces
  fig.add_trace(go.Scatter(
                      x=recoveryData["simulatedAlpha"],
                      y=recoveryData["recoveredAlpha"],
                      mode='markers',
                      marker=dict(color="blue"),
                      hovertemplate = 'Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<extra></extra>',
                      customdata = np.stack((round(recoveryData["simulatedBeta"],3), round(recoveryData["recoveredBeta"],3)), axis=-1)),
            row=1, col=1)

  fig.add_trace(go.Scatter(
                      x=recoveryData["simulatedBeta"],
                      y=recoveryData["recoveredBeta"],
                      mode='markers',
                      marker=dict(color='blue'),
                      hovertemplate = '<br>Simulated Alpha: %{customdata[0]}<br>Recovered Alpha: %{customdata[1]}<extra></extra>',
                      customdata = np.stack((round(recoveryData["simulatedAlpha"],3), round(recoveryData["recoveredAlpha"],3)), axis=-1)),
            row=1, col=2)


  # Setting the ticks and range on x and y axes
  fig.update_xaxes(tickvals=list(np.arange(0,1.2,0.2)), range=[0,0.7], row=1, col=1)
  fig.update_yaxes(tickvals=list(np.arange(0,1.2,0.2)), range=[0,1.1], row=1, col=1)
  fig.update_xaxes(tickvals=list(np.arange(0,0.8,0.1)), range=[0,0.4], row=1, col=2)
  fig.update_yaxes(tickvals=list(np.arange(0,0.8,0.1)), range=[0,0.7], row=1, col=2)

  fig.update_layout(xaxis1_title="simulated", yaxis1_title="recovered", showlegend=False)
  fig.update_layout(xaxis2_title="simulated", yaxis2_title="recovered", showlegend=False)

  # Show the figure
  fig.show()

def visualise_alpha_difference(
    stableAlphas:   npt.NDArray,
    volatileAlphas: npt.NDArray,
    title:          str
    ) -> None:
  fig = px.histogram(pd.DataFrame({"stable block": stableAlphas,
                                  "volatile block": volatileAlphas}).melt(),
                                  color="variable", x="value", marginal="box", barmode="overlay")

  fig.update_layout(title= title, xaxis_title="learning rate", legend_title_text="")
  fig.update_xaxes(range=[0, .7])

  fig.show()